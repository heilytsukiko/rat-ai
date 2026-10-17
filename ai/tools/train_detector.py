"""CPU transfer learning of a compact temporal detector, never Qwen fine-tuning."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["DO_NOT_TRACK"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def unique_videos(rows):
    strategies = {r.get("split_strategy", "exact_duplicates_v1") for r in rows}
    if len(strategies) != 1 or not strategies <= {"exact_duplicates_v1", "reviewed_sessions_v1"}:
        raise ValueError("Inconsistent or unknown dataset split strategy")
    if any(r.get("session_group") for r in rows) and strategies != {"reviewed_sessions_v1"}:
        raise ValueError("Session annotations require rebuilding with --group-by-session")
    if strategies == {"reviewed_sessions_v1"}:
        seen = {}
        for row in rows:
            if not row.get("session_group") or not row.get("split_group"):
                raise ValueError("Session manifest lacks grouping information")
            for name in ("session_group", "split_group"):
                key = (name, row[name])
                if key in seen and seen[key] != row["split"]:
                    raise ValueError("A recording session crosses dataset splits")
                seen[key] = row["split"]
    groups = {}
    for row in rows:
        group = groups.setdefault(row["sha256"], {**row, "event_times_sec": []})
        if group["split"] != row["split"] or group["label"] != row["label"]:
            raise ValueError("Duplicate file crosses splits or has conflicting labels")
        if row["event_time_sec"] is not None:
            group["event_times_sec"].append(row["event_time_sec"])
    result = list(groups.values())
    for group in result:
        group["event_times_sec"] = sorted(set(group["event_times_sec"]))
    if {r["split"] for r in result} != {"train", "validation"}:
        raise ValueError("Both training and validation splits are required")
    return result


def training_samples(videos, dense_negative_stride=None, dense_validation=False):
    import numpy as np
    samples = []
    for row in videos:
        if row["label"]:
            offsets = (-0.5, 0, 0.5) if row["split"] == "train" else (0,)
            centers = sorted({round(min(max(0, t + offset), row["duration_sec"] - 0.1), 4)
                              for t in row["event_times_sec"] for offset in offsets})
        elif dense_negative_stride is not None and (row["split"] == "train" or dense_validation):
            centers = np.arange(0.5, row["duration_sec"], dense_negative_stride).tolist()
        else:
            centers = np.linspace(0.5, row["duration_sec"] - 0.5, 4).tolist()
        for center in centers:
            samples.append({"video_id": row["video_id"], "sha256": row["sha256"],
                            "video_path": row["video_path"], "duration_sec": row["duration_sec"],
                            "center_sec": center, "label": row["label"], "split": row["split"]})
    return samples


def mil_samples(videos):
    """Full uniform bags; annotated auxiliary windows belong to training only."""
    import numpy as np
    samples = []
    for row in videos:
        centers = np.arange(min(0.5, row["duration_sec"] / 2), row["duration_sec"], 2).tolist()
        specs = [(center, True, False) for center in centers]
        if row["split"] == "train" and row["label"]:
            specs.extend((point, False, True) for point in row["event_times_sec"])
        for center, scan_window, point_supervised in specs:
            samples.append({"video_id": row["video_id"], "sha256": row["sha256"],
                            "video_path": row["video_path"], "duration_sec": row["duration_sec"],
                            "center_sec": center, "label": row["label"], "split": row["split"],
                            "scan_window": scan_window, "point_supervised": point_supervised})
    return samples


def video_groups(samples, split):
    """One contribution per unique video, independent of duration/window count."""
    groups = {}
    for index, sample in enumerate(samples):
        if sample["split"] != split:
            continue
        group = groups.setdefault(sample["sha256"], {"label": sample["label"], "indices": [],
                                                     "scan_positions": [], "point_positions": []})
        if group["label"] != sample["label"]:
            raise ValueError("Conflicting video labels")
        position = len(group["indices"])
        group["indices"].append(index)
        if sample.get("scan_window", True):
            group["scan_positions"].append(position)
        if sample.get("point_supervised", False):
            group["point_positions"].append(position)
    if {g["label"] for g in groups.values()} != {0, 1}:
        raise ValueError("Both video classes are required")
    return list(groups.values())


def video_loss(logits, label):
    """Known positive points; penalize worst false alarm plus negative coverage."""
    import torch
    import torch.nn.functional as F
    losses = F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, label), reduction="none")
    return losses.mean() if label else 0.75 * losses.max() + 0.25 * losses.mean()


def mil_video_loss(logits, label, scan_positions=None, point_positions=()):
    """Match max-window inference, plus a fixed 0.25 training-point auxiliary loss.

    A positive bag does not assign a positive label to every background window.
    Annotated auxiliary windows are excluded from the bag's deployment scan.
    """
    import torch.nn.functional as F
    scanned = logits if scan_positions is None else logits[scan_positions]
    if not len(scanned):
        raise ValueError("MIL video requires at least one uniform scan window")
    peak = scanned.max()
    loss = F.binary_cross_entropy_with_logits(peak, peak.new_tensor(float(label)))
    if label and len(point_positions):
        loss = loss + 0.25 * F.softplus(-logits[list(point_positions)]).mean()
    return loss


def group_loss(model, features, group, objective):
    logits = model.forward_cached(features[group["indices"]])
    if objective == "mil-video":
        return mil_video_loss(logits, group["label"], group["scan_positions"], group["point_positions"])
    return video_loss(logits, group["label"])


def balanced_video_loss(model, features, groups, objective="hard-negative-video"):
    import torch
    # Weight each class equally, then each video equally within its class.
    losses = [group_loss(model, features, g, objective) for g in groups]
    return torch.stack([torch.stack([loss for loss, group in zip(losses, groups)
                                     if group["label"] == label]).mean() for label in (0, 1)]).mean()


def binary_metrics(labels, scores, threshold=0.5):
    tp = sum(y == 1 and s >= threshold for y, s in zip(labels, scores))
    tn = sum(y == 0 and s < threshold for y, s in zip(labels, scores))
    fp = sum(y == 0 and s >= threshold for y, s in zip(labels, scores))
    fn = sum(y == 1 and s < threshold for y, s in zip(labels, scores))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": (tp + tn) / len(labels) if labels else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "threshold": threshold}


def cached_video_metrics(model, features, groups, threshold=0.5):
    """Diagnostic uniform-scan metrics; never include auxiliary annotation clips."""
    import torch
    import torch.nn.functional as F
    scores, labels, losses = [], [], []
    with torch.inference_mode():
        for group in groups:
            selected = [group["indices"][position] for position in group["scan_positions"]]
            logits = model.forward_cached(features[selected].float())
            peak = logits.max()
            labels.append(group["label"])
            scores.append(float(peak.sigmoid()))
            losses.append(float(F.binary_cross_entropy_with_logits(peak, peak.new_tensor(float(group["label"])))))
    metrics = binary_metrics(labels, scores, threshold)
    positive = [score for score, label in zip(scores, labels) if label]
    negative = [score for score, label in zip(scores, labels) if not label]
    metrics["roc_auc"] = sum((p > n) + .5 * (p == n) for p in positive for n in negative) / (
        len(positive) * len(negative))
    balanced_loss = sum(sum(loss for loss, y in zip(losses, labels) if y == label) / labels.count(label)
                        for label in (0, 1)) / 2
    return metrics, balanced_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("ai/local/dataset/manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("ai/local/training/mobilenet-temporal-v1"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dense-negative-stride-sec", type=float)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--objective", choices=("window", "hard-negative-video", "mil-video"), default="window")
    parser.add_argument("--backbone", choices=("mobilenet", "ssdlite-coco"), default="mobilenet")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100 or not 1 <= args.threads <= 8:
        parser.error("epochs must be 1–100 and threads 1–8")
    if args.dense_negative_stride_sec is not None and not 0.5 <= args.dense_negative_stride_sec <= 30:
        parser.error("negative scan stride must be between 0.5 and 30 seconds")
    if args.objective == "hard-negative-video" and args.dense_negative_stride_sec != 2:
        parser.error("hard-negative-video requires --dense-negative-stride-sec 2 to match inference")
    import numpy as np
    import psutil
    import torch
    import torch.nn.functional as F
    import torchvision
    from torchvision.models import MobileNet_V3_Small_Weights
    from rat_ai.temporal import TemporalDetector
    from rat_ai.video import clip_tensor
    from rat_ai.inference import score_video
    architecture = "mobilenet_v3_small_temporal_v1"
    model_class = TemporalDetector
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    if args.backbone == "ssdlite-coco":
        from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
        from rat_ai.semantic import SemanticTemporalDetector
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        model_class = SemanticTemporalDetector
        architecture = "ssdlite_coco_temporal_v1"

    if psutil.virtual_memory().available < 1024 ** 3:
        raise RuntimeError("Need at least 1 GiB available RAM for this small CPU training run")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    state = {"status": "running", "stage": "preparing", "training_started": False,
             "model": architecture, "qwen_weights_modified": False}

    def update(**fields):
        state.update(fields)
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = args.output / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(args.output / "status.json")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    try:
        update(stage="downloading_pretrained_backbone")
        pretrained_dir = args.output.parent / "pretrained"
        base_state = torch.hub.load_state_dict_from_url(
            weights.url, model_dir=str(pretrained_dir), map_location="cpu",
            check_hash=True, weights_only=True)
        weight_path = pretrained_dir / Path(weights.url).name
        with weight_path.open("rb") as stream:
            weights_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        model = model_class()
        model.load_pretrained_backbone(base_state)
        del base_state
        warm_start_sha = None
        if args.warm_start:
            payload = torch.load(args.warm_start, map_location="cpu", weights_only=True)
            if payload["config"]["architecture"] != architecture:
                raise ValueError("Incompatible warm-start model")
            model.load_state_dict(payload["state_dict"], strict=True)
            with args.warm_start.open("rb") as stream:
                warm_start_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        initial_tail = {k: v.detach().clone() for k, v in model.tail.named_parameters()}
        rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
        videos = unique_videos(rows)
        # Detect source changes before caching features from restricted source files.
        for row in videos:
            with Path(row["video_path"]).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                    raise ValueError("A dataset file changed since manifest preparation")
        samples = (mil_samples(videos) if args.objective == "mil-video" else
                   training_samples(videos, args.dense_negative_stride_sec,
                                    dense_validation=args.objective == "hard-negative-video"))
        config = {"architecture": architecture, "input_size": getattr(model, "input_size", 224),
                  "frames_per_clip": 3, "clip_span_sec": 2, "seed": 42,
                  "pretrained_url": weights.url, "pretrained_sha256": weights_sha,
                  "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                  "frozen_prefix": True, "trainable_tail": True,
                  "preprocessing": "RGB letterbox mean padding, ImageNet normalization",
                  "sample_spec": samples, "validation_scan_stride_sec": 2.0,
                  "threshold": 0.5}
        config["objective"] = args.objective
        config["trainable_tail_kind"] = "object_feature_projection" if args.backbone == "ssdlite-coco" else "visual_convolution"
        config["warm_start_sha256"] = warm_start_sha
        config["dense_training_negative_stride_sec"] = args.dense_negative_stride_sec
        signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        cache_path = args.output / "prefix_features.pt"
        cached = torch.load(cache_path, weights_only=True, map_location="cpu") if cache_path.exists() else None
        if cached and cached["signature"] == signature:
            features = cached["features"]
        else:
            update(stage="extracting_frozen_prefix_features", samples=len(samples))
            model.eval()
            features_list = []
            with torch.inference_mode():
                for i, sample in enumerate(samples):
                    clip, _ = clip_tensor(sample["video_path"], sample["center_sec"], sample["duration_sec"], size=config["input_size"])
                    features_list.append(model.prefix(clip).half())
                    if (i + 1) % 10 == 0 or i + 1 == len(samples):
                        update(samples_cached=i + 1)
                        print(json.dumps({"stage": "feature_cache", "completed": i + 1,
                                          "total": len(samples)}), flush=True)
            features = torch.stack(features_list)
            torch.save({"signature": signature, "features": features}, cache_path)
        # Training tensors are ordinary tensors, not inference-mode tensors.
        features = features.float().clone()
        labels = torch.tensor([s["label"] for s in samples], dtype=torch.float32)
        train_index = torch.tensor([i for i, s in enumerate(samples) if s["split"] == "train"])
        val_index = torch.tensor([i for i, s in enumerate(samples) if s["split"] == "validation"])
        for index in (train_index, val_index):
            if set(labels[index].tolist()) != {0, 1}:
                raise ValueError("Each split must contain both labels")
        train_groups = video_groups(samples, "train")
        val_groups = video_groups(samples, "validation")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        update(stage="training", training_started=True, trainable_parameters=trainable)
        optimizer = torch.optim.AdamW([
            {"params": model.tail.parameters(), "lr": 0.001 if args.backbone == "ssdlite-coco" else 0.0001},
            {"params": model.head.parameters(), "lr": 0.003}], weight_decay=0.01)
        train_y = labels[train_index]
        pos_weight = (len(train_y) - train_y.sum()) / train_y.sum()
        history, best_loss, best_epoch = [], float("inf"), None
        best_state = None
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            if args.objective in {"hard-negative-video", "mil-video"}:
                # Accumulate a video at a time to bound memory even for long clips.
                # Balanced class weights use all videos and never oversample validation.
                order = torch.randperm(len(train_groups)).tolist()
                counts = {label: sum(g["label"] == label for g in train_groups) for label in (0, 1)}
                for start in range(0, len(order), 4):
                    optimizer.zero_grad(set_to_none=True)
                    batch = order[start:start + 4]
                    for index in batch:
                        group = train_groups[index]
                        loss = group_loss(model, features, group, args.objective)
                        weight = len(train_groups) / (2 * counts[group["label"]] * len(batch))
                        (loss * weight).backward()
                        losses.append((float(loss.detach()) * len(train_groups) / (2 * counts[group["label"]]), 1))
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5)
                    optimizer.step()
            else:
                order = train_index[torch.randperm(len(train_index))]
                for indices in order.split(16):
                    optimizer.zero_grad(set_to_none=True)
                    logits = model.forward_cached(features[indices])
                    loss = F.binary_cross_entropy_with_logits(logits, labels[indices], pos_weight=pos_weight)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5)
                    optimizer.step()
                    losses.append((float(loss.detach()), len(indices)))
            model.eval()
            with torch.inference_mode():
                if args.objective in {"hard-negative-video", "mil-video"}:
                    val_loss = float(balanced_video_loss(model, features, val_groups, args.objective))
                else:
                    val_logits = model.forward_cached(features[val_index])
                    val_losses = F.binary_cross_entropy_with_logits(val_logits, labels[val_index], reduction="none")
                    val_y = labels[val_index]
                    val_loss = float((val_losses[val_y == 0].mean() + val_losses[val_y == 1].mean()) / 2)
            entry = {"epoch": epoch, "train_loss": sum(x * n for x, n in losses) / sum(n for _, n in losses),
                     "validation_balanced_loss": val_loss}
            if args.objective == "mil-video":
                entry["train_video_metrics"], entry["train_video_balanced_loss"] = cached_video_metrics(
                    model, features, train_groups)
                entry["validation_video_metrics"], _ = cached_video_metrics(model, features, val_groups)
            history.append(entry)
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                best_state = copy.deepcopy(model.state_dict())
                torch.save({"state_dict": best_state, "config": config, "epoch": epoch},
                           args.output / "best_detector.pt")
            update(epoch=epoch, max_epochs=args.epochs, **{k: v for k, v in entry.items() if k != "epoch"})
            print(json.dumps(entry), flush=True)
        model.load_state_dict(best_state)
        model.eval()
        tail_change = sum(float((p.detach() - initial_tail[k]).square().sum())
                          for k, p in model.tail.named_parameters()) ** 0.5
        if tail_change <= 0:
            raise RuntimeError("Backbone tail did not change; refusing to report fine-tuning")
        update(stage="evaluating_full_validation_videos", best_epoch=best_epoch)
        results = []
        with torch.inference_mode():
            for row in (r for r in videos if r["split"] == "validation"):
                # Uniform scan only: validation event timestamps never choose input windows here.
                scored = score_video(model, row["video_path"], row["duration_sec"], stride_sec=2)
                results.append({"video_id": row["video_id"], "label": row["label"],
                                "annotated_event_times_sec": row["event_times_sec"],
                                **scored})
                if row["label"]:
                    results[-1]["peak_abs_error_sec"] = min(
                        abs(scored["peak_time_sec"] - t) for t in row["event_times_sec"])
                print(json.dumps({"stage": "validation_video", "completed": len(results),
                                  "processing_sec": results[-1]["processing_sec"]}), flush=True)
        report = {
            "model": config["architecture"], "qwen_weights_modified": False,
            "objective": args.objective,
            "training_completed": True, "best_epoch": best_epoch, "epochs": args.epochs,
            "train_unique_videos": sum(r["split"] == "train" for r in videos),
            "validation_unique_videos": len(results),
            "training_windows": len(train_index), "validation_supervised_windows": len(val_index),
            "trainable_parameters": trainable, "tail_parameter_change_l2": tail_change,
            "validation_video_metrics": binary_metrics([r["label"] for r in results], [r["video_score"] for r in results]),
            "validation_video_results": results, "history": history,
            "positive_peak_mean_abs_error_sec": float(np.mean(
                [r["peak_abs_error_sec"] for r in results if r["label"]])),
            "total_elapsed_sec": time.perf_counter() - started,
            "peak_process_rss_bytes": getattr(psutil.Process().memory_info(), "peak_wset",
                                               psutil.Process().memory_info().rss),
            "limitations": ["Only eight validation videos; no independent test set.",
                            "Related filming sessions are not yet grouped; exact copies are grouped.",
                            "Positive labels are point annotations; category and event duration unknown.",
                            "Binary detector only; no legal source or violation type classification.",
                            ("Qwen and the COCO object detector were not fine-tuned; temporal projection and classifier were trained."
                             if args.backbone == "ssdlite-coco" else
                             "Qwen was not fine-tuned; MobileNet tail and temporal head were trained."),
                            ("Checkpoint selected by full-video validation loss; point annotations used only for training auxiliary windows and final localization metrics."
                             if args.objective == "mil-video" else
                             "Checkpoint selected using supervised validation windows; metrics are validation, not test.")]}
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        update(status="done", stage="complete", best_epoch=best_epoch,
               validation_video_metrics=report["validation_video_metrics"], tail_parameter_change_l2=tail_change)
        print(json.dumps({k: report[k] for k in ["training_completed", "best_epoch", "train_unique_videos",
              "validation_unique_videos", "trainable_parameters", "tail_parameter_change_l2",
              "validation_video_metrics", "total_elapsed_sec"]}), flush=True)
    except Exception as error:
        update(status="failed", reason=str(error))
        raise


if __name__ == "__main__":
    main()
