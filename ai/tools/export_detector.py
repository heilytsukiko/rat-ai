"""Export tensors and an allowlisted public manifest; omit dataset annotations/paths."""
import argparse
import hashlib
import json
from pathlib import Path


def export_checkpoint(source, output, candidate_policy=None):
    import torch
    from safetensors.torch import save_file
    from rat_ai.inference import validate_candidate_policy

    payload = torch.load(source, map_location="cpu", weights_only=True)
    config = payload["config"]
    input_sizes = {"mobilenet_v3_small_temporal_v1": 224, "ssdlite_coco_temporal_v1": 320}
    if config["architecture"] not in input_sizes:
        raise ValueError("Unsupported checkpoint architecture")
    policy = validate_candidate_policy(config.get("candidate_policy") if candidate_policy is None else candidate_policy)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tensors = {name: tensor.detach().cpu().contiguous() for name, tensor in payload["state_dict"].items()}
    save_file(tensors, str(output), metadata={"architecture": config["architecture"], "format": "pt"})
    with output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    # Never copy the complete training config: it contains restricted sample_spec.
    manifest = {
        "schema_version": 1, "architecture": config["architecture"],
        "input_size": input_sizes[config["architecture"]], "frames_per_clip": 3, "clip_span_sec": 2,
        "preprocessing": config["preprocessing"], "threshold": config["threshold"],
        "weights_sha256": digest, "best_epoch": int(payload["epoch"]),
        "base_weights_url": config["pretrained_url"],
        "base_weights_sha256": config["pretrained_sha256"],
        "experimental": True,
        "candidate_policy": policy,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("ai/local/training/mobilenet-temporal-v2/best_detector.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("ai/rat_ai/artifacts/detector-v2.safetensors"))
    parser.add_argument("--min-windows", type=int, help="Override confirmation policy together with --representative")
    parser.add_argument("--representative", choices=("peak", "center"))
    args = parser.parse_args()
    if (args.min_windows is None) != (args.representative is None):
        parser.error("Set both --min-windows and --representative")
    policy = None if args.min_windows is None else {"min_windows": args.min_windows, "representative": args.representative}
    manifest = export_checkpoint(args.checkpoint, args.output, policy)
    print(json.dumps({"exported": str(args.output), "sha256": manifest["weights_sha256"]}))


if __name__ == "__main__":
    main()
