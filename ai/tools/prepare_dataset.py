"""Audit the supplied MOV/XLSX dataset locally; never upload or decode frames.

Run from the repository root:
  python ai/tools/prepare_dataset.py --data-root PATH --output ai/local/dataset
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import struct


def video_id(value: str) -> str:
    """Normalize only the known Latin/Cyrillic X naming difference."""
    name = Path(str(value).strip()).stem.upper().replace("Х", "X")
    if not re.fullmatch(r"X-\d+", name):
        raise ValueError(f"Unexpected video identifier: {value!r}")
    return name


def timestamp_seconds(value) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = float(value)
    else:
        text = str(value).strip().lower().replace(",", ".")
        match = re.fullmatch(
            r"(?:(\d+)\s*мин(?:ута|уты|ут)?\s*)?"
            r"(?:(\d+(?:\.\d+)?)\s*сек(?:унда|унды|унд)?\s*)?", text
        )
        if not match or not any(match.groups()):
            raise ValueError(f"Unsupported timestamp: {value!r}")
        seconds = float(match[1] or 0) * 60 + float(match[2] or 0)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"Invalid timestamp: {value!r}")
    return seconds


def atoms(stream, start, end):
    """Iterate bounded QuickTime/ISO BMFF boxes without reading video payload."""
    offset = start
    while offset < end:
        if end - offset < 8:
            raise ValueError("Truncated MOV box header")
        stream.seek(offset)
        size, kind = struct.unpack(">I4s", stream.read(8))
        header_size = 8
        if size == 1:
            extra = stream.read(8)
            if len(extra) != 8:
                raise ValueError("Truncated MOV extended box")
            size = struct.unpack(">Q", extra)[0]
            header_size = 16
        elif size == 0:
            size = end - offset
        if size < header_size or offset + size > end:
            raise ValueError("Invalid MOV box size")
        yield kind, offset + header_size, offset + size
        offset += size


def movie_metadata(path: Path) -> dict:
    """Read movie duration and video-track dimensions; not a decode check."""
    duration = None
    dimensions = []
    with path.open("rb") as stream:
        for kind, begin, end in atoms(stream, 0, path.stat().st_size):
            if kind != b"moov":
                continue
            for child, cbegin, cend in atoms(stream, begin, end):
                if child == b"mvhd":
                    stream.seek(cbegin)
                    header = stream.read(min(32, cend - cbegin))
                    if not header or header[0] not in (0, 1):
                        raise ValueError("Unsupported movie header version")
                    offset = 20 if header[0] == 1 else 12
                    fmt = ">IQ" if header[0] == 1 else ">II"
                    scale, ticks = struct.unpack_from(fmt, header, offset)
                    if not scale:
                        raise ValueError("Zero movie timescale")
                    duration = ticks / scale
                elif child == b"trak":
                    track_dimensions = None
                    is_video = False
                    for item, ibegin, iend in atoms(stream, cbegin, cend):
                        if item == b"tkhd":
                            stream.seek(iend - 8)
                            width, height = struct.unpack(">II", stream.read(8))
                            track_dimensions = [width / 65536, height / 65536]
                        elif item == b"mdia":
                            for media, mbegin, mend in atoms(stream, ibegin, iend):
                                if media == b"hdlr":
                                    stream.seek(mbegin + 8)
                                    is_video = stream.read(4) == b"vide"
                    if is_video and track_dimensions:
                        dimensions.append(track_dimensions)
    if duration is None or not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"No valid movie duration in {path.name}")
    if len(dimensions) != 1:
        raise ValueError(f"Expected one video track in {path.name}")
    return {"duration_sec": duration, "track_width": dimensions[0][0],
            "track_height": dimensions[0][1]}


def assign_splits(rows: list[dict], seed: int = 42) -> None:
    """Stratify by video label; exact duplicate files share a split."""
    rng = random.Random(seed)
    groups = {}
    for row in rows:
        groups.setdefault(row["sha256"], []).append(row)
    for group in groups.values():
        if len({r["label"] for r in group}) != 1:
            raise ValueError("Identical video files have conflicting labels")
    for label in (0, 1):
        selected = sorted(
            [g for g in groups.values() if g[0]["label"] == label],
            key=lambda g: g[0]["video_id"],
        )
        if len(selected) < 2:
            raise ValueError("Need at least two distinct videos in each class")
        rng.shuffle(selected)
        n_validation = max(1, round(len(selected) * 0.2))
        for i, group in enumerate(selected):
            for row in group:
                row["split"] = "validation" if i < n_validation else "train"


def apply_review(rows: list[dict], review_path: Path) -> list[dict]:
    """Import human annotations; source labels and timestamps remain authoritative."""
    indexed = {r["video_id"]: r for r in rows}
    reviewed = {}
    fields = ("event_type", "observed_action", "rule_id", "session_group", "review_status")
    with review_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"video_id", "label", "event_time_sec", *fields}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Review CSV is missing required columns")
        for record in reader:
            if None in record or any(record.get(field) is None for field in required):
                raise ValueError("Malformed review CSV row")
            key = video_id(record["video_id"])
            if key not in indexed or key in reviewed:
                raise ValueError("Review CSV has unknown or duplicate video IDs")
            original = indexed[key]
            if record["label"].strip() != str(original["label"]):
                raise ValueError("Review must not change source labels")
            raw_timestamp = record["event_time_sec"].strip()
            timestamp = timestamp_seconds(float(raw_timestamp.replace(",", "."))) if raw_timestamp else None
            if timestamp != original["event_time_sec"]:
                raise ValueError("Review must not change source timestamps")
            annotation = {field: (record.get(field) or "").strip() for field in fields}
            if annotation["review_status"] not in {"pending", "reviewed", "needs_review"}:
                raise ValueError("Unknown review status")
            if annotation["review_status"] == "reviewed":
                necessary = ("session_group", "event_type", "observed_action", "rule_id") if original["label"] else ("session_group",)
                if any(not annotation[f] for f in necessary):
                    raise ValueError("Reviewed row lacks required human annotations")
            reviewed[key] = {**original, **annotation}
    if reviewed.keys() != indexed.keys():
        raise ValueError("Review CSV must cover every source video exactly once")
    return [reviewed[r["video_id"]] for r in rows]


def assign_session_splits(rows: list[dict], seed: int = 42) -> None:
    """Keep transitive session/exact-copy components together, including mixed labels.

    Exact subset selection targets 20% of unique files per class on this small
    dataset. Source row order and duplicate annotation counts cannot affect it.
    """
    if not rows or any(not str(r.get("session_group") or "").strip() for r in rows):
        raise ValueError("Session splitting requires a session_group for every video")
    unique = {}
    for row in rows:
        if row["label"] not in (0, 1):
            raise ValueError("Only binary labels are supported")
        if row["sha256"] in unique and unique[row["sha256"]]["label"] != row["label"]:
            raise ValueError("Identical video files have conflicting labels")
        unique[row["sha256"]] = row
    if len(unique) > 500:
        raise ValueError("Exact grouped split supports up to 500 unique files")
    parent = {digest: digest for digest in unique}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    sessions = {}
    for row in rows:
        digest = row["sha256"]
        session = str(row["session_group"]).strip()
        if session in sessions:
            a, b = find(digest), find(sessions[session])
            parent[max(a, b)] = min(a, b)
        sessions[session] = digest
    components = {}
    for digest in sorted(unique):
        components.setdefault(find(digest), []).append(digest)
    groups = sorted(components.values())
    random.Random(seed).shuffle(groups)
    totals = tuple(sum(r["label"] == label for r in unique.values()) for label in (0, 1))
    if min(totals) < 2:
        raise ValueError("Need two unique files per class for grouped splitting")
    # Each DP entry stores one whole-component subset with those class counts.
    subsets = {(0, 0): ()}
    for index, group in enumerate(groups):
        counts = tuple(sum(unique[d]["label"] == label for d in group) for label in (0, 1))
        for (negative, positive), subset in list(subsets.items()):
            pair = (negative + counts[0], positive + counts[1])
            if pair[0] < totals[0] and pair[1] < totals[1]:
                subsets.setdefault(pair, (*subset, index))
    candidates = [(pair, subset) for pair, subset in subsets.items() if min(pair) > 0]
    if not candidates:
        raise ValueError("Cannot split whole sessions with both classes in both splits")
    target = tuple(max(1, round(total * 0.2)) for total in totals)
    _, selected = min(candidates, key=lambda item: (
        sum(((item[0][label] - target[label]) / totals[label]) ** 2 for label in (0, 1)),
        abs(sum(item[0]) - sum(target)), item[1]))
    assignments = {}
    for index, group in enumerate(groups):
        group_id = hashlib.sha256("\n".join(group).encode()).hexdigest()
        for digest in group:
            assignments[digest] = {"split": "validation" if index in selected else "train",
                                   "split_group": group_id, "split_strategy": "reviewed_sessions_v1"}
    for row in rows:
        row.update(assignments[row["sha256"]])


def prepare(root: Path) -> list[dict]:
    import openpyxl

    root = root.resolve()
    if not (root / "open_labels.xlsx").is_file() and (root / "open").is_dir():
        root = root / "open"
    files = {}
    for path in sorted((root / "videos").iterdir()):
        if path.suffix.lower() != ".mov":
            continue
        key = video_id(path.stem)
        if key in files:
            raise ValueError(f"Ambiguous video identifier: {key}")
        files[key] = path
    workbook = openpyxl.load_workbook(root / "open_labels.xlsx", read_only=True,
                                      data_only=True)
    result = []
    seen = set()
    try:
        sheet = workbook["labels"]
        values = iter(sheet.values)
        expected = ("наименование файла", "таймстемп", "нарушение / без нарушения")
        if tuple(next(values)) != expected:
            raise ValueError("Unexpected spreadsheet columns")
        for row_number, cells in enumerate(values, start=2):
            if not any(value is not None for value in cells):
                continue
            name, raw_timestamp, raw_label = cells
            key = video_id(name)
            if key in seen:
                raise ValueError(f"Duplicate spreadsheet row: {key}")
            seen.add(key)
            path = files.get(key)
            if path is None:
                raise ValueError(f"Missing video: {key}")
            labels = {"нарушение": 1, "без нарушения": 0}
            if str(raw_label).strip().lower() not in labels:
                raise ValueError(f"Unknown label on row {row_number}")
            label = labels[str(raw_label).strip().lower()]
            timestamp = timestamp_seconds(raw_timestamp)
            meta = movie_metadata(path)
            if label == 1 and timestamp is None:
                raise ValueError(f"Positive video lacks timestamp: {key}")
            if label == 0 and timestamp is not None:
                raise ValueError(f"Negative video has event timestamp: {key}")
            if timestamp is not None and timestamp >= meta["duration_sec"]:
                raise ValueError(f"Timestamp outside video: {key}")
            with path.open("rb") as video_file:
                digest = hashlib.file_digest(video_file, "sha256").hexdigest()
            result.append({"video_id": key, "video_path": str(path),
                           "label": label, "event_time_sec": timestamp,
                           "event_type": None, "source_row": row_number,
                           "bytes": path.stat().st_size, "sha256": digest, **meta})
    finally:
        workbook.close()
    if set(files) != seen:
        raise ValueError(f"Unlabelled videos: {sorted(set(files) - seen)}")
    assign_splits(result)
    return result


def write_outputs(rows: list[dict], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    durations = [r["duration_sec"] for r in rows]
    duplicate_groups = {}
    for row in rows:
        duplicate_groups.setdefault(row["sha256"], []).append(row)
    summary = {
        "videos": len(rows), "positive": sum(r["label"] for r in rows),
        "negative": sum(1 - r["label"] for r in rows),
        "duration_min_sec": min(durations), "duration_max_sec": max(durations),
        "total_duration_sec": sum(durations),
        "total_bytes": sum(r["bytes"] for r in rows),
        "unique_files": len({r["sha256"] for r in rows}),
        "duplicate_groups": [
            [{"video_id": r["video_id"], "event_time_sec": r["event_time_sec"]}
             for r in group]
            for group in duplicate_groups.values() if len(group) > 1
        ],
        "splits": dict(Counter(f'{r["split"]}_{r["label"]}' for r in rows)),
        "track_dimensions": sorted({f'{r["track_width"]:g}x{r["track_height"]:g}' for r in rows}),
        "seed": 42,
        "split_strategy": rows[0].get("split_strategy", "exact_duplicates_v1"),
        "session_groups": len({r["split_group"] for r in rows}) if all(r.get("split_group") for r in rows) else None,
        "reviewed_rows": sum(r.get("review_status") == "reviewed" for r in rows),
        "limitations": ["Container metadata only; frames not decoded or inspected.",
                        ("Whole reviewed session components share a split; grouping accuracy is human supplied."
                         if rows[0].get("split_strategy") == "reviewed_sessions_v1" else
                         "Exact duplicates checked; related takes not yet grouped."),
                        "Human event types and rule IDs, if supplied, are not automatically verified.",
                        "Event interval boundaries are not annotated.",
                        "Different timestamps on identical files need manual review; both retained.",
                        "Split quality depends on complete, correct recording-session review."]
    }
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Keep manual annotations separate: rerunning the audit must not erase them.
    review_path = output / "event_review.csv"
    if not review_path.exists():
        fields = ["video_id", "label", "event_time_sec", "event_type",
                  "observed_action", "rule_id", "session_group", "review_status"]
        with review_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({**{f: row.get(f, "") for f in fields},
                                 "review_status": "pending"})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ai/local/dataset"))
    parser.add_argument("--review", type=Path, help="Import the human-reviewed event_review.csv")
    parser.add_argument("--group-by-session", action="store_true",
                        help="Require all session groups and keep them intact across splits")
    args = parser.parse_args()
    if args.group_by_session and args.review is None:
        parser.error("--group-by-session requires --review")
    rows = prepare(args.data_root)
    if args.review:
        rows = apply_review(rows, args.review)
    if args.group_by_session:
        assign_session_splits(rows)
    summary = write_outputs(rows, args.output)
    # Keep IDs, source annotations and local paths out of routine tool output.
    print(json.dumps({k: summary[k] for k in ("videos", "unique_files", "positive", "negative",
                                              "splits", "split_strategy", "session_groups", "reviewed_rows")}, indent=2))


if __name__ == "__main__":
    main()
