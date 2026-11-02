"""Fully decode every review output and check its declared frame/time mapping."""

import hashlib
import json
import pathlib
import collections

import av

from scripts.preprocess import DATA_ROOT

CHECK = DATA_ROOT / "agibot_world/check"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inspect_source_fields(manifest):
    raw = pathlib.Path(manifest["raw_root"]) / "files"
    rows = []
    for sample in manifest["samples"]:
        info = json.loads((raw / sample["info_path"]).read_text())
        eid = str(sample["episode_index"])
        segments = info["instruction_segments"].get(eid, [])
        aff = info.get("affordance_annotations", {}).get("episodes", {}).get(eid, {})
        aseg = {s["seg_idx"]: s for s in aff.get("segments", [])}
        frames = [f for s in aseg.values() for f in s.get("frames", [])]
        endpoints = []
        for j, segment in enumerate(segments):
            frame = next((f for f in aseg.get(j, {}).get("frames", []) if f["frame_index"] == segment["end_frame_index"] - 1), None)
            if frame and frame.get("modalities", {}).get("bbox", {}).get("found"):
                endpoints.append(j)
        rows.append({"sample_number": sample["sample_number"], "task": sample["task"], "episode_index": sample["episode_index"], "frames": sample["expected_frames"], "segments": len(segments), "raw_skills": dict(collections.Counter(s.get("skill", "") for s in segments)), "tracks": dict(collections.Counter(s.get("track", "") for s in segments)), "source_annotated_frame_records": len(frames), "bbox_found_records": sum(f.get("modalities", {}).get("bbox", {}).get("found") is True for f in frames), "segments_with_bbox_at_end_minus_1": endpoints, "source_interval_violations": sum(not (0 <= s["start_frame_index"] < s["end_frame_index"] <= sample["expected_frames"]) for s in segments)})
    result = {"scope": "10 selected episodes only; end-1 convention is provisional and bbox identity not verified", "date": "2026-10-30", "requirements_sha256": manifest["requirements_sha256"], "script_sha256": digest(pathlib.Path(__file__)), "samples": rows}
    (CHECK / "sample_fields.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def main():
    manifest = json.loads((CHECK / "manifest.json").read_text())
    inspect_source_fields(manifest)
    report = json.loads((CHECK / "technical_check.json").read_text())
    rows = {r["sample_number"]: r for r in report["results"]}
    if len(rows) != 10 or report["errors"]:
        raise RuntimeError("Rendering has not completed all ten samples")
    checks = []
    for sample in manifest["samples"]:
        path = CHECK / sample["output_video"]
        count = 0
        first_time, last_time = None, None
        max_error = 0.0
        geometry_ok = True
        with av.open(str(path)) as video:
            stream = video.streams.video[0]
            stream.codec_context.thread_count = 2
            for frame in video.decode(stream):
                geometry_ok &= (frame.width, frame.height) == (manifest["render"]["width"], manifest["render"]["height"])
                current_time = float(frame.pts * frame.time_base)
                if first_time is None:
                    first_time = current_time
                if last_time is not None and current_time <= last_time:
                    raise RuntimeError("Non-monotonic output timestamps")
                max_error = max(max_error, abs(current_time - count / sample["fps"]))
                last_time = current_time
                count += 1
        checksum = digest(path)
        passed = bool(count == sample["expected_frames"] and geometry_ok and max_error < 1e-5 and checksum == rows[sample["sample_number"]]["output_sha256"])
        result = {"sample_number": sample["sample_number"], "decoded_frames": count, "first_time_s": first_time, "last_time_s": last_time, "duration_s": count / sample["fps"], "geometry_matches_manifest": bool(geometry_ok), "timestamp_grid_max_error_s": max_error, "hash_matches_render_record": checksum == rows[sample["sample_number"]]["output_sha256"], "pass": passed}
        checks.append(result)
        print(json.dumps(result), flush=True)
    report["full_output_decode_check"] = {"script_sha256": digest(pathlib.Path(__file__)), "checks": checks, "all_pass": all(c["pass"] for c in checks)}
    # The final batch was rendered after the preview layout was adjusted.
    report["render_script_sha256"] = digest(pathlib.Path(__file__).with_name("render_check.py"))
    (CHECK / "technical_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not report["full_output_decode_check"]["all_pass"]:
        raise SystemExit("One or more review videos failed verification")


if __name__ == "__main__":
    main()
