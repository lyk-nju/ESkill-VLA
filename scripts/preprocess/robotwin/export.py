"""Explicit transition mapping and LeRobot export, retaining auxiliary source links."""
import argparse
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .annotate import (build_annotation, event_at, region_at, remaining_path,
                      validate_annotation)
from .source import (digest, load_episode, read_json, validate_episode, verify_entry,
                    write_json)


def _matrix(values):
    return np.asarray(values).reshape(len(values), -1)


def transitions(episode, state_source, action_source):
    """Align once, without changing the meaning of historical stored actions."""
    if episode.entry["format"] == "xpolicylab-v1":
        if (state_source, action_source) != ("stored", "stored"):
            raise ValueError("Historical files require explicit stored state/action semantics")
        def pack(group):
            return np.concatenate([_matrix(episode.signals[f"{group}/{arm}_{kind}"])
                                   for arm in ("left", "right")
                                   for kind in ("arm_joint_states", "ee_joint_states")], axis=1)
        state, action = pack("state"), pack("action")
    else:
        if action_source != "next-command":
            raise ValueError("Raw v2 action mapping must explicitly select next-command")
        command = np.concatenate([_matrix(episode.signals[f"joint_action/{arm}_{kind}"])
                                  for arm in ("left", "right") for kind in ("arm", "gripper")], axis=1)
        if state_source == "command":
            state = command[:-1]
        elif state_source == "measured-arm-command-gripper":
            state = np.concatenate([_matrix(episode.signals[f"joint_state/{arm}_{kind}"])
                                    for arm in ("left", "right")
                                    for kind in ("arm", "gripper_command")], axis=1)[:-1]
        else:
            raise ValueError("Raw v2 requires command or measured-arm-command-gripper state")
        action = command[1:]
    if state.shape != action.shape or not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("State/action dimensions or finite values invalid")
    return state.astype(np.float32), action.astype(np.float32)


def auxiliary_sample(episode, annotation, frame, camera="cam_head", path_points=16):
    """Project supervision references, with missing targets explicit (no model padding)."""
    result = {}
    for arm in ("left", "right"):
        event = event_at(annotation, arm, frame)
        item = {"event": event, "gripper": region_at(annotation, arm, camera, frame, "gripper"),
                "object": None, "goal": None, "goal_frame": None, "path": None}
        if event:
            item["object"] = region_at(annotation, arm, camera, frame, "object", event.get("object"))
            item["goal_frame"] = event["end_frame"]
            item["goal"] = region_at(annotation, arm, camera, event["end_frame"], "object", event.get("object"))
            item["path"] = remaining_path(episode, event, frame, path_points)
        result[arm] = item
    return result


def export_dataset(manifest_path, output, *, annotations, repo_id, env_cfg,
                   state_source, action_source, use_videos=True, image_size=(240, 320)):
    # Heavy dependencies are loaded only for export.
    from XPolicyLab.utils.process_data import get_robot_action_dim_info
    from XPolicyLab.scripts.transform_lerobot_v30_format import (
        DEFAULT_DATASET_CONFIG, CAMERA_CANDIDATES, create_empty_dataset,
        configure_video_encoding, finalize_dataset, _build_motor_names_from_dims,
    )
    manifest = read_json(manifest_path)
    entries = manifest["samples"]
    if not entries:
        raise ValueError("Empty input manifest")
    identities = [(e["task"], e["episode"]) for e in entries]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate source episode in export manifest")
    if any(e.get("split") not in ("train", "val", "test") for e in entries):
        raise ValueError("Assign each source episode an explicit train/val/test split first")
    dims = get_robot_action_dim_info(env_cfg)
    arm_dims = [a + g for a, g in zip(dims["arm_dim"], dims["ee_dim"])]
    motors = _build_motor_names_from_dims(arm_dims)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    stage = output.with_name(f".{output.name}.building")
    stage.mkdir(parents=True)  # An interrupted build stays separate and inspectable.
    dataset = None
    report = {"input_manifest": str(Path(manifest_path).resolve()),
              "input_sha256": digest(manifest_path), "state_source": state_source,
              "action_source": action_source, "env_cfg": env_cfg,
              "image_size": list(image_size), "regions_coordinate_space": "source_rgb",
              "raw_source_required_for_auxiliary_rgb": True,
              "manual_review": "pending", "episodes": [], "complete": False}
    try:
        frequency = None
        for index, entry in enumerate(entries):
            verify_entry(entry)
            episode = load_episode(entry)
            checks = validate_episode(episode)
            if checks["errors"]:
                raise ValueError(checks["errors"])
            state, action = transitions(episode, state_source, action_source)
            if state.shape[1] != len(motors):
                raise ValueError("Source action dimensions differ from robot registry")
            annotation_path = Path(annotations) / entry["task"] / f"{entry['episode']}.json" if annotations else None
            annotation = read_json(annotation_path) if annotation_path else build_annotation(episode)
            validate_annotation(episode, annotation)
            if frequency is None:
                frequency = episode.fps
                if frequency != int(frequency):
                    raise ValueError("This LeRobot export requires integer fps; no implicit resampling")
                config = replace(DEFAULT_DATASET_CONFIG, use_videos=use_videos,
                                 video_backend="pyav", streaming_encoding=use_videos,
                                 image_writer_processes=0, image_writer_threads=1)
                dataset = create_empty_dataset(repo_id, "robotwin", motors, int(frequency),
                                               mode="video" if use_videos else "image",
                                               image_height=image_size[0], image_width=image_size[1],
                                               dataset_config=config, root=stage / "lerobot")
                configure_video_encoding(dataset, config)
            elif episode.fps != frequency:
                raise ValueError("Mixed fps: use separate exports or an explicit resampling protocol")
            selected_cameras = {target: candidates[0][1] for target, candidates in CAMERA_CANDIDATES.items()}
            for camera in selected_cameras.values():
                if camera not in episode.cameras or len(episode.cameras[camera]) < len(state):
                    raise ValueError(f"Required camera missing or too short: {camera}")
            for t in range(len(state)):
                item = {"observation.state": state[t], "action": action[t], "task": episode.instructions[0]}
                for target, source in selected_cameras.items():
                    item[f"observation.images.{target}"] = cv2.resize(
                        episode.rgb(source, t), image_size[::-1], interpolation=cv2.INTER_AREA)
                dataset.add_frame(item)
            dataset.save_episode()
            sidecar = f"annotations/episode_{index:07d}.json"
            write_json(stage / sidecar, annotation)
            np.savez_compressed(stage / "annotations" / f"episode_{index:07d}_motion.npz", **episode.poses)
            report["episodes"].append({"episode_index": index, "source": entry,
                                       "split": entry["split"], "frames": len(state),
                                       "source_frames": list(range(len(state))),
                                       "annotation": sidecar, "missing": annotation.get("missing", []),
                                       "source_semantics": episode.semantics})
        finalize_dataset(dataset)
        dataset = None
        report["complete"] = True
        write_json(stage / "manifest.json", report)
        stage.replace(output)
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
        write_json(stage / "manifest.json", report)
        raise
    finally:
        if dataset is not None:
            dataset.finalize()
            dataset.stop_image_writer()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--env-cfg", default="aloha_agilex")
    parser.add_argument("--state-source", choices=("stored", "command", "measured-arm-command-gripper"), required=True)
    parser.add_argument("--action-source", choices=("stored", "next-command"), required=True)
    args = parser.parse_args(argv)
    export_dataset(args.manifest, args.output, annotations=args.annotations,
                   repo_id=args.repo_id, env_cfg=args.env_cfg,
                   state_source=args.state_source, action_source=args.action_source)


if __name__ == "__main__":
    main()
