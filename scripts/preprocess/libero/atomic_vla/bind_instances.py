"""Bind published Events to LIBERO names; keep rule, geometry and motion checks separate.

The matcher is pure Python. Native LIBERO is imported only by the audit CLI.
Rules describe categories/parts/regions, never episode numbers or instance ordinals.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from scripts.preprocess import DATA_ROOT

ROOT = DATA_ROOT / "libero/atomic-vla"
RULES_PATH = Path(__file__).with_name("binding_rules.json")
ACTION = r"pick up|pick|place|put|hold|open|close|turn on|turn off|turn"
RELATION = r"next to|between|on top of|to left of|to right of|left of|right of|under|behind|near|on|onto|inside|in|into|from|to"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normal(text):
    words = re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
    return " ".join(w for w in words if w not in {"the", "a", "an"})


def catalog(scene):
    return {name: kind for section in ("objects", "fixtures")
            for kind, names in scene[section].items() for name in names}


def candidates(head, scene, rules):
    """Exact phrase/alias match: unknown modifiers must not fall back to a generic noun."""
    return sorted(name for name, kind in catalog(scene).items()
                  if normal(head) in {normal(name), normal(kind),
                                      *(normal(a) for a in rules["aliases"].get(kind, []))})


def parse_action(description, skill, rules):
    text = normal(description)
    match = re.match(rf"^(?:{ACTION})\s+(.+)$", text)
    if not match:
        return None
    verb = text[:match.start(1)].strip()
    allowed = {"Pick": {"pick", "pick up"}, "Place": {"place", "put"}, "Hold": {"hold"},
               "Open": {"open"}, "Close": {"close"}, "Turn": {"turn", "turn on", "turn off"}}
    if verb not in allowed.get(skill, set()):
        return None
    split = re.split(rf"\s+((?:{RELATION})\s+.+)$", match[1], maxsplit=1)
    head, location = split[0], split[1] if len(split) > 1 else None
    part = None
    if skill in {"Open", "Close", "Turn"}:
        parts = {p for spec in rules["parts"].values() for p in spec}
        for label in sorted(parts, key=len, reverse=True):
            if head.endswith(" " + label):
                head, part = head[:-(len(label)+1)], label
                break
            if head.startswith(label + " of "):
                head, part = head[len(label)+4:], label
                break
    return {"head": head, "location": location, "part": part}


def location_candidates(location, scene, rules):
    """Only supported initial-state relations; unknown spatial language stays unresolved."""
    types = catalog(scene)
    mapped = rules["region_relations"].get(location)
    if mapped:
        refs = [name for name, region in scene["regions"].items()
                if types.get(region["target"]) == mapped["target_type"]
                and name == region["target"] + "_" + mapped["region"]]
        predicate = mapped["predicate"]
        method = "named_initial_region"
    else:
        match = re.fullmatch(r"(on|on top of|in|inside|from) (.+)", location)
        if not match:
            return None
        refs = candidates(match[2], scene, rules)
        if not refs:
            return None
        predicate = {"on top of": "on", "inside": "in"}.get(match[1], match[1])
        method = "initial_relation"
    predicates = [p for p in scene["initial_state"] if len(p) == 3
                  and (p[0] == predicate or predicate == "from" and p[0] in {"on", "in"})
                  and p[2] in refs]
    return {"candidates": sorted({p[1] for p in predicates}), "method": method,
            "location": location, "predicates": predicates, "references": refs,
            "time_scope": "task_initial_identity; not re-evaluated after motion"}


def task_identity_constraints(task, pool, scene, rules):
    """Collect task Pick descriptors without assigning task clauses by Event ordinal."""
    result = []
    for clause in re.split(rf"\s+and\s+(?=(?:{ACTION})\b)", normal(task)):
        if not clause.startswith(("pick ", "pick up ")):
            continue
        parsed = parse_action(clause, "Pick", rules)
        if parsed and parsed["location"] and set(candidates(parsed["head"], scene, rules)) & set(pool):
            constraint = location_candidates(parsed["location"], scene, rules)
            result.append(constraint or {"method": "unsupported_relation", "location": parsed["location"]})
    return result


def select_scope(kind, skill, part, rules):
    if skill in {"Pick", "Place", "Hold"}:
        return {"part": "whole", "body_suffix": None}
    options = [(label, spec) for label, spec in rules["parts"].get(kind, {}).items()
               if skill in spec["skills"] and (part == label or part is None and spec.get("default"))]
    if len(options) == 1:
        label, spec = options[0]
        return {"part": label, "body_suffix": spec["body"]}
    return None


def bind_events(scene, events, task, rules):
    """Rule candidates only; neither a successful grasp nor manual correctness is inferred."""
    types, carry, rows = catalog(scene), {}, []
    for event in events:
        row = {**event, "instance": None, "scope": None, "status": "unresolved",
               "candidates": [], "evidence": [], "geometry": {"status": "not_checked"},
               "motion": {"status": "no_matched_states"}, "manual_review": "pending"}
        arm, skill = event["arm"], event["skill"]
        previous = carry.pop(arm, None)
        parsed = parse_action(event["description"], skill, rules)
        if parsed is None or skill not in {"Pick", "Place", "Hold", "Open", "Close", "Turn"}:
            row["reason"] = "unsupported_event_language_or_skill"
            rows.append(row)
            continue
        row["parsed"] = parsed
        pool = candidates(parsed["head"], scene, rules)
        contiguous = previous is not None and previous["stop_frame"] == event["start_frame"]
        if parsed["head"] in {"it", "object"} and skill in {"Place", "Hold"} and contiguous:
            pool = [previous["instance"]]
            row["evidence"].append({"method": "same_arm_pronoun", "source_event": previous["source_id"]})
        row["candidates"] = pool.copy()
        row["evidence"].append({"method": "noun_alias", "phrase": parsed["head"], "candidates": pool.copy()})
        constraints = task_identity_constraints(task, pool, scene, rules) if skill in {"Pick", "Place", "Hold"} else []
        if constraints:
            row["evidence"].extend(constraints)
            if any("candidates" not in c for c in constraints):
                row["reason"] = "unsupported_task_spatial_relation"
                rows.append(row)
                continue
            # Multiple same-category task clauses form a union, never an ordinal assignment.
            allowed = {name for c in constraints for name in c["candidates"]}
            pool = [name for name in pool if name in allowed]
        if skill == "Pick" and parsed["location"]:
            constraint = location_candidates(parsed["location"], scene, rules)
            if constraint is None:
                row["reason"] = "unsupported_event_spatial_relation"
                rows.append(row)
                continue
            row["evidence"].append(constraint)
            pool = [name for name in pool if name in constraint["candidates"]]
        # Destination evidence is separate from the manipulated object.
        if skill == "Place" and parsed["location"]:
            destination = re.sub(rf"^(?:{RELATION})\s+", "", parsed["location"])
            row["reference_candidates"] = candidates(destination, scene, rules)
        if skill in {"Place", "Hold"} and contiguous:
            if previous["instance"] not in pool:
                row.update(status="conflict", reason="continuation_object_disagrees_with_previous_pick_or_hold")
                rows.append(row)
                continue
            pool = [previous["instance"]]
            row["evidence"].append({"method": "same_arm_continuation", "source_event": previous["source_id"],
                                    "note": "identity proposal; does not prove a successful grasp or Hold"})
        row["remaining_candidates"] = pool
        if len(pool) != 1:
            row["status"] = "ambiguous" if len(pool) > 1 else "conflict" if row["candidates"] else "unresolved"
            row["reason"] = "multiple_instances" if len(pool) > 1 else "no_consistent_candidate"
        else:
            scope = select_scope(types[pool[0]], skill, parsed["part"], rules)
            if scope is None:
                row["reason"] = "part_scope_not_defined"
            else:
                row.update(instance=pool[0], scope=scope, status="rule_bound")
                row["evidence"].append({"method": "goal_context_only", "predicates":
                                        [p for p in scene["goal_state"] if pool[0] in p[1:]]})
                if skill in {"Pick", "Hold"}:
                    carry[arm] = row
        rows.append(row)
    return rows


def resolve_geometry(env, row):
    """Resolve stable names to actual visual geoms; articulated parts include their subtree."""
    if row["status"] != "rule_bound":
        raise ValueError("Unresolved binding cannot be passed to the region renderer")
    objects = {**env.env.objects_dict, **env.env.fixtures_dict}
    obj, model = objects[row["instance"]], env.sim.model
    suffix = row["scope"]["body_suffix"]
    body_name = row["instance"] + "_" + suffix if suffix else obj.root_body
    root_id = model.body_name2id(body_name)

    def belongs(body):
        while body:
            if body == root_id:
                return True
            body = int(model.body_parentid[body])
        return False

    names = [name for name in obj.visual_geoms if belongs(int(model.geom_bodyid[model.geom_name2id(name)]))]
    if not names:
        raise ValueError(f"Empty visual geometry for {row['instance']} / {body_name}")
    joints = [model.joint_id2name(i) for i in range(model.njnt) if belongs(int(model.jnt_bodyid[i]))]
    return {"status": "resolved", "root_body": body_name, "visual_geom_names": names, "joint_names": joints}


def motion_evidence(env, rows, match_path, manifest_path, sample):
    """Measure the matched trajectory. Report evidence, without inventing acceptance thresholds."""
    import numpy as np
    match = json.loads(match_path.read_text())
    if sample["key"] != match["source_key"]:
        return None
    if digest(manifest_path) != match["source_manifest_sha256"] or digest(match["excerpt_path"]) != match["excerpt_sha256"]:
        raise ValueError("Scene excerpt / manifest provenance mismatch")
    with np.load(match["excerpt_path"]) as excerpt:
        states, mirror_joints = excerpt["states"].copy(), excerpt["mirror_joints"].copy()
    if len(states) != sample["frames"] or states.shape[1] != len(env.sim.get_state().flatten()):
        raise ValueError("Scene states do not match trajectory length / compiled model state size")
    objects = {**env.env.objects_dict, **env.env.fixtures_dict}
    model, sim = env.sim.model, env.sim
    tracked = {name for row in rows for name in row["candidates"]}
    positions, contacts = {n: [] for n in tracked}, {n: [] for n in tracked}
    part_positions, joint_values = {}, {}
    for row in rows:
        geometry = row["geometry"]
        if geometry["status"] == "resolved":
            part_positions[geometry["root_body"]] = []
            for name in geometry["joint_names"]:
                if int(model.jnt_type[model.joint_name2id(name)]) in (2, 3):
                    joint_values[name] = []
    object_geoms = {n: {model.geom_name2id(g) for g in objects[n].contact_geoms} for n in tracked}
    gripper_geoms = {model.geom_name2id(g) for g in env.robots[0].gripper.contact_geoms}
    joint_errors = []
    for index, state in enumerate(states):
        sim.set_state_from_flattened(state)
        sim.forward()
        joint_errors.append(float(np.abs(sim.data.qpos[env.robots[0]._ref_joint_pos_indexes] - mirror_joints[index]).max()))
        pairs = [(int(c.geom1), int(c.geom2)) for c in sim.data.contact[:sim.data.ncon]]
        for name in tracked:
            positions[name].append(sim.data.get_body_xpos(objects[name].root_body).copy())
            contacts[name].append(any(a in gripper_geoms and b in object_geoms[name] or
                                      b in gripper_geoms and a in object_geoms[name] for a, b in pairs))
        for name in part_positions:
            part_positions[name].append(sim.data.get_body_xpos(name).copy())
        for name in joint_values:
            joint_values[name].append(float(sim.data.qpos[model.jnt_qposadr[model.joint_name2id(name)]]))
    for row in rows:
        metrics = {}
        for name in row["candidates"]:
            start, stop = row["start_frame"], row["stop_frame"]
            geometry = row["geometry"]
            body = geometry["root_body"] if name == row["instance"] else objects[name].root_body
            xyz = np.asarray(part_positions.get(body, positions[name])[start:stop])
            metrics[name] = {"displacement_m": float(np.linalg.norm(xyz[-1]-xyz[0])),
                             "path_length_m": float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
                             "gripper_contact_frames": int(sum(contacts[name][start:stop])),
                             "frames": len(xyz), "position_body": body,
                             "contact_scope": "whole instance collision geometry"}
            if name == row["instance"]:
                metrics[name]["joint_ranges"] = {joint: {
                    "min": min(joint_values[joint][start:stop]), "max": max(joint_values[joint][start:stop]),
                    "unit": "m" if int(model.jnt_type[model.joint_name2id(joint)]) == 2 else "rad"}
                    for joint in geometry["joint_names"] if joint in joint_values}
        row["motion"] = {"status": "measured", "candidate_metrics": metrics, "acceptance": "pending",
                         "note": "Contacts/motion support review; absence or magnitude alone is not a binding rule."}
    return {"source_match": str(match_path), "source_match_sha256": digest(match_path),
            "excerpt_sha256": match["excerpt_sha256"], "restored_joint_max_error_vs_mirror": max(joint_errors),
            "restored_joint_max_error_per_frame_rad": joint_errors,
            "mirror_max_joint_difference_vs_rlds": match["max_joint_difference"],
            "mirror_rgb_comparisons": match["rgb_comparisons"],
            "note": "Matched mirror, not bit-identical RLDS scene ground truth; see source_match."}


def audit(args):
    from .source import source_intervals
    sys.path.insert(0, str(args.libero_root))
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    from libero.libero.envs.env_wrapper import ControlEnv

    started = time.monotonic()
    manifest = json.loads(args.manifest.read_text())
    rules = json.loads(RULES_PATH.read_text())
    bddl_root = args.libero_root / "libero/libero/bddl_files"
    episodes = []
    for sample in manifest["samples"]:
        source_file = Path(sample["metadata"]["file_path"])
        bddl = bddl_root / source_file.parent.name / source_file.name.replace("_demo.hdf5", ".bddl")
        scene = robosuite_parse_problem(str(bddl))
        events = source_intervals(sample["annotation"], sample["frames"])
        rows = bind_events(scene, events, sample["task"], rules)
        # Only model structure is used here. Random initialization is NEVER episode evidence.
        env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_offscreen_renderer=False, hard_reset=False)
        try:
            for row in rows:
                if row["status"] == "rule_bound":
                    row["geometry"] = resolve_geometry(env, row)
            motion = motion_evidence(env, rows, args.state_match, args.manifest, sample) if args.state_match else None
        finally:
            env.close()
        episode = {"number": sample["number"], "key": sample["key"], "task": sample["task"],
                   "annotation_id": sample["annotation_candidate_id"], "frames": sample["frames"],
                   "bddl_path": str(bddl), "bddl_sha256": digest(bddl), "scene_instances": catalog(scene),
                   "events": rows, "motion_source": motion, "manual_review": "pending"}
        episodes.append(episode)
        print(json.dumps({"number": sample["number"], "bindings": [(r["skill"], r["instance"], r["status"]) for r in rows]}), flush=True)
    rows = [r for e in episodes for r in e["events"]]
    report = {"rules_version": rules["version"], "rules_sha256": digest(RULES_PATH), "script_sha256": digest(__file__),
              "source_adapter_sha256": digest(Path(__file__).with_name("source.py")),
              "source_manifest": str(args.manifest.resolve()), "source_manifest_sha256": digest(args.manifest),
              "source_revision": manifest["revision"], "sampling_seed": manifest["seed"],
              "libero_commit": subprocess.check_output(["git", "-C", str(args.libero_root), "rev-parse", "HEAD"], text=True).strip(),
              "summary": {"episodes": len(episodes), "events": len(rows),
                          "rule_bound": sum(r["status"] == "rule_bound" for r in rows),
                          "geometry_resolved": sum(r["geometry"]["status"] == "resolved" for r in rows),
                          "motion_measured": sum(r["motion"]["status"] == "measured" for r in rows), "manual_accepted": 0},
              "region_extent": "full_in_frame", "endpoint_rule": "same bound object at source-adapted goal_frame",
              "elapsed_seconds": round(time.monotonic()-started, 3), "episodes": episodes, "manual_review": "pending"}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "bindings.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Event → 实例／部件绑定检查", "", "规则结果与人工正确率分开；所有人工核验仍为 pending。", "",
             "固定原版 10 条、seed=42，不换样。本轮仅本地 CPU 解析与模型结构检查，预算 5 分钟，无下载／VLM／GPU 渲染；"
             "复用已有 case01 场景状态测量运动，其余轨迹缺匹配完整状态，尚未作物理核验。", "",
             "本规则取源事件描述与技能，不消费旧 VLM 对象 ID。类别／部件／区域别名见 "
             f"[binding_rules.json]({RULES_PATH})；没有逐任务实例编号或事件序号映射。"
             "BDDL goal 只记作上下文，obj_of_interest 不用来替代操作对象。", "",
             "| 样本 / Event | 源技能与描述 | 绑定实例 / 部件 | 规则依据 | 轨迹证据 | 人工 |",
             "| --- | --- | --- | --- | --- | --- |"]
    methods = {"noun_alias": "类别／别名", "named_initial_region": "初始区域", "initial_relation": "初始关系",
               "same_arm_continuation": "同臂继承", "same_arm_pronoun": "同臂指代"}
    for episode in episodes:
        for row in episode["events"]:
            evidence = "、".join(dict.fromkeys(methods[e["method"]] for e in row["evidence"] if e["method"] in methods))
            target = f"`{row['instance']}` / {row['scope']['part']}" if row["instance"] else row["status"] + ": " + row["reason"]
            motion = "已测量，待核验" if row["motion"]["status"] == "measured" else "缺匹配场景状态"
            lines.append(f"| {episode['number']:02d} / {row['source_id']} | {row['skill']}: {row['description']} | {target} | {evidence} | {motion} | pending |")
    lines += ["", "详细候选、BDDL 谓词、几何名称、源边界和运动数值见 [bindings.json](bindings.json)。",
              "", "规则选出的实例尚不等于实际操作已验证；源状态／RGB 与匹配镜像之间已有残差。"
              "终点仍读取本 Event 的同一实例，区域采用 full_in_frame；本轮没有重写旧 bbox 或视频。", "",
              "可复现入口：项目根目录使用已有 libero-native 环境运行 "
              "`python -m scripts.preprocess.libero.atomic_vla.bind_instances --state-match "
              "data/libero/atomic-vla/check/test/native_object/source_match.json --output <新目录>`。"
              "设置 LIBERO_CONFIG_PATH 指向该环境现有 config；不传 --state-match 时只检查规则与模型结构。", "",
              "首轮范围仅为这 10 条的词汇和部件：未知关系／部件、多个实例、前后冲突均留待核验。"
              "Hold 只支持已有标签的身份延续，不据此新增 Hold 事件；尚未接 Qwen 候选辅助或自动运动判定阈值。", ""]
    (args.output / "README.md").write_text("\n".join(lines))
    print(json.dumps(report["summary"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "check/raw/manifest.json")
    parser.add_argument("--libero-root", type=Path, default=Path("/data1/yuankai/robot-research/workspace/third_party/libero-native"))
    parser.add_argument("--output", type=Path, default=ROOT / "check/test/instance_binding")
    parser.add_argument("--state-match", type=Path, help="Existing matched-scene report; never downloads missing states")
    args = parser.parse_args()
    if (args.output / "bindings.json").exists():
        parser.error("Use a new output directory to retain the prior audit")
    audit(args)


if __name__ == "__main__":
    main()
