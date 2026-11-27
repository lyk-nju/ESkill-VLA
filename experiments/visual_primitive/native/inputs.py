"""Frozen Stage 1 rows, disjoint training-parent development data, and text queries."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np


IMAGE_SIZE = 256
ALLOWED_UNITS = {"A", "B", "C", "D"}
SPANS = ("oracle", "source", "destination")
DISPLAY = {"akita_black_bowl": "black bowl"}
ALIASES = {
    "black bowl": ("bowl",),
    "cream cheese": ("cream cheese box",),
    "flat stove": ("stove",),
    "wine rack": ("rack",),
    "wooden cabinet": ("cabinet",),
    "bbq sauce": ("barbecue sauce",),
}
TRANSFER = re.compile(
    r"(?:pick up|put|place) (?:the |a |an )?(?P<source>.+?)"
    r"(?:(?: and (?:put|place) it)? (?:on top of|into|onto|in|on) (?:the |a |an )?"
    r"(?P<destination>.+?))?[.!]?$",
    re.IGNORECASE,
)
DRAWER = re.compile(
    r"open the (?P<destination>top drawer) and put the (?P<source>bowl) inside[.!]?$",
    re.IGNORECASE,
)


def sha256_file(path):
    with Path(path).open("rb") as file:
        digest = hashlib.sha256()
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def canonical_prompt(prompt):
    """Same stateless prompt normalization as openpi.transforms.TokenizePrompt."""
    return prompt.strip().replace("_", " ").replace("\n", " ")


def role_spans(prompt):
    """Parse Source and Destination from text alone; unsupported syntax has neither."""
    text = canonical_prompt(prompt)
    match = DRAWER.fullmatch(text) or TRANSFER.fullmatch(text)
    if match is None:
        return {"source": None, "destination": None}
    return {role: list(match.span(role)) if match[role] else None for role in ("source", "destination")}


def object_noun(target):
    if "::" in target:
        return target.split("::", 1)[1].replace("_", " ")
    category = re.sub(r"_\d+$", "", target)
    return DISPLAY.get(category, category.replace("_", " "))


def oracle_span(prompt, target):
    """Resolve the labeled target noun exactly. Ambiguous or absent mentions are invalid."""
    text = canonical_prompt(prompt)
    noun = object_noun(target)
    names = (noun, *ALIASES.get(noun, ()))
    spans = {m.span() for name in names
             for m in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.IGNORECASE)}
    maximal = {span for span in spans if not any(
        other != span and other[0] <= span[0] <= span[1] <= other[1] for other in spans)}
    return list(next(iter(maximal))) if len(maximal) == 1 else None


def phrase_fields(prompt, target):
    noun = object_noun(target)
    locate = "Locate the " + noun
    return {
        "phrases": {**role_spans(prompt), "oracle": oracle_span(prompt, target)},
        "locate_prompt": locate,
        "locate_span": [len("Locate the "), len(locate)],
    }


def _piece_spans(processor, text):
    """Convert SentencePiece original-text boundaries to Python character intervals."""
    proto = processor.encode_as_immutable_proto(text)
    pieces = list(proto.pieces)
    # SentencePiece distributions have exposed both bytes and character offsets.
    # Verify against the reported surface before using either interpretation.
    if all(text[p.begin:p.end] == p.surface for p in pieces):
        return [(p.id, p.begin, p.end) for p in pieces]
    encoded = text.encode("utf-8")
    boundaries = {len(text[:i].encode("utf-8")): i for i in range(len(text) + 1)}
    result = []
    for p in pieces:
        if p.begin not in boundaries or p.end not in boundaries:
            raise ValueError("SentencePiece split a Unicode boundary")
        if encoded[p.begin:p.end].decode("utf-8") != p.surface:
            raise ValueError("SentencePiece surfaces failed offset validation")
        result.append((p.id, boundaries[p.begin], boundaries[p.end]))
    return result


def _annotated_tokens(prompt, tokenizer, spans):
    text = canonical_prompt(prompt)
    processor = tokenizer.tokenizer
    pieces = _piece_spans(processor, text)
    tokens, token_mask = tokenizer(prompt)
    expected = [processor.bos_id(), *(p[0] for p in pieces), *processor.encode("\n")]
    n = min(len(expected), len(tokens))
    if not np.array_equal(tokens[:n], expected[:n]) or not np.array_equal(
            token_mask, np.arange(len(tokens)) < n):
        raise ValueError("Phrase tokenization differs from actual model input")
    # Keep every original-text token, including a tail punctuation token when present.
    # The policy's separately appended newline and BOS have no phrase/text surface.
    body = [(i, start, stop) for i, (_, start, stop) in enumerate(pieces, start=1)
            if i < n and start < stop]

    def select(span):
        if span is None:
            return []
        left, right = span
        if not 0 <= left < right <= len(text):
            raise ValueError("Phrase span outside the exact normalized prompt")
        selected = [(i, start, stop) for i, start, stop in body if start < right and stop > left]
        covered = set().union(*(range(start, stop) for _, start, stop in selected))
        if any(i not in covered and not text[i].isspace() for i in range(left, right)):
            return []
        return [i for i, _, _ in selected]

    queries = {role: select(span) for role, span in spans.items()}
    return {
        "tokens": tokens,
        "token_mask": token_mask,
        "queries": queries,
        "body": [i for i, _, _ in body],
    }


def query_inputs(row, tokenizer):
    """Return actual model tokens and text-local query indices for both prompt variants."""
    instruction = _annotated_tokens(
        row["prompt"], tokenizer, {role: row["phrases"].get(role) for role in SPANS},
    )
    body = instruction.pop("body")
    instruction["queries"].update(tail=body[-1:], sentence=body)
    locate = _annotated_tokens(
        row["locate_prompt"], tokenizer, {"locate": row["locate_span"]},
    )
    locate.pop("body")
    return {"instruction": instruction, "locate": locate}


def _box(box):
    if box is None:
        return None
    values = np.asarray(box, float)
    if (values.shape != (4,) or not np.isfinite(values).all() or
            np.any(values < 0) or np.any(values > IMAGE_SIZE) or
            not np.all(values[2:] > values[:2])):
        return None
    return values.tolist()


def _rgb_sha256(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def load_rgb(row):
    path = row["rgb_path"]
    with np.load(path, allow_pickle=False) as saved:
        image = saved["main"].copy()
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or image.dtype != np.uint8:
        raise ValueError(f"Unexpected image format: {path}")
    if _rgb_sha256(image) != row["rgb_sha256"]:
        raise ValueError(f"Input pixels changed: {path}")
    return image


def _task_roster(stage_root):
    return [t for t in read_json(Path(stage_root) / "plan.json")["tasks"]
            if t["suite"] in ("object", "goal") and t["eligible"]]


def load_frozen(stage_root):
    """Reuse historical RGB, prompts, GT, IDs and pairing; add only probe metadata."""
    root = Path(stage_root).resolve()
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("mode") != "formal" or manifest.get("threshold_px") != 10:
        raise ValueError("Expected the formal Stage 1 center-error ≤10 px protocol")
    if sha256_file(root / "plan.json") != manifest["plan_sha256"]:
        raise ValueError("Frozen Stage 1 plan SHA256 differs from the manifest")
    records = manifest["records"]
    by_id = {r["id"]: r for r in records}
    if len(by_id) != len(records) or {r["unit"] for r in records} != ALLOWED_UNITS:
        raise ValueError("Expected unique IDs and all four frozen units A–D")
    roster = {(t["suite"], t["task_id"]): t for t in _task_roster(root)}
    labeled_boxes = {}
    for r in records:
        if r["unit"] not in ALLOWED_UNITS or _box(r["current_bbox"]) is None:
            raise ValueError(f"Invalid frozen Current label: {r['id']}")
        key = r["rgb_sha256"], r["target"]
        if key in labeled_boxes and not np.allclose(labeled_boxes[key], r["current_bbox"]):
            raise ValueError(f"Conflicting frozen label for identical image/target: {r['id']}")
        labeled_boxes[key] = r["current_bbox"]
    rows = []
    for r in records:
        task = roster[r["suite"], r["task_id"]]
        image = (root / r["image"]).resolve()
        if not image.is_relative_to(root) or not image.is_file():
            raise ValueError(f"Frozen input missing or escaped the run: {r['image']}")
        if r.get("paired_id") and r["paired_id"] not in by_id:
            raise ValueError(f"Unknown frozen pair: {r['paired_id']}")
        fields = phrase_fields(r["prompt"], r["target"])
        receiver = task["receiver"]
        if task.get("pick_drawer_region"):
            receiver += "::top_drawer"
        destination_bbox = (labeled_boxes.get((r["rgb_sha256"], receiver))
                            if fields["phrases"]["destination"] is not None else None)
        rows.append(dict(r, rgb_path=str(image), split="test", group_id=f"{r['suite']}:{r['task_id']}",
                         source_bbox=r["current_bbox"] if r["stage"] == "Pick" else None,
                         destination_bbox=destination_bbox, **fields))
    return rows


def _episode_region(reader, target, frame):
    entity = reader.meta["scene_entities"].get(target, {})
    region = reader.region(entity.get("regions", {}).get("image"), frame)
    return _box(region["bbox"]) if region is not None else None


def _event_target(event):
    value = event.get("interaction_target") or {}
    return value.get("entity_id") if value.get("status") == "resolved" else None


def _anchor(reader, event, stage):
    """Use the nearest valid RGB and target around a prefixed Event fraction."""
    target = _event_target(event)
    if target is None:
        return None
    start, stop = int(event["start_frame"]), int(event["stop_frame"])
    if start >= stop:
        return None
    # A Pick anchor at one-quarter of the event is likely before physical contact.
    # A Place anchor is an Event diagnostic, not certified precontact test E.
    fraction = 0.25 if stage == "Pick" else 0.5
    middle = start + int((stop - start - 1) * fraction)
    candidates = [middle + offset for offset in (0, -1, 1, -2, 2)]
    for frame in candidates:
        if not start <= frame < stop:
            continue
        box = _episode_region(reader, target, frame)
        if box is None:
            continue
        image = reader.rgb("image", frame)
        if image is not None and image.shape == (IMAGE_SIZE, IMAGE_SIZE, 3) and image.dtype == np.uint8:
            return frame, box, image
    return None


def _sorted_key(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def _train_roster(config_file, stage_root):
    """Audit exact parent membership against the actual configured training asset."""
    config_file = Path(config_file).resolve()
    config = read_json(config_file)
    index_path, split_path = Path(config["data_index"]), Path(config["split_path"])
    split = read_json(split_path)
    if split["validation_fraction"] != config["validation_fraction"]:
        raise ValueError("Recorded split fraction differs from actual training config")
    train, val = set(split["train"]), set(split["val"])
    if train & val or len(train) != len(split["train"]) or len(val) != len(split["val"]):
        raise ValueError("Duplicate or shared parent IDs in fixed training split")
    tasks = {canonical_prompt(t["prompt"]).casefold(): t for t in _task_roster(stage_root)}
    index = read_json(index_path)
    actual, episodes, used_ids = set(), defaultdict(list), set()
    for entry in index["episodes"]:
        path = (index_path.parent / entry["path"]).resolve()
        if not path.is_relative_to(index_path.parent.resolve()):
            raise ValueError(f"Episode reference escaped training index: {path}")
        meta = read_json(path)
        eid = meta["episode_id"]
        if eid != entry["episode_id"] or eid in used_ids:
            raise ValueError(f"Duplicate or mismatched episode {eid}")
        used_ids.add(eid)
        parent = meta["source"]["parent_trajectory_id"]
        actual.add(parent)
        if parent not in train:
            continue
        prompt = meta["instructions"][0]
        task = tasks.get(canonical_prompt(prompt).casefold())
        if task is not None:
            episodes[task["name"]].append((parent, path, meta, task))
    if actual != train | val:
        raise ValueError("Grounding index does not cover exactly the parent IDs in the fixed split")
    return episodes, dict(
        train_config=str(config_file), train_config_sha256=sha256_file(config_file),
        data_index=str(index_path), data_index_sha256=sha256_file(index_path),
        split_path=str(split_path), split_sha256=sha256_file(split_path),
        training_parents=len(train), validation_parents=len(val), heldout_parent_overlap=0,
    )


def prepare_dev(config_file, stage_root, output, *, fit_per_task=8, check_per_task=4, seed=42):
    """One Pick and Place anchor per parent, plus unaugmented training-only short-query probes."""
    from openpi.policies.eskill_dataset import EpisodeReader

    if min(fit_per_task, check_per_task) < 1:
        raise ValueError("Each development group requires at least one parent per task")
    output = Path(output).resolve()
    images = output / "dev_rgb"
    images.mkdir(parents=True, exist_ok=False)
    episodes, metadata = _train_roster(config_file, stage_root)
    rows, parents = [], defaultdict(list)
    used_parents = set()

    for task in _task_roster(stage_root):
        task_name = task["name"]
        rank = sorted(episodes[task_name], key=lambda pair: (_sorted_key(seed, pair[0]), pair[1]))
        selected = 0
        for parent, path, meta, _ in rank:
            if parent in used_parents:
                continue
            events = {stage: [e for e in meta["events"]
                              if e["skill"] == stage and e.get("skill_valid") and e.get("boundary_valid")
                              and e.get("arm") == "single"] for stage in ("Pick", "Place")}
            if any(len(events[k]) != 1 for k in events):
                continue
            pick, place = events["Pick"][0], events["Place"][0]
            source = _event_target(pick)
            receiver = _event_target(place)
            prompt = meta["instructions"][0]
            roles = role_spans(prompt)
            # Exclude ambiguous or source-labeled Place events from the ordinary Place diagnostic.
            if not source or not receiver or source == receiver or (
                    oracle_span(prompt, source) != roles["source"] or
                    oracle_span(prompt, receiver) != roles["destination"] or
                    roles["source"] is None or roles["destination"] is None):
                continue
            reader = EpisodeReader(path, metadata=meta)
            anchors = {stage: _anchor(reader, events[stage][0], stage) for stage in ("Pick", "Place")}
            if any(v is None for v in anchors.values()):
                continue
            split = "fit" if selected < fit_per_task else "check"
            parents[split].append(parent)
            used_parents.add(parent)
            selected += 1
            for stage, (frame, gt, rgb) in anchors.items():
                role_target = source if stage == "Pick" else receiver
                base_id = f"dev:{meta['episode_id']}:{frame}"
                rgb_path = images / f"{meta['episode_id']}__{frame:04d}.npz"
                np.savez_compressed(rgb_path, main=rgb)
                shared = dict(unit="dev", suite=task["suite"], task=task_name, task_id=task["task_id"],
                              stage=stage, split=split, parent_id=parent, group_id=parent,
                              episode_id=meta["episode_id"], event_id=events[stage][0]["event_id"],
                              episode_ref=str(path), frame=frame, rgb_path=str(rgb_path),
                              image=str(rgb_path), rgb_sha256=_rgb_sha256(rgb))
                rows.append(dict(shared, id=base_id, condition="native_event", prompt=prompt,
                                 target=role_target, current_bbox=gt,
                                 source_bbox=_episode_region(reader, source, frame),
                                 destination_bbox=_episode_region(reader, receiver, frame),
                                 **phrase_fields(prompt, role_target)))
                if stage != "Pick":
                    continue
                native_prompt = f"pick up the {object_noun(source)}"
                native_id = base_id + ":native"
                rows.append(dict(shared, id=native_id, condition="native_query", prompt=native_prompt,
                                 target=source, current_bbox=gt, source_bbox=gt,
                                 destination_bbox=None, **phrase_fields(native_prompt, source)))
                annotated = meta["scene_entities"]
                candidates = [k for k, e in annotated.items() if e.get("kind") == "object" and
                              k not in (source, receiver) and
                              sum(item.get("category") == e.get("category") for item in annotated.values()) == 1]
                for alternate in sorted(candidates, key=lambda k: _sorted_key(seed, parent + ":" + k)):
                    alt_gt = _episode_region(reader, alternate, frame)
                    if alt_gt is None:
                        continue
                    alt_prompt = f"pick up the {object_noun(alternate)}"
                    rows.append(dict(shared, id=base_id + ":alternate", condition="other_query",
                                     prompt=alt_prompt, target=alternate, current_bbox=alt_gt,
                                     source_bbox=alt_gt, destination_bbox=None, paired_id=native_id,
                                     reference_bbox=gt, reference_kind="original_target",
                                     **phrase_fields(alt_prompt, alternate)))
                    break
            if selected == fit_per_task + check_per_task:
                break
        if selected != fit_per_task + check_per_task:
            raise ValueError(f"Insufficient valid train parents for {task_name}: found {selected}")

    metadata.update(seed=seed, fit_per_task=fit_per_task, check_per_task=check_per_task,
                    selected_parents=dict(parents),
                    counts=dict(Counter(x["split"] for x in rows)),
                    stages=dict(Counter(x["stage"] for x in rows)),
                    protocol="One audited native Pick/Place Event per stage; optional same-image Pick name probes. "
                             "Development only, without official test rows or test-label fitting.")
    return rows, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New inputs directory within the E0 run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit-per-task", type=int, default=8)
    parser.add_argument("--check-per-task", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Input preparation will not replace {args.output}")
    # Dev is prepared first from its train parent roster. Final rows are copied only
    # after the development membership and input choices have been fixed.
    dev, audit = prepare_dev(args.train_config, args.stage_root, args.output, seed=args.seed,
                             fit_per_task=args.fit_per_task, check_per_task=args.check_per_task)
    frozen = load_frozen(args.stage_root)
    dev_hashes = {row["rgb_sha256"] for row in dev}
    test_hashes = {row["rgb_sha256"] for row in frozen}
    if dev_hashes & test_hashes:
        raise ValueError("Development and historical final RGB hashes overlap")
    audit.update(stage_root=str(args.stage_root.resolve()),
                 frozen_manifest_sha256=sha256_file(args.stage_root / "manifest.json"),
                 frozen_plan_sha256=sha256_file(args.stage_root / "plan.json"),
                 test_rows=len(frozen), test_images=len(test_hashes), dev_images=len(dev_hashes),
                 dev_test_rgb_sha256_overlap=0,
                 phrase_coverage={mode: sum(r["phrases"][mode] is not None for r in frozen)
                                  for mode in SPANS})
    write_json(args.output / "dev.json", dev)
    write_json(args.output / "test.json", frozen)
    write_json(args.output / "audit.json", audit)
    print(json.dumps({k: audit[k] for k in
                      ("counts", "stages", "test_rows", "test_images", "phrase_coverage")}, indent=2))


if __name__ == "__main__":
    main()
