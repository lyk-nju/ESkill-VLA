"""Event interaction targets and scene entity inventory, independent of rendering.

These are offline annotation proposals. Source manipulated-object bindings stay intact;
task goal predicates disambiguate destinations only, never provide online model inputs.
"""
import re

from .binding_v2 import SceneIndex, parse_action, reference_descriptor, words


def entity_id(instance, part="whole"):
    return instance if part == "whole" else instance + "::" + part.replace(" ", "_")


def scene_entities(scene):
    index = SceneIndex(scene)
    fixtures = {n for names in scene["fixtures"].values() for n in names}
    result = {}
    for instance, category in index.types.items():
        kind = "fixture" if instance in fixtures else "object"
        # LIBERO arenas are supporting surfaces, not finite object instances.
        if category in {"floor", "table", "kitchen_table", "living_room_table", "study_table", "coffee_table"}:
            kind = "background"
        result[instance] = dict(entity_id=instance, instance=instance, part="whole", category=category,
                                kind=kind, scope=index.scope((instance, "whole")))
        for part in index.parts(instance):
            key = entity_id(instance, part)
            result[key] = dict(entity_id=key, instance=instance, part=part, category=category,
                               kind="part", scope=index.scope((instance, part)))
    for name, region in scene["regions"].items():
        if region["target"] not in index.types:
            continue  # Some upstream BDDLs declare sites on absent fixtures.
        key = "region::" + name
        result[key] = dict(entity_id=key, instance=region["target"], part=name, kind="region",
                           category="spatial_region", site_name=name, definition=region)
    return result


def _goal_entity(index, name):
    if name in index.types:
        return name
    region = index.scene["regions"].get(name)
    if region is None or region["target"] not in index.types:
        return None
    owner = region["target"]
    for part, spec in index.parts(owner).items():
        if name == owner + "_" + spec.get("region", ""):
            return entity_id(owner, part)
    return "region::" + name


def _destination(index, parsed):
    """Resolve language without throwing away spatial/part qualifiers."""
    location = parsed["location"] or ""
    desc = reference_descriptor(parsed)
    if desc is None:
        return set(), "missing_destination"
    compartment = re.fullmatch(r"(back|front|left|right) compartment of (.+)", desc["name"] or "")
    if compartment:
        side, owner = compartment.groups()
        return {"region::" + name for name, r in index.scene["regions"].items()
                if r["target"] in index.entities(owner)
                and name == r["target"] + "_" + side + "_contain_region"}, "named_compartment"
    spatial = re.fullmatch(r"(?:to )?(left|right) of (.+)", location)
    if spatial:
        side, reference = spatial.groups()
        refs = index.entities(reference)
        if len(refs) != 1:
            return set(), "ambiguous_spatial_reference"
        matches = set()
        # Region names are native BDDL semantics, not pixel left/right heuristics.
        for name, r in index.scene["regions"].items():
            tail = name.removeprefix(r["target"] + "_").removesuffix("_region")
            if any(set(words(tail)) == set(words(label)) | {side}
                   for ref in refs for label in index.labels[ref]):
                matches.add("region::" + name)
        return matches, "named_spatial_region"
    if not re.match(r"^(on top of|inside|into|onto|on|in) ", location):
        # E.g. between / next to must become regions, never the reference object's box.
        return set(), "unsupported_spatial_relation"
    positional = re.fullmatch(r"(left|right) (.+)", desc["name"] or "")
    if positional and desc["part"] is None:
        side, noun = positional.groups()
        pool = index.entities(noun)
        selected = set()
        for predicate in index.scene["initial_state"]:
            if len(predicate) != 3 or predicate[0] not in {"on", "in"} or predicate[1] not in pool:
                continue
            region = index.scene["regions"].get(predicate[2])
            if region is None:
                continue
            tail = predicate[2].removeprefix(region["target"] + "_").removesuffix("_region")
            if set(words(tail)) == set(words(noun)) | {side}:
                selected.add(predicate[1])
        return selected, "initial_named_position"
    return {entity_id(*t) for t in index.targets(desc)}, "destination_text"


def bind_target(event, scene):
    """Pick/Push -> manipulated entity; Place -> destination; articulated skills -> part."""
    index, entities = SceneIndex(scene), scene_entities(scene)
    source = event.get("source_annotation", {})
    manipulated = source.get("instance")
    result = dict(status="unresolved", entity_id=None, candidates=[], evidence=[], manual_review="pending")
    if not event["boundary_valid"] or not event["skill_valid"]:
        return {**result, "reason": "invalid_event_or_skill"}
    if event["skill"] in {"Pick", "Push", "Open", "Close", "Turn", "Hold"}:
        scope = source.get("scope") or {}
        key = entity_id(manipulated, scope.get("part", "whole")) if manipulated else None
        pool = {key} if source.get("status") in {"rule_bound", "manual_bound"} and key in entities else set()
        result["evidence"].append({"method": "accepted_manipulated_binding", "source_status": source.get("status")})
        if event["skill"] == "Push":
            result["semantic_review"] = "Push currently targets the pushed object, not the destination region"
    elif event["skill"] == "Place":
        parsed = parse_action(event["description"])
        if parsed is None or parsed["action"] != "Place":
            return {**result, "reason": "unsupported_event_language"}
        pool, method = _destination(index, parsed)
        result["evidence"].append({"method": method, "description": event["description"], "candidates": sorted(pool)})
        goals = [p for p in scene["goal_state"] if len(p) == 3 and p[0] in {"on", "in"} and p[1] == manipulated]
        goal_pool = {_goal_entity(index, p[2]) for p in goals} - {None}
        # Narrow an ambiguous textual pool. Never override a unique textual target
        # with the task's final destination (an Event may have an intermediate goal).
        if len(pool) > 1 and len(pool & goal_pool) == 1:
            result["evidence"].append({"method": "task_goal_disambiguation", "predicates": goals})
            pool &= goal_pool
        if not pool:
            result["reason"] = method
    else:
        return {**result, "reason": "unsupported_skill"}
    result["candidates"] = sorted(pool)
    if len(pool) == 1:
        result.update(status="resolved", entity_id=next(iter(pool)))
        result.pop("reason", None)
    else:
        result.setdefault("reason", "ambiguous_destination" if pool else "missing_entity")
    return result
