"""Native full-in-frame scene grounding; writes a review overlay on an existing Episode."""
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

from openpi.augmentation import COPY_MIN_VISIBLE_FRACTION
from openpi.policies.eskill_dataset import EpisodeReader
from scripts.preprocess.artifacts import digest
from .bind_instances import resolve_geometry
from .interaction_targets import scene_entities


def mask_box(mask):
    y, x = np.nonzero(mask)
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1] if len(x) else None


def site_mask(sim, site_id, calibration, height, width):
    """Project a native box site's volume, ignoring occlusion; this is not an object mask."""
    corners = np.array(list(itertools.product((-1, 1), repeat=3))) * sim.model.site_size[site_id]
    world = corners @ sim.data.site_xmat[site_id].reshape(3, 3).T + sim.data.site_xpos[site_id]
    intrinsic, extrinsic, image_transform = calibration
    camera = world @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    if (camera[:, 2] <= 0).any():
        raise ValueError("Region intersects the camera near plane; explicit clipping is required")
    pixel = camera @ intrinsic.T
    pixel /= pixel[:, 2:3]
    pixel = pixel @ image_transform.T
    uv = pixel[:, :2] / pixel[:, 2:3]
    polygon = [tuple(p) for p in uv[ConvexHull(uv).vertices]]
    image = Image.new("1", (width, height))
    ImageDraw.Draw(image).polygon(polygon, fill=1)
    return np.asarray(image, dtype=bool)


def rebase_refs(value, source_dir, destination_dir):
    if isinstance(value, dict):
        if value.get("uri") and not Path(value["uri"]).is_absolute():
            value["uri"] = os.path.relpath(source_dir / value["uri"], destination_dir)
        for item in value.values():
            rebase_refs(item, source_dir, destination_dir)
    elif isinstance(value, list):
        for item in value:
            rebase_refs(item, source_dir, destination_dir)


def extract_scene(row, scene, destination, *, gpu=0, include_robot=False, environment=None):
    """Restore matched source states, isolate each entity, save labels before visualization."""
    import mujoco
    import robosuite.macros as macros
    from libero.libero.envs import OffScreenRenderEnv

    started = time.monotonic()
    destination = Path(destination)
    if (destination / "episode.json").exists():
        raise FileExistsError(destination)
    reader = EpisodeReader(row["source_episode"])
    original = reader.meta
    if digest(reader.path) != row["source_episode_sha256"] or digest(row["bddl"]) != row["bddl_sha256"]:
        raise ValueError("Episode or BDDL changed since inventory")
    record = json.loads(Path(row["source_regions"]).read_text())
    match = record["scene_match"]
    if match["status"] != "actions_exact_unique_match" or digest(match["excerpt_path"]) != match["excerpt_sha256"]:
        raise ValueError("Missing or changed matched simulator states")
    with np.load(match["excerpt_path"]) as archive:
        states = archive["states"]
    h, w = original["cameras"]["image"]["height"], original["cameras"]["image"]["width"]
    n = original["frame_count"]
    calibration = [reader.array(original["cameras"]["image"]["calibration"][k])
                   for k in ("intrinsics", "extrinsics", "image_transform")]
    entities = scene_entities(scene)
    used = {e["interaction_target"]["entity_id"] for e in row["events"]}
    # Every finite object and fixture, every known articulated part, and requested regions.
    entities = {k: v for k, v in entities.items() if v["kind"] != "region" or k in used}
    env = environment or OffScreenRenderEnv(bddl_file_name=row["bddl"], camera_names=["agentview"],
                            camera_heights=h, camera_widths=w, render_gpu_device_id=gpu, hard_reset=False)
    sim, context = env.sim, env.sim._render_context_offscreen
    groups, options = sim.model.geom_group.copy(), context.vopt.geomgroup.copy()
    sitegroups = context.vopt.sitegroup.copy()
    try:
        if environment is None:
            env.seed(0)
            env.reset()
        sim, context = env.sim, env.sim._render_context_offscreen
        if states.shape != (n, len(sim.get_state().flatten())):
            raise ValueError("Matched states do not fit compiled scene")
        groups, options = sim.model.geom_group.copy(), context.vopt.geomgroup.copy()
        context.vopt.sitegroup[:] = 0
        camera_id = sim.model.camera_name2id('agentview')
        tracks, geometry_errors = {}, []
        if include_robot:
            robot = env.robots[0]
            names = list(robot.robot_model.visual_geoms) + list(robot.gripper.visual_geoms)
            entities['robot::single'] = dict(entity_id='robot::single', instance='robot', part='whole',
                kind='robot', category='robot', geometry=dict(status='resolved', visual_geom_names=names,
                method='isolated robot visual geometry with source joints'))
            channels = [c['name'] for c in original['state']['channels']]
            state = reader.array(original['state']['values'])
            joint_columns = [channels.index(f'joint_{i}') for i in range(7)]
            finger_columns = [channels.index(f'gripper_finger_{i}') for i in range(2)]
            if not reader.array(original['state']['valid'])[:, joint_columns+finger_columns].all():
                raise ValueError('Full robot protection requires valid source joints for every frame')
        for key, entity in entities.items():
            entity["regions"] = {}
            if entity["kind"] == "background":
                entity["geometry"] = {"status": "excluded", "reason": "arena_support_surface"}
                continue
            try:
                if entity['kind'] == 'robot':
                    selector = [sim.model.geom_name2id(name) for name in entity['geometry']['visual_geom_names']]
                elif entity["kind"] == "region":
                    site = sim.model.site_name2id(entity["site_name"])
                    if int(sim.model.site_type[site]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                        raise ValueError("Only native box sites are supported")
                    entity["geometry"] = {"status": "resolved", "site_name": entity["site_name"],
                                           "method": "native_site_volume_projection"}
                    selector = site
                else:
                    entity["geometry"] = resolve_geometry(env, {**entity, "status": "rule_bound"})
                    selector = [sim.model.geom_name2id(name) for name in entity["geometry"]["visual_geom_names"]]
                tracks[key] = dict(selector=selector, boxes=np.full((n, 4), np.nan, np.float32),
                                   bits=np.zeros((n, (h*w+7)//8), np.uint8))
            except (ValueError, KeyError) as error:
                entity["geometry"] = {"status": "unresolved", "reason": str(error)}
                geometry_errors.append(key)

        def oriented(value):
            convention = 1 if macros.IMAGE_CONVENTION == "opengl" else -1
            return value[::convention][::-1, ::-1].copy()

        checks = sorted({0, n//2, n-1, *(e["start_frame"] for e in row["events"])})
        alignment, restored = [], []
        for frame in range(n):
            sim.set_state_from_flattened(states[frame])
            sim.forward()
            context.gl_ctx.make_current()
            if frame in checks:
                sim.model.geom_group[:] = groups
                context.vopt.geomgroup[:] = options
                rgb = oriented(sim.render(width=w, height=h, camera_name="agentview"))
                restored.append(rgb)
                alignment.append(dict(frame=frame, rgb_mae=float(np.abs(rgb.astype(float)-reader.rgb("image", frame)).mean())))
            for key, track in tracks.items():
                if entities[key]['kind'] == 'robot':
                    sim.data.qpos[robot._ref_joint_pos_indexes] = state[frame, joint_columns]
                    sim.data.qpos[robot._ref_gripper_joint_pos_indexes] = state[frame, finger_columns]
                    sim.forward()  # Robot is the last track; all object masks use matched scene states.
                if entities[key]["kind"] == "region":
                    mask = site_mask(sim, track["selector"], calibration, h, w)
                else:
                    selected = track["selector"]
                    pose = np.concatenate((sim.data.geom_xpos[selected].ravel(), sim.data.geom_xmat[selected].ravel(),
                                           sim.data.cam_xpos[camera_id], sim.data.cam_xmat[camera_id]))
                    # Exact pose equality only; isolated masks do not depend on other objects.
                    if frame and np.array_equal(pose, track.get('last_pose')):
                        track['boxes'][frame] = track['boxes'][frame-1]
                        track['bits'][frame] = track['bits'][frame-1]
                        continue
                    track['last_pose'] = pose
                    sim.model.geom_group[:] = 5
                    sim.model.geom_group[selected] = 1
                    context.vopt.geomgroup[:] = 0
                    context.vopt.geomgroup[1] = 1
                    segmentation = oriented(sim.render(width=w, height=h, camera_name="agentview", segmentation=True))
                    mask = ((segmentation[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM))
                            & np.isin(segmentation[..., 1], selected))
                box = mask_box(mask)
                if box is not None:
                    track["boxes"][frame] = box
                track["bits"][frame] = np.packbits(mask.reshape(-1), bitorder="big")
    finally:
        if environment is None:
            env.close()
        else:
            sim.model.geom_group[:] = groups
            context.vopt.geomgroup[:] = options
            context.vopt.sitegroup[:] = sitegroups

    meta = deepcopy(original)
    rebase_refs(meta, reader.path.parent, destination)
    arrays = {"frame_indices": np.arange(n, dtype=np.int64), "rendered": np.ones(n, bool),
              "alignment_frames": np.array(checks), "restored_rgb": np.asarray(restored)}

    def add(key, value, *, shape=None, dtype=None, encoding=None):
        arrays[key] = value
        ref = dict(uri="scene_regions.npz", key=key, shape=shape or list(value.shape), dtype=dtype or str(value.dtype))
        if encoding:
            ref["encoding"] = encoding
        return ref

    indices = add("frame_indices", arrays["frame_indices"])
    rendered = add("rendered", arrays["rendered"])
    for number, (key, track) in enumerate(tracks.items()):
        prefix = f"entity_{number:03d}"
        entities[key]["regions"]["image"] = dict(
            frame_indices=indices, boxes_xyxy=add(prefix+"_boxes", track["boxes"]),
            boxes_valid=add(prefix+"_boxes_valid", np.isfinite(track["boxes"]).all(-1)),
            masks=add(prefix+"_mask_bits", track["bits"], shape=[n, h, w], dtype="bool", encoding="packbits"),
            masks_valid=rendered, region_extent="full_in_frame",
            evidence="native isolated visual geometry" if entities[key]["kind"] != "region" else "projected native region volume")
    targets = {e["event_id"]: e["interaction_target"] for e in row["events"]}
    for event in meta["events"]:
        event["interaction_target"] = deepcopy(targets[event["event_id"]])
        key = event["interaction_target"]["entity_id"]
        event["interaction_target"]["grounding_status"] = "generated" if key in tracks else "missing"
    meta["scene_entities"] = entities
    meta["interaction_annotation"] = dict(version="interaction_target_v1_review", source_episode=str(reader.path),
        source_episode_sha256=row["source_episode_sha256"], bddl=row["bddl"], bddl_sha256=row["bddl_sha256"],
        source_regions=row["source_regions"], source_regions_sha256=digest(row["source_regions"]),
        scene_match=match, manual_review="pending", geometry_errors=geometry_errors, alignment=alignment,
        seconds=round(time.monotonic()-started, 2), train_interface="unchanged; original object_regions semantics",
        scope="All BDDL objects/finite fixtures and known parts; arena floor/table/walls are background",
        rules_sha256=digest(Path(__file__).with_name("interaction_targets.py")), renderer_sha256=digest(__file__))
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "scene_regions.npz", **arrays)
    (destination / "episode.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    return destination / "episode.json"


def interaction_locked(box, near):
    """The existing full-bbox proximity proxy, not a grasp/contact detector."""
    return (np.isfinite(box).all(-1) & np.isfinite(near).all(-1) &
            (box[:,0] < near[:,2]) & (near[:,0] < box[:,2]) &
            (box[:,1] < near[:,3]) & (near[:,1] < box[:,3]))


def copy_eligibility(kind, locked, visible_fraction, full_count):
    """Movable scene objects use the same rule, including bowls, cups and baskets."""
    return (kind == 'object') & ~locked & (visible_fraction >= COPY_MIN_VISIBLE_FRACTION) & (full_count > 0)


def interaction_regions(gripper, object_boxes, height, width):
    """Cheap proximity proxy: preserve the gripper and overlapping object/part together."""
    near = (np.asarray(gripper, dtype=float) + np.array([-4,-4,4,4])).clip([0,0,0,0],[width,height,width,height])
    contact, active = near.copy(), np.zeros(len(near), bool)
    for box in object_boxes:
        locked = interaction_locked(box, near)
        active |= locked
        contact[locked,:2] = np.minimum(contact[locked,:2],box[locked,:2])
        contact[locked,2:] = np.maximum(contact[locked,2:],box[locked,2:])
    contact[~active] = np.nan
    return near, contact


def prepare_online_episode(path, destination, *, gpu=0, environment=None):
    """Add visible masks and cheap editing metadata; retain every original full-region label."""
    import re
    import mujoco
    import robosuite.macros as macros
    from libero.libero.envs import OffScreenRenderEnv

    reader = EpisodeReader(path)
    meta = deepcopy(reader.meta)
    destination = Path(destination)
    if destination.exists() and (destination.resolve() != reader.path.parent or 'online_grounding' in meta):
        raise FileExistsError(destination)
    match = meta['interaction_annotation']['scene_match']
    with np.load(match['excerpt_path']) as archive:
        states = archive['states']
    n, h, w = meta['frame_count'], meta['cameras']['image']['height'], meta['cameras']['image']['width']
    entities = meta['scene_entities']
    state = reader.array(meta['state']['values'])
    channels = [c['name'] for c in meta['state']['channels']]
    joints = [channels.index(f'joint_{i}') for i in range(7)]
    fingers = [channels.index(f'gripper_finger_{i}') for i in range(2)]
    # Full boxes remain the authority for supervision, including invisible objects.
    boxes = {k: reader.array(e['regions']['image']['boxes_xyxy']) for k, e in entities.items()
             if 'image' in e.get('regions', {}) and e['kind'] not in ('background', 'robot', 'region')}
    grip = reader.array(meta['effector_regions']['single']['image']['boxes_xyxy'])
    arrays, selectors, alignment = {}, {}, []
    lookup = np.array([int(i).bit_count() for i in range(256)], np.uint8)
    env = environment or OffScreenRenderEnv(bddl_file_name=meta['interaction_annotation']['bddl'], camera_names=['agentview'],
                            camera_heights=h, camera_widths=w, render_gpu_device_id=gpu, hard_reset=False)
    context = env.sim._render_context_offscreen
    options, sitegroups = context.vopt.geomgroup.copy(), context.vopt.sitegroup.copy()
    try:
        if environment is None:
            env.reset()
        sim, robot = env.sim, env.robots[0]
        context = sim._render_context_offscreen
        context.vopt.geomgroup[0] = 0  # Collision geometry is not an RGB surface.
        context.vopt.geomgroup[1] = 1
        context.vopt.sitegroup[:] = 0
        for k in boxes:
            selectors[k] = [sim.model.geom_name2id(name) for name in entities[k]['geometry']['visual_geom_names']]
            arrays[k+'_visible'] = np.zeros((n, (h*w+7)//8), np.uint8)
        def oriented(value):
            convention = 1 if macros.IMAGE_CONVENTION == 'opengl' else -1
            return value[::convention][::-1, ::-1].copy()
        for frame in range(n):
            sim.set_state_from_flattened(states[frame])
            sim.data.qpos[robot._ref_joint_pos_indexes] = state[frame, joints]
            sim.data.qpos[robot._ref_gripper_joint_pos_indexes] = state[frame, fingers]
            sim.forward()
            context.gl_ctx.make_current()
            seg = oriented(sim.render(width=w, height=h, camera_name='agentview', segmentation=True))
            for k, ids in selectors.items():
                visible = (seg[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[..., 1], ids)
                arrays[k+'_visible'][frame] = np.packbits(visible.ravel(), bitorder='big')
            if frame in {0, n//2, n-1}:
                rgb = oriented(sim.render(width=w, height=h, camera_name='agentview'))
                arrays[f'restored_{frame}'] = rgb
                alignment.append(dict(frame=frame, rgb_mae=float(np.abs(rgb.astype(float)-reader.rgb('image', frame)).mean())))
    finally:
        if environment is None:
            env.close()
        else:
            context.vopt.geomgroup[:] = options
            context.vopt.sitegroup[:] = sitegroups

    rebase_refs(meta, reader.path.parent, destination)
    def add(key, value, **extra):
        arrays[key] = value
        return dict(uri='online_grounding.npz', key=key, shape=list(value.shape), dtype=str(value.dtype), **extra)
    near, contact = interaction_regions(grip, [box for k,box in boxes.items()
                                               if entities[k]['kind'] in ('object','part')], h, w)
    for k in boxes:
        e, sequence = entities[k], entities[k]['regions']['image']
        bits = arrays[k+'_visible']
        ref = add(k+'_visible', bits, encoding='packbits')
        ref.update(shape=[n,h,w], dtype='bool')
        sequence['visible_masks'] = ref
        sequence['visible_masks_valid'] = sequence['masks_valid']
        count = lookup[bits].sum(1)
        original = reader.meta['scene_entities'][k]['regions']['image']['masks']
        full_count = lookup[reader._raw(original['uri'], original['key'])].sum(1)
        ratio = count / np.maximum(full_count, 1)
        locked = interaction_locked(boxes[k], near)
        e['interaction_locked'] = add(k+'_interaction_locked', locked)
        e['copy_eligible'] = add(k+'_copy_eligible', copy_eligibility(e['kind'], locked, ratio, full_count))
        e['visible_fraction'] = add(k+'_visible_fraction', ratio.astype(np.float32))
    # Determine replaceable noun slots once, not in each worker on every training step.
    for event in meta['events']:
        text = event['description'].strip().rstrip('.')
        match = re.match(r'(?i)^(pick up|pick|grasp|lift|push|open|close|turn on|turn off|turn)\s+(?:the\s+|a\s+)?(.+)$', text)
        relational = re.search(r'(?i)\b(beside|behind|near|left|right|next|between|into|on|in|to)\b', match[2]) if match else True
        event['grounding_template'] = match[1]+' the {object}' if match and not relational else None
    meta['online_grounding'] = dict(version=1, interaction_boxes=add('interaction_boxes', contact),
        interaction_region='union of expanded gripper and overlapping full object/part boxes; proximity proxy',
        visible_alignment=alignment, source_episode=str(reader.path),
        native_occlusion='preserve full bbox and existing RGB canvas; visible mask does not invalidate labels',
        copy_rule=f'movable object without category blacklist, visible/full >= {COPY_MIN_VISIBLE_FRACTION:.2f}, '
                  'no proximity to gripper; source/destination safety checked online; bbox proxy is not a contact label')
    destination.mkdir(parents=True, exist_ok=True)
    np.savez(destination/'online_grounding.npz', **arrays)  # Uncompressed members support row-wise mmap reads.
    (destination/'episode.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return destination/'episode.json'
