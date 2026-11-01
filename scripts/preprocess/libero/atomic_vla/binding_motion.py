"""Read matched simulator states; evaluate contact and motion without ranking objects."""
import numpy as np

from .binding_v2 import CONFIG, SceneIndex, resolve_geometry


def contact_runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def evaluate(action, track, start, stop, config=CONFIG['motion']):
    """A whole contact interval must agree in pose and alignment; no largest-motion fallback."""
    xyz = track['position'][start:stop]
    relative = track['relative_position'][start:stop]
    rotations = track['relative_rotation'][start:stop]
    contact = track['contact'][start:stop]
    bilateral = track['bilateral_contact'][start:stop]
    quality = track['eef_error'][start:stop] <= config['max_scene_eef_error_m']
    grasp = action in {'Pick', 'Place', 'Hold'}
    evidence = (bilateral if grasp else contact) & quality
    intervals = []
    for left, right in contact_runs(evidence):
        if right-left < config['min_contact_frames']:
            continue
        movement = float(np.linalg.norm(xyz[left:right]-xyz[left], axis=1).max())
        position_spread = float(np.linalg.norm(relative[left:right]-relative[left], axis=1).max())
        rotation_delta = np.einsum('ij,njk->nik', rotations[left].T, rotations[left:right])
        rotation_spread = float(np.arccos(np.clip((np.trace(rotation_delta, axis1=1, axis2=2)-1)/2, -1, 1)).max())
        joints = {name: {'range': float(np.ptp(values[start+left:start+right])), 'unit': unit}
                  for name, (values, unit) in track['joints'].items()}
        if grasp:
            supported = (position_spread <= config['max_relative_position_spread_m']
                         and rotation_spread <= config['max_relative_rotation_spread_rad']
                         and (action == 'Hold' or movement >= config['min_translation_m']))
        elif action in {'Open', 'Close', 'Turn'}:
            supported = any(v['range'] >= config['min_slide_motion_m' if v['unit']=='m' else 'min_hinge_motion_rad']
                            for v in joints.values())
        elif action == 'Push':
            supported = movement >= config['min_translation_m']
        else:
            supported = False
        intervals.append({'start_frame': int(start+left), 'stop_frame': int(start+right),
                          'translation_m': movement, 'relative_position_spread_m': position_spread,
                          'relative_rotation_spread_rad': rotation_spread, 'joint_motion': joints,
                          'supported': bool(supported)})
    return {'supported': any(w['supported'] for w in intervals),
            'contact_frames': int(contact.sum()), 'bilateral_contact_frames': int(bilateral.sum()),
            'quality_frames': int(quality.sum()), 'frames': stop-start,
            'translation_m': float(np.linalg.norm(xyz-xyz[0], axis=1).max()),
            'max_eef_error_m': float(track['eef_error'][start:stop].max()),
            'contact_scope': 'target collision geometry subtree', 'intervals': intervals}


def collect(env, scene, rows, states, source):
    """LIBERO single-arm adapter. Random reset states and region contact stubs are never evidence."""
    if any(row['arm'] != 'single' for row in rows):
        raise ValueError('This native motion adapter requires LIBERO single-arm events')
    index = SceneIndex(scene)
    targets_by_event = {}
    previous = None
    for row in rows:
        broad = next((e['targets'] for e in row['evidence'] if e['method']=='category_attributes_part'), [])
        pool = {tuple(t) for t in broad}
        parsed = row['parsed']
        if (not pool and parsed and parsed['action'] in {'Place','Hold'}
                and parsed['object']['name'] in {'it','object'} and previous
                and previous['stop_frame'] == row['start_frame'] and previous['parsed']
                and previous['parsed']['action'] in {'Pick','Hold'}):
            pool = targets_by_event[previous['source_id']]
        targets_by_event[row['source_id']] = pool
        previous = row
    targets = sorted(set().union(*targets_by_event.values()))
    sim, robot = env.sim, env.robots[0]
    model, data = sim.model, sim.data
    if states.shape != (len(source['state']), len(sim.get_state().flatten())):
        raise ValueError('Matched state sequence does not fit source frames / compiled model')
    objects = {**env.env.objects_dict, **env.env.fixtures_dict}
    fingers = [{model.geom_name2id(n) for n in robot.gripper.important_geoms[side]}
               for side in ['left_finger', 'right_finger']]
    gripper = {model.geom_name2id(n) for n in robot.gripper.contact_geoms}
    tracks, geometry = {}, {}
    for target in targets:
        instance, part = target
        resolved = resolve_geometry(env, {'status':'rule_bound', 'instance':instance, 'scope':index.scope(target)})
        root = model.body_name2id(resolved['root_body'])
        def belongs(body):
            while body:
                if body == root:
                    return True
                body = int(model.body_parentid[body])
            return False
        geom_ids = {model.geom_name2id(n) for n in objects[instance].contact_geoms
                    if belongs(int(model.geom_bodyid[model.geom_name2id(n)]))}
        joints = {name: (int(model.jnt_qposadr[model.joint_name2id(name)]),
                         'm' if int(model.jnt_type[model.joint_name2id(name)]) == 2 else 'rad')
                  for name in resolved['joint_names'] if int(model.jnt_type[model.joint_name2id(name)]) in (2, 3)}
        geometry[target] = (root, geom_ids, joints)
        tracks[target] = {k: [] for k in ['position','relative_position','relative_rotation','contact','bilateral_contact','eef_error']}
        tracks[target]['joints'] = {name: ([], unit) for name, (_, unit) in joints.items()}
    for frame, state in enumerate(states):
        sim.set_state_from_flattened(state)
        sim.forward()
        eef_pos = data.site_xpos[robot.eef_site_id]
        eef_rot = data.site_xmat[robot.eef_site_id].reshape(3, 3)
        eef_error = float(np.linalg.norm(eef_pos-source['state'][frame,:3]))
        pairs = [(int(c.geom1),int(c.geom2)) for c in data.contact[:data.ncon]]
        for target, (root, geoms, joints) in geometry.items():
            track = tracks[target]
            touching = {a for a,b in pairs if b in geoms} | {b for a,b in pairs if a in geoms}
            pos, rot = data.body_xpos[root], data.body_xmat[root].reshape(3,3)
            track['position'].append(pos.copy())
            track['relative_position'].append(eef_rot.T @ (pos-eef_pos))
            track['relative_rotation'].append(eef_rot.T @ rot)
            track['contact'].append(bool(touching & gripper))
            track['bilateral_contact'].append(all(bool(touching & finger) for finger in fingers))
            track['eef_error'].append(eef_error)
            for name, (address, _) in joints.items():
                track['joints'][name][0].append(float(data.qpos[address]))
    for track in tracks.values():
        for key in track.keys()-{'joints'}:
            track[key] = np.asarray(track[key])
    result = {}
    for row in rows:
        metrics = []
        for target in sorted(targets_by_event[row['source_id']]):
            metric = evaluate(row['parsed']['action'], tracks[target], row['start_frame'], row['stop_frame'])
            metric.update(instance=target[0], part=target[1])
            metrics.append(metric)
        count = sum(m['supported'] for m in metrics)
        result[row['source_id']] = {'status': 'unique_support' if count == 1 else 'multiple_support' if count > 1 else 'inconclusive',
                                    'candidates':metrics}
    return result
