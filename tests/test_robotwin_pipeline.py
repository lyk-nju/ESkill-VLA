"""File-level regression tests; no simulator, policy or collection runs."""
from pathlib import Path
import pickle
import json
import subprocess
import sys

import numpy as np
import pytest
import h5py

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.append(str(WORKSPACE / "RoboTwin"))

from collection.storage import write_episode
from scripts.preprocess.robotwin.source import load_episode, validate_episode, freeze_entry, write_json
from scripts.preprocess.robotwin.annotate import build_annotation, validate_annotation, remaining_path
from scripts.preprocess.robotwin.export import transitions, auxiliary_sample, export_dataset
from scripts.preprocess.robotwin.render import render_episode


@pytest.mark.parametrize('duplicate',[False,True])
def test_clean_summary_keeps_source_versions_and_unique_episode_identity(tmp_path,monkeypatch,duplicate):
    from scripts.preprocess.robotwin import check
    parent=tmp_path/'source'
    parent.mkdir()
    write_json(parent/'collection.json',{'fingerprint':'prior'})
    write_json(parent/'collection_run.json',{'tasks':['stack_blocks_two'],'episodes_per_task':2})
    child=parent/'supplement'
    write_json(child/'collection.json',{'fingerprint':'new','config':{
        'resume_from':str(parent),'parent_fingerprint':'prior'}})
    for root,index,identity in ((parent,0,'prior'),(child,0 if duplicate else 1,'new')):
        path=root/'stack_blocks_two/aloha_agilex/raw'/f'episode_{index:07d}.hdf5'
        path.parent.mkdir(parents=True)
        with h5py.File(path,'w') as f:
            f.attrs.update(complete=True,task_success=True,condition='clean',seed=index,
                           collection_fingerprint=identity)
    monkeypatch.setattr(check,'SOURCE_ROOT',tmp_path/'outputs')
    if duplicate:
        with pytest.raises(ValueError,match='Duplicate episode identity'):
            check.collection_summary(parent)
    else:
        summary=check.collection_summary(parent)
        assert summary['all_complete'] and summary['collected']==2
        assert set(summary['source_versions'].values())=={'prior','new'}


def _native_rule_fixture(tmp_path, task, primitives, names, commands, contact_gap=None):
    count=len(commands['left'])
    path=tmp_path/f'{task}.hdf5'
    entities={f'object_{key}':[{'name':name,'id':i+60}] for i,(key,name) in enumerate(names.items())}
    roles={f'object_{key}':{'label':key,'kind':'object'} for key in names}
    observations=[]
    if task=='place_burger_fries':
        for i in range(count):
            observations.append(dict(frame_index=i,arm='right',actor='fries',actor_position=[0.,0.,0.],
                gripper_command_open=False,contact_points=[] if i==contact_gap else [[0.,0.,0.]]))
    with h5py.File(path,'w') as f:
        f.attrs.update(task=task,seed=0,complete=True,entities=json.dumps(entities),region_roles=json.dumps(roles))
        f.create_dataset('frames',data=json.dumps([dict(frame=i,sim_time=i*.06) for i in range(count)]))
        source=[dict(primitive_id=i,label=label,arm=arm,actor=actor,start_frame=start,end_frame=end)
                for i,(label,arm,actor,start,end) in enumerate(primitives)]
        f.create_dataset('source_evidence',data=json.dumps(dict(subtasks=source,observations=observations)))
        for arm in ('left','right'):
            pose=np.zeros((count,7));pose[:,3]=1
            f.create_dataset(f'signals/endpose/{arm}_endpose',data=pose)
            f.create_dataset(f'signals/joint_action/{arm}_gripper',data=commands[arm])
    return path


@pytest.mark.parametrize('contact_gap',[None,4])
def test_primitive_merge_and_hold_need_per_frame_evidence(tmp_path,contact_gap):
    from scripts.preprocess.robotwin.annotate import build_native_event_record
    parts=[('grasp','left','hamburg',0,1),('grasp','right','fries',0,1),
           ('lift','left','',2,2),('lift','right','',2,2),('place','left','hamburg',3,5),
           ('lift','left','',6,6),('back_to_origin','left','',7,9),
           ('place','right','fries',7,9),('lift','right','',10,10)]
    path=_native_rule_fixture(tmp_path,'place_burger_fries',parts,{'hamburg':'hamburg','frenchfries':'fries'},
        {'left':np.r_[np.zeros(6),np.ones(6)],'right':np.zeros(12)},contact_gap)
    record=build_native_event_record(path)
    right=[e for e in record['events'] if e['arm']=='right']
    assert [(e['skill'],e['start_frame'],e['stop_frame']) for e in right]==[
        ('Pick',0,3),('Hold' if contact_gap is None else 'Unknown',3,7),('Place',7,10),('Return',10,11)]
    assert right[0]['goal_frame']==2
    assert {s for e in record['events'] for s in e['source_ids']}=={p['source_id'] for p in record['source_primitives']}
    assert record['manual_review']=='pending'


def test_zero_span_place_attempt_is_retained_inside_real_place(tmp_path):
    from scripts.preprocess.robotwin.annotate import build_native_event_record
    parts=[('grasp','right','071_can',0,1),('place','right','071_can',2,2),
           ('move_to_pose','right','',2,3),('move_by_displacement','right','',4,4),
           ('open_gripper','right','',5,6),('back_to_origin','right','',7,9),
           ('grasp','left','110_basket',7,9),('close_gripper','left','',10,10),('lift','left','',11,12)]
    path=_native_rule_fixture(tmp_path,'place_can_basket',parts,{'can':'071_can','basket':'110_basket'},
        {'left':np.r_[np.ones(7),np.zeros(7)],'right':np.r_[np.zeros(5),np.ones(9)]})
    r=build_native_event_record(path)
    place=next(e for e in r['events'] if e['skill']=='Place')
    assert (place['start_frame'],place['stop_frame'],place['goal_frame'])==(2,7,6)
    assert place['zero_span_source_attempts']==['primitive_0001']
    assert len(place['source_ids'])==4
    assert next(e for e in r['events'] if e['object_key']=='object_basket')['skill']=='Pick'


def test_full_gripper_renderer_keeps_region_when_completely_occluded(tmp_path):
    from scripts.preprocess.robotwin.render import render_native_record
    from PIL import Image
    path = tmp_path / "full.hdf5"
    rgb = np.full((16, 20, 3), [80, 100, 120], dtype=np.uint8)
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[3:7, 5:9] = 1
    invisible = {"bbox": None, "pixels": 0, "full_bbox": [5, 3, 9, 7],
                 "full_pixels": 16, "visible_fraction": 0.0}
    cameras = ("head_camera", "left_camera", "right_camera")
    roles = ("left_gripper", "right_gripper")
    with h5py.File(path, "w") as f:
        f.attrs.update(full_gripper=True, complete=True, task="occluded", seed=0, playback_fps=15,
                       entities=json.dumps({role: [{"id": i+33}] for i, role in enumerate(roles)}),
                       region_roles="{}")
        f.create_dataset("frames", data=json.dumps([dict(physical_step=0, sim_time=0,
            views={camera: {role: invisible for role in roles} for camera in cameras})]))
        for camera in cameras:
            f.create_dataset(f"{camera}/rgb", data=rgb[None])
            f.create_dataset(f"{camera}/actor_ids", data=np.zeros((1, 16, 20), dtype=np.uint32))
            for role in roles:
                f.create_dataset(f"{camera}/{role}_full_mask", data=mask[None])
    examples = tmp_path / "examples"
    report = render_native_record(path, tmp_path / "rendered.mp4", examples)
    assert report["frames"] == 1
    assert all(v["fully_occluded"] == 1 and v["out_of_view"] == 0 for v in report["views"].values())
    for camera in cameras:
        crop = np.asarray(Image.open(examples / "frame_00000" / camera / "left_gripper_full_crop.png"))
        np.testing.assert_array_equal(crop, rgb[3:7, 5:9])


@pytest.fixture
def raw(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    for frame in range(4):
        image = np.zeros((16, 20, 3), dtype=np.uint8)
        image[..., 0], image[..., 1] = 180 + frame, 20
        data = {"observation": {camera: {"rgb": image.copy()} for camera in
                                ("head_camera", "left_camera", "right_camera")},
                "joint_action": {}, "joint_state": {}, "endpose": {}}
        for arm in ("left", "right"):
            data["joint_action"][f"{arm}_arm"] = np.full(6, frame + .5)
            data["joint_action"][f"{arm}_gripper"] = float(frame) / 4
            data["joint_state"][f"{arm}_arm"] = np.full(6, frame)
            data["joint_state"][f"{arm}_gripper_command"] = float(frame) / 4
            data["endpose"][f"{arm}_endpose"] = np.array([frame * .1, 0, .2, 1, 0, 0, 0])
        with (cache / f"{frame}.pkl").open("wb") as stream:
            pickle.dump(data, stream)
    path = tmp_path / "raw.hdf5"
    write_episode(cache, path, instructions=["move cube"], frequency=10,
                  evidence={"frame_count": 4, "events": [], "subtasks": []},
                  metadata={"collection_fingerprint": "test"})
    entry = freeze_entry({"hdf5": str(path), "format": "robotwin.raw.v2", "task": "cube", "episode": "episode_0000000"})
    return entry, load_episode(entry)


def test_full_raw_timeline_and_explicit_transition_mapping(raw):
    _, episode = raw
    assert validate_episode(episode)["errors"] == []
    assert episode.count == 4
    for camera in episode.cameras:
        np.testing.assert_array_equal(episode.rgb(camera, 3)[0, 0], [183, 20, 0])
    state, action = transitions(episode, "measured-arm-command-gripper", "next-command")
    assert state.shape == action.shape == (3, 14)
    np.testing.assert_array_equal(state[:, 0], [0, 1, 2])
    np.testing.assert_array_equal(action[:, 0], [1.5, 2.5, 3.5])
    with pytest.raises(ValueError):
        transitions(episode, "stored", "stored")


def test_legacy_actions_are_not_shifted_twice_and_missing_rgb_stays_missing(raw, tmp_path):
    from XPolicyLab.utils.process_data import images_encoding
    _, original = raw
    path, events = tmp_path / "legacy.hdf5", tmp_path / "events.json"
    write_json(events, {"frame_count": 4, "frequency": 10, "events": [], "subtasks": []})
    with h5py.File(path, "w") as f:
        f.create_dataset("data_format_version", data="v1.0")
        f.create_dataset("instructions", data=json.dumps(["move cube"]))
        f.create_dataset("additional_info/frequency", data=10)
        for arm in ("left", "right"):
            for group, offset in (("state", 0), ("action", 1)):
                f.create_dataset(f"{group}/{arm}_arm_joint_states", data=np.tile(np.arange(offset, offset+3)[:, None], (1, 6)))
                f.create_dataset(f"{group}/{arm}_ee_joint_states", data=np.zeros((3, 1)))
                f.create_dataset(f"{group}/{arm}_ee_poses", data=original.poses[arm][offset:offset+3])
        for name in original.cameras:
            encoded, width = images_encoding(original.cameras[name][:-1])
            f.create_dataset(f"vision/{name}/colors", data=encoded, dtype=f"S{width}")
    episode = load_episode({"hdf5": str(path), "events": str(events), "format": "xpolicylab-v1"})
    state, action = transitions(episode, "stored", "stored")
    np.testing.assert_array_equal(action[:, 0], [1, 2, 3])
    assert episode.rgb("cam_left_wrist", 3) is None
    assert episode.rgb("cam_head", 0)[0, 0, 0] > 170
    np.testing.assert_allclose(episode.poses["left"][-1], original.poses["left"][-1])
    assert validate_episode(episode)["errors"] == []


def test_annotations_do_not_invent_events_and_allow_concurrency(raw):
    _, episode = raw
    blank = build_annotation(episode)
    assert blank["events"] == [] and blank["missing"] == ["events", "regions"]
    events = [{"arm": arm, "start_frame": 0, "end_frame": 3, "skill": "Hold", "object": "cube"}
              for arm in ("left", "right")]
    annotations = build_annotation(episode, {"events": events})
    labels = auxiliary_sample(episode, annotations, 1)
    assert labels["left"]["goal_frame"] == 3 and labels["left"]["goal"] is None
    path = remaining_path(episode, events[0], 1, points=3)
    np.testing.assert_allclose(path[:, 0], [0, .1, .2], atol=1e-6)
    annotations["events"].append(dict(events[0]))
    with pytest.raises(ValueError, match="Overlapping"):
        validate_annotation(episode, annotations)


def test_goal_uses_same_object_at_end(raw):
    _, episode = raw
    annotations = build_annotation(episode, {
        "events": [{"arm": "left", "start_frame": 0, "end_frame": 3, "skill": "Place", "object": "cube"}],
        "regions": [{"arm": "left", "role": "object", "object": obj, "frame_index": 3,
                     "camera": "cam_head", "bbox": [0, 0, 10, 10]} for obj in ("tray", "cube")]})
    assert auxiliary_sample(episode, annotations, 0)["left"]["goal"]["object"] == "cube"


def test_renderer_rgb_and_frame_count(raw, tmp_path):
    import av
    _, episode = raw
    path = tmp_path / "check.mp4"
    result = render_episode(episode, path, cameras=list(episode.cameras))
    assert result["frames"] == 4
    with av.open(str(path)) as video:
        frames = list(video.decode(video=0))
        assert len(frames) == 4
        image = frames[0].to_ndarray(format="rgb24")
        assert int(image[180, 200, 0]) > int(image[180, 200, 2]) + 100


def test_imports_do_not_require_simulator_or_training():
    code = f"import sys; sys.path.append({str(WORKSPACE / 'RoboTwin')!r}); from scripts.preprocess.robotwin import source, annotate, check, render, export; import collection.collect; assert not any(n.split('.')[0] in ('sapien','torch','lerobot') for n in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("use_videos", [False, True])
def test_export_writes_reloadable_data_and_source_mapping(raw, tmp_path, use_videos):
    entry, _ = raw
    entry["split"] = "train"
    manifest = tmp_path / "input.json"
    write_json(manifest, {"samples": [entry]})
    output = tmp_path / "training"
    result = export_dataset(manifest, output, annotations=None, repo_id="local/fixture",
                            env_cfg="aloha_agilex", state_source="command",
                            action_source="next-command", use_videos=use_videos, image_size=(16, 20))
    assert result["complete"]
    assert result["episodes"][0]["source_frames"] == [0, 1, 2]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset("local/fixture", root=output / "lerobot", video_backend="pyav")
    assert len(dataset) == 3
    np.testing.assert_allclose(dataset[0]["action"].numpy()[0], 1.5)
    assert (output / "annotations/episode_0000000_motion.npz").exists()
