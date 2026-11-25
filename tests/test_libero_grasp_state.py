from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pytest

from scripts.preprocess.libero.grasp_state import grasp_frames, local_model_xml


def test_native_grasp_uses_manipulated_object_through_place_and_omits_unknown_frames():
    events = []
    for t, (skill, obj) in enumerate([('Pick', 'milk'), ('Place', 'milk'), ('Open', 'drawer'), ('Pick', 'pudding')]):
        binding = dict(status='rule_bound', instance=obj, scope={'part': 'whole'})
        events.append(dict(arm='single', event_id=str(t), start_frame=t, stop_frame=t+1, skill=skill,
                           source_annotation=binding, interaction_target={'entity_id': 'basket' if skill=='Place' else obj}))
    reader = SimpleNamespace(meta=dict(frame_count=4, events=events, arm_ids=['single'], event_index={'single': 'idx'}),
                             array=lambda ref: np.arange(4))
    checked = []
    sim = SimpleNamespace(get_state=lambda: np.zeros(3), set_state_from_flattened=lambda row: checked.append(int(row[0])),
                          forward=lambda: None)
    def check(gripper, obj):
        assert gripper == 'hand' and obj == ('pudding' if checked[-1] == 3 else 'milk')
        return checked[-1] == 1
    env = SimpleNamespace(robots=[SimpleNamespace(gripper='hand')], sim=sim,
                          env=SimpleNamespace(objects_dict={'milk': 'milk', 'pudding': 'pudding'}, _check_grasp=check))
    frames, objects = grasp_frames(reader, env, np.broadcast_to(np.arange(4)[:,None], (4,3)))
    assert frames == {'0': False, '1': True, '3': False} and checked == [0,1,3]
    assert objects == {'0': 'milk', '1': 'milk', '3': 'pudding'}


def test_saved_xml_keeps_geometries_and_relocates_only_matched_assets(tmp_path):
    (tmp_path / 'milk.stl').touch()
    xml = '<mujoco><asset><mesh name="milk" file="/remote/assets/milk.stl"/></asset><worldbody/></mujoco>'
    scene = ElementTree.fromstring(local_model_xml(xml, tmp_path, tmp_path/'robosuite'))
    assert scene.find('asset/mesh').attrib == {'name': 'milk', 'file': str(tmp_path/'milk.stl')}
    with pytest.raises(FileNotFoundError):
        local_model_xml(xml.replace('milk.stl', 'missing.stl'), tmp_path, tmp_path/'robosuite')
