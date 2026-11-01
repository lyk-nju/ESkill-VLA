"""General binding mechanisms, including changed instance IDs and contradictory evidence."""
from copy import deepcopy
import importlib
import unittest

import numpy as np

b = importlib.import_module('scripts.preprocess.libero.atomic_vla.binding_v2')
m = importlib.import_module('scripts.preprocess.libero.atomic_vla.binding_motion')


def scene():
    return {'objects': {'akita_black_bowl':['bowl_z','bowl_a'], 'plate':['plate_q'],
                        'glazed_rim_porcelain_ramekin':['ramekin_k'], 'butter':['butter_f'],
                        'black_book':['book_j']},
            'fixtures': {'table':['table_t'], 'flat_stove':['stove_s'], 'white_cabinet':['cabinet_c']},
            'regions': {'stove_s_cook_region':{'target':'stove_s'},
                        'table_t_next_to_plate_region':{'target':'table_t'},
                        'table_t_between_plate_ramekin_region':{'target':'table_t'},
                        'table_t_table_center_region':{'target':'table_t'},
                        'cabinet_c_bottom_region':{'target':'cabinet_c'}},
            'initial_state': [['on','bowl_z','stove_s_cook_region'], ['on','bowl_a','table_t_next_to_plate_region']],
            'goal_state': [], 'obj_of_interest': []}


def event(i, skill, text, arm='single'):
    return {'source_id':f'e{i}', 'skill':skill, 'description':text, 'arm':arm,
            'start_frame':i*10, 'stop_frame':(i+1)*10, 'goal_frame':(i+1)*10-1}


def support(instance, part='whole', supported=True):
    return {'instance':instance, 'part':part, 'supported':supported}


class BindingV2(unittest.TestCase):
    def bind(self, events, task='', data=None, **kwargs):
        return b.bind_events(data or scene(), events, task, **kwargs)

    def test_subtype_and_unknown_attributes(self):
        result = self.bind([event(0,'Pick','pick up the book'), event(1,'Pick','pick up the blue bowl')])
        self.assertEqual(result[0]['instance'], 'book_j')
        self.assertIsNone(result[1]['instance'])

    def test_region_ownership_is_independent_of_instance_id(self):
        data = scene()
        result = self.bind([event(0,'Pick','pick up the bowl')], 'pick up the bowl on the stove', data)
        self.assertEqual(result[0]['instance'], 'bowl_z')
        data['objects']['akita_black_bowl'].reverse()
        data['obj_of_interest'] = ['bowl_a']
        self.assertEqual(self.bind([event(0,'Pick','pick up the bowl')], 'pick up the bowl on the stove', data)[0]['instance'], 'bowl_z')

    def test_named_spatial_relations(self):
        data = scene()
        for phrase, region in [('next to the plate','table_t_next_to_plate_region'),
                               ('between the plate and the ramekin','table_t_between_plate_ramekin_region'),
                               ('from the table center','table_t_table_center_region')]:
            with self.subTest(phrase=phrase):
                data['initial_state'] = [['on','bowl_a',region]]
                result = self.bind([event(0,'Pick','pick up the bowl')], 'pick up the bowl '+phrase, data)
                self.assertEqual(result[0]['instance'], 'bowl_a')

    def test_ambiguous_reference_does_not_choose_ordinal(self):
        data = scene(); data['objects']['plate'].append('plate_r')
        result = self.bind([event(0,'Pick','pick up the bowl')], 'pick up the bowl next to the plate', data)
        self.assertIsNone(result[0]['instance'])

    def test_owned_region_query_reuses_logic_for_another_object_and_relation(self):
        data=scene(); data['objects']['butter'].append('butter_g')
        data['initial_state']=[['in','butter_g','cabinet_c_bottom_region']]
        result=self.bind([event(0,'Pick','pick up the butter')], 'pick up the butter in the bottom drawer',data)
        self.assertEqual(result[0]['instance'],'butter_g')

    def test_semantic_region_metadata_does_not_require_a_named_region(self):
        data=scene()
        data['regions']['r73']={'target':'table_t','semantic':{'kind':'next_to','references':['plate']}}
        data['initial_state']=[['on','bowl_z','r73']]
        result=self.bind([event(0,'Pick','pick up the bowl')], 'pick up the bowl next to the plate',data)
        self.assertEqual(result[0]['instance'],'bowl_z')

    def test_unsupported_spatial_query_is_not_dropped(self):
        result = self.bind([event(0,'Pick','pick up the bowl')], 'pick up the bowl behind the plate')
        self.assertEqual(result[0]['reason'], 'unsupported_task_spatial_relation')

    def test_initial_identity_and_current_relation_are_distinct(self):
        events = [event(1,'Pick','pick up the bowl on the stove')]
        self.assertEqual(self.bind(events)[0]['reason'], 'current_spatial_evidence_unavailable')
        current = {'e1':[['on','bowl_a','stove_s_cook_region']]}
        self.assertEqual(self.bind(events, current_relations=current)[0]['instance'], 'bowl_a')
        self.assertEqual(self.bind([event(1,'Pick','pick up the bowl')], 'pick up the bowl on the stove')[0]['instance'], 'bowl_z')

    def test_conflicting_spatial_evidence_remains_conflict(self):
        result = self.bind([event(0,'Pick','pick up the bowl next to the plate')], 'pick up the bowl on the stove')
        self.assertEqual(result[0]['status'], 'conflict')

    def test_multiple_task_objects_do_not_encode_event_order(self):
        result = self.bind([event(0,'Pick','pick up the bowl')],
                           'pick up the bowl on the stove and pick up the bowl next to the plate')
        self.assertEqual(result[0]['status'], 'ambiguous')

    def test_generic_drawer_uses_task_owner_and_part(self):
        result = self.bind([event(0,'Close','close the drawer')], 'put the book in the bottom drawer of the cabinet')
        self.assertEqual((result[0]['instance'],result[0]['scope']['part']), ('cabinet_c','bottom drawer'))

    def test_multiple_owners_remain_ambiguous(self):
        data=scene(); data['fixtures']['white_cabinet'].append('cabinet_d')
        result=self.bind([event(0,'Open','open the top drawer')], data=data)
        self.assertEqual(result[0]['status'],'ambiguous')

    def test_explicit_part_conflict_is_not_overridden(self):
        result=self.bind([event(0,'Close','close the top drawer')], 'put the book in the bottom drawer')
        self.assertEqual(result[0]['status'],'conflict')

    def test_reference_and_manipulated_object_are_separate(self):
        result=self.bind([event(0,'Place','put the book in the bottom drawer'), event(1,'Close','close the drawer')])
        self.assertEqual(result[0]['instance'],'book_j')
        self.assertEqual(result[1]['scope']['part'],'bottom drawer')

    def test_identity_can_be_bound_while_source_skill_is_flagged(self):
        events=[event(0,'Pick','push the plate to the front')]; before=deepcopy(events)
        result=self.bind(events)[0]
        self.assertEqual(result['instance'],'plate_q')
        self.assertEqual(result['skill_check'],'conflict')
        self.assertEqual(result['skill'],'Pick')
        self.assertEqual(events,before)

    def test_same_arm_hold_inherits_then_release_clears(self):
        result=self.bind([event(0,'Pick','pick up the butter'),event(1,'Hold','hold it'),
                          event(2,'Place','put it on the plate'),event(3,'Hold','hold it')])
        self.assertEqual([r['instance'] for r in result],['butter_f']*3+[None])

    def test_gap_other_arm_and_contradictory_object_cannot_inherit(self):
        for following in [event(2,'Place','put it on the plate'), event(1,'Place','put it on the plate','left'),
                          event(1,'Place','put the book on the plate')]:
            with self.subTest(following=following):
                self.assertIsNone(self.bind([event(0,'Pick','pick up the butter'),following])[-1]['instance'])

    def test_parallel_arms_keep_separate_contexts(self):
        events=[event(0,'Pick','pick up the butter','left'),event(0,'Pick','pick up the book','right'),
                event(1,'Place','put it on the plate','left'),event(1,'Place','put it on the stove','right')]
        result=self.bind(events)
        self.assertEqual([r['instance'] for r in result],['butter_f','book_j','butter_f','book_j'])

    def test_motion_resolves_ambiguous_pick_then_propagates(self):
        motion={'e0':{'status':'unique_support','candidates':[support('bowl_a'),support('bowl_z',supported=False)]}}
        result=self.bind([event(0,'Pick','pick up the bowl'),event(1,'Place','put the bowl on the plate')],motion=motion)
        self.assertEqual([r['instance'] for r in result],['bowl_a']*2)

    def test_multiple_or_weak_motion_does_not_select_largest(self):
        for supported in [True,False]:
            motion={'e0':{'status':'measured','candidates':[support('bowl_a',supported=supported),support('bowl_z',supported=supported)]}}
            self.assertEqual(self.bind([event(0,'Pick','pick up the bowl')],motion=motion)[0]['status'],'ambiguous')

    def test_motion_cannot_override_hard_semantics(self):
        motion={'e0':{'status':'unique_support','candidates':[support('bowl_a')]}}
        result=self.bind([event(0,'Pick','pick up the bowl')], 'pick up the bowl on the stove',motion=motion)[0]
        self.assertEqual(result['status'],'conflict')

    def test_both_objects_can_move_but_only_contacted_one_gets_support(self):
        moving=track(); unheld=track(); unheld['bilateral_contact'][:]=False; unheld['contact'][:]=False
        metrics=[dict(instance=name,part='whole',**m.evaluate('Pick',t,0,10))
                 for name,t in [('bowl_z',unheld),('bowl_a',moving)]]
        result=self.bind([event(0,'Pick','pick up the bowl')],motion={'e0':{'status':'unique_support','candidates':metrics}})
        self.assertEqual(result[0]['instance'],'bowl_a')


def track():
    xyz=np.c_[np.linspace(0,.02,10),np.zeros((10,2))]
    return {'position':xyz, 'relative_position':np.zeros((10,3)),
            'relative_rotation':np.tile(np.eye(3),(10,1,1)), 'contact':np.ones(10,bool),
            'bilateral_contact':np.ones(10,bool), 'eef_error':np.zeros(10), 'joints':{}}


class MotionEvidence(unittest.TestCase):
    def test_contact_and_relative_pose_support_pick(self):
        self.assertTrue(m.evaluate('Pick',track(),0,10)['supported'])

    def test_moving_without_contact_is_insufficient(self):
        t=track(); t['contact'][:]=False; t['bilateral_contact'][:]=False
        self.assertFalse(m.evaluate('Pick',t,0,10)['supported'])

    def test_stationary_hold_does_not_require_translation(self):
        t=track(); t['position'][:]=0
        self.assertTrue(m.evaluate('Hold',t,0,10)['supported'])
        self.assertFalse(m.evaluate('Pick',t,0,10)['supported'])

    def test_relative_drift_rejects_incidental_contact(self):
        t=track(); t['relative_position'][:,0]=np.linspace(0,.1,10)
        self.assertFalse(m.evaluate('Pick',t,0,10)['supported'])

    def test_scene_misalignment_cannot_resolve_identity(self):
        t=track(); t['eef_error'][:]=.02
        self.assertFalse(m.evaluate('Pick',t,0,10)['supported'])

    def test_contacts_must_be_sustained_and_bilateral_for_grasp(self):
        t=track(); t['bilateral_contact'][::2]=False
        self.assertFalse(m.evaluate('Pick',t,0,10)['supported'])
        self.assertTrue(m.evaluate('Push',t,0,10)['supported'])

    def test_joint_motion_can_move_a_part_with_static_root(self):
        t=track(); t['position'][:]=0; t['joints']={'door':(np.linspace(0,.2,10),'rad')}
        self.assertTrue(m.evaluate('Open',t,0,10)['supported'])
        t['joints']['door']=(np.zeros(10),'rad')
        self.assertFalse(m.evaluate('Open',t,0,10)['supported'])


if __name__=='__main__':
    unittest.main()
