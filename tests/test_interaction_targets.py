"""Target identity follows Event semantics, not instance order or final object pose."""
from copy import deepcopy
import unittest

from scripts.preprocess.libero.atomic_vla.interaction_targets import bind_target, scene_entities


def scene():
    return dict(objects={"milk": ["milk_x"], "basket": ["basket_b"], "plate": ["plate_z", "plate_a"]},
                fixtures={"table": ["table"], "wooden_cabinet": ["cabinet_q"], "desk_caddy": ["caddy_r"]},
                regions={"table_plate_left_region": {"target": "table"},
                         "table_plate_right_region": {"target": "table"},
                         "cabinet_q_top_region": {"target": "cabinet_q"},
                         "caddy_r_back_contain_region": {"target": "caddy_r"}},
                initial_state=[["on", "plate_z", "table_plate_right_region"],
                               ["on", "plate_a", "table_plate_left_region"]],
                goal_state=[["in", "milk_x", "basket_b"]])


def event(skill, description, instance="milk_x", part="whole"):
    return dict(skill=skill, description=description, boundary_valid=True, skill_valid=True,
                source_annotation=dict(instance=instance, scope=dict(part=part), status="rule_bound"))


class InteractionTargets(unittest.TestCase):
    def test_pick_place_switch_preserves_original_binding(self):
        pick = event("Pick", "pick up the milk")
        place = event("Place", "place the milk into the basket")
        original = deepcopy(place)
        self.assertEqual(bind_target(pick, scene())["entity_id"], "milk_x")
        self.assertEqual(bind_target(place, scene())["entity_id"], "basket_b")
        self.assertEqual(place, original)

    def test_left_right_is_scene_relation_not_lexical_instance_order(self):
        for side, expected in (("left", "plate_a"), ("right", "plate_z")):
            self.assertEqual(bind_target(event("Place", f"place milk on {side} plate"), scene())["entity_id"], expected)

    def test_unknown_modifier_is_not_discarded(self):
        self.assertIsNone(bind_target(event("Place", "place milk on blue plate"), scene())["entity_id"])

    def test_region_right_of_plate_is_not_plate_itself(self):
        s = scene()
        s["objects"]["plate"] = ["plate_a"]
        result = bind_target(event("Place", "place milk to the right of the plate"), s)
        self.assertEqual(result["entity_id"], "region::table_plate_right_region")

    def test_spatial_reference_with_multiple_plates_stays_unresolved(self):
        result = bind_target(event("Place", "place milk to the right of the plate"), scene())
        self.assertEqual(result["reason"], "ambiguous_spatial_reference")

    def test_unsupported_between_does_not_become_a_whole_object(self):
        result = bind_target(event("Place", "place milk between the plate and the basket"), scene())
        self.assertIsNone(result["entity_id"])
        self.assertEqual(result["reason"], "unsupported_spatial_relation")

    def test_ambiguous_drawer_requires_evidence(self):
        e, s = event("Place", "place milk in the drawer"), scene()
        self.assertIsNone(bind_target(e, s)["entity_id"])
        s["goal_state"] = [["in", "milk_x", "cabinet_q_top_region"]]
        self.assertEqual(bind_target(e, s)["entity_id"], "cabinet_q::top_drawer")

    def test_intermediate_destination_not_replaced_by_final_goal(self):
        s = scene()
        s["objects"]["plate"] = ["plate_a"]
        self.assertEqual(bind_target(event("Place", "place milk on plate"), s)["entity_id"], "plate_a")

    def test_compartment_has_its_own_region(self):
        result = bind_target(event("Place", "place milk in the back compartment of the caddy"), scene())
        self.assertEqual(result["entity_id"], "region::caddy_r_back_contain_region")

    def test_articulated_target_reuses_accepted_part(self):
        result = bind_target(event("Open", "open the drawer", "cabinet_q", "top drawer"), scene())
        self.assertEqual(result["entity_id"], "cabinet_q::top_drawer")

    def test_other_instances_and_background_remain_in_inventory(self):
        entities = scene_entities(scene())
        self.assertEqual(entities["table"]["kind"], "background")
        self.assertEqual(entities["plate_z"]["kind"], "object")
        self.assertIn("cabinet_q::bottom_drawer", entities)


if __name__ == "__main__":
    unittest.main()


def test_interaction_region_preserves_object_beyond_gripper_without_protecting_unrelated_objects():
    import numpy as np
    from scripts.preprocess.libero.atomic_vla.scene_regions import interaction_regions
    grip=np.array([[10,10,20,20],[10,10,20,20]],dtype=float)
    held=np.array([[15,18,22,38],[35,35,40,40]],dtype=float)
    distant=np.array([[40,40,50,50],[40,40,50,50]],dtype=float)
    near,contact=interaction_regions(grip,[held,distant],64,64)
    np.testing.assert_array_equal(contact[0],[6,6,24,38])
    assert np.isnan(contact[1]).all()
    np.testing.assert_array_equal(near[0],[6,6,24,24])
