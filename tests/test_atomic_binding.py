"""Behavioral checks for identity ambiguity, event continuity and articulated targets."""
from copy import deepcopy
import importlib
import json
import unittest

binding = importlib.import_module("scripts.preprocess.libero.atomic_vla.bind_instances")
RULES = json.loads(binding.RULES_PATH.read_text())


def scene():
    return {
        "objects": {"akita_black_bowl": ["akita_black_bowl_9", "akita_black_bowl_2"],
                    "cookies": ["cookies_1"], "plate": ["plate_1"],
                    "white_yellow_mug": ["white_yellow_mug_1"], "porcelain_mug": ["porcelain_mug_1"],
                    "cream_cheese": ["cream_cheese_1"], "butter": ["butter_1"]},
        "fixtures": {"table": ["main_table"], "microwave": ["microwave_1"],
                     "flat_stove": ["flat_stove_1"], "wooden_cabinet": ["wooden_cabinet_1"]},
        "regions": {"main_table_between_plate_ramekin_region": {"target": "main_table"}},
        "initial_state": [["on", "akita_black_bowl_9", "plate_1"],
                          ["on", "akita_black_bowl_2", "cookies_1"]],
        "goal_state": [["on", "akita_black_bowl_2", "plate_1"]],
        "obj_of_interest": ["plate_1", "akita_black_bowl_2"],
    }


def event(index, skill, text, arm="single"):
    return {"source_id": f"event_{index:03d}", "skill": skill, "description": text, "arm": arm,
            "start_frame": index*10, "stop_frame": (index+1)*10, "goal_frame": (index+1)*10-1}


class BindingBehavior(unittest.TestCase):
    def bind(self, events, task="", data=None):
        return binding.bind_events(data or scene(), events, task, RULES)

    def test_initial_relation_beats_instance_ordinal_and_interest_order(self):
        events = [event(0, "Pick", "pick up the black bowl"), event(1, "Place", "place the bowl on the plate")]
        before = deepcopy(events)
        result = self.bind(events, "pick up the black bowl on the cookie box and place it on the plate")
        self.assertEqual([r["instance"] for r in result], ["akita_black_bowl_2"]*2)
        self.assertEqual(result[1]["reference_candidates"], ["plate_1"])
        self.assertEqual(result[1]["goal_frame"], 19)
        self.assertEqual(events, before)

    def test_goal_and_interest_alone_do_not_resolve_ambiguous_bowls(self):
        result = self.bind([event(0, "Pick", "pick up the bowl")])[0]
        self.assertEqual(result["status"], "ambiguous")
        self.assertIsNone(result["instance"])

    def test_unknown_colour_is_not_silently_dropped(self):
        result = self.bind([event(0, "Pick", "pick up the blue bowl")])[0]
        self.assertEqual(result["status"], "unresolved")

    def test_conflicting_task_and_event_relations_are_not_forced(self):
        result = self.bind([event(0, "Pick", "pick up the black bowl on the plate")],
                           "pick up the black bowl on the cookie box")[0]
        self.assertEqual(result["status"], "conflict")

    def test_unknown_spatial_relation_stays_unresolved(self):
        result = self.bind([event(0, "Pick", "pick up the black bowl")],
                           "pick up the black bowl behind the cookie box")[0]
        self.assertEqual(result["status"], "unresolved")

    def test_multiple_task_bowls_are_not_assigned_by_event_ordinal(self):
        result = self.bind([event(0, "Pick", "pick up the black bowl")],
                           "pick up the black bowl on the cookie box and pick up the black bowl on the plate")[0]
        self.assertEqual(result["status"], "ambiguous")

    def test_mug_identity_continues_then_switches_to_door(self):
        result = self.bind([event(0, "Pick", "pick up the yellow and white mug"),
                            event(1, "Place", "place the mug into the microwave"),
                            event(2, "Close", "close the microwave door")])
        self.assertEqual([r["instance"] for r in result], ["white_yellow_mug_1", "white_yellow_mug_1", "microwave_1"])
        self.assertEqual(result[2]["scope"]["body_suffix"], "microdoorroot")

    def test_explicit_object_conflict_cannot_inherit(self):
        result = self.bind([event(0, "Pick", "pick up the butter"),
                            event(1, "Place", "place the cream cheese box on the plate")])
        self.assertEqual(result[1]["status"], "conflict")

    def test_new_pick_resets_identity(self):
        result = self.bind([event(0, "Pick", "pick up the cream cheese box"),
                            event(1, "Place", "place it on the plate"),
                            event(2, "Pick", "pick up the butter"), event(3, "Place", "place it on the plate")])
        self.assertEqual([r["instance"] for r in result], ["cream_cheese_1"]*2 + ["butter_1"]*2)

    def test_hold_keeps_identity_without_claiming_physical_hold_verified(self):
        result = self.bind([event(0, "Pick", "pick up the butter"), event(1, "Hold", "hold it"),
                            event(2, "Place", "place it on the plate")])
        self.assertEqual([r["instance"] for r in result], ["butter_1"]*3)
        self.assertEqual(result[1]["motion"]["status"], "no_matched_states")
        self.assertEqual(result[1]["manual_review"], "pending")

    def test_gap_and_release_prevent_stale_pronoun_inheritance(self):
        gap = self.bind([event(0, "Pick", "pick up the butter"), event(2, "Place", "place it on the plate")])
        released = self.bind([event(0, "Pick", "pick up the butter"), event(1, "Place", "place it on the plate"),
                              event(2, "Hold", "hold it")])
        self.assertIsNone(gap[-1]["instance"])
        self.assertIsNone(released[-1]["instance"])

    def test_another_arm_cannot_borrow_identity(self):
        result = self.bind([event(0, "Pick", "pick up the butter", "left"),
                            event(1, "Place", "place it on the plate", "right")])
        self.assertIsNone(result[1]["instance"])

    def test_parts_require_explicit_scope_or_known_default(self):
        result = self.bind([event(0, "Turn", "turn on the stove"),
                            event(1, "Open", "open the wooden cabinet"),
                            event(2, "Open", "open the top drawer of the wooden cabinet")])
        self.assertEqual(result[0]["scope"]["body_suffix"], "button")
        self.assertIsNone(result[1]["instance"])
        self.assertEqual(result[2]["scope"]["body_suffix"], "cabinet_top")

    def test_skill_description_disagreement_is_not_ignored(self):
        result = self.bind([event(0, "Pick", "place the butter on the plate")])[0]
        self.assertEqual(result["status"], "unresolved")


if __name__ == "__main__":
    unittest.main()
