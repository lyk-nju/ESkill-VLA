"""Canonical planner names used in configurations and evaluation reports."""


def planner_name(value):
    # Historical training recipes and checkpoints used numeric architecture IDs.
    return {'v1': 'trajectory', 'v2': 'boundary_first', 'v2_joint': 'joint_decode'}.get(value, value)
