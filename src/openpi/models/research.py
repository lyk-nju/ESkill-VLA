"""Architectures published by this research branch. Training infrastructure is shared."""
GROUNDING = ('visual_primitive', 'motion_planner', 'action_execution')
PLANNERS = ('joint_decode',)
SUBJECT = False
ACTION_ABLATIONS = True
DEFAULT_GROUNDING = 'visual_primitive'
DEFAULT_PLANNER = 'joint_decode'


def build_planner(config):
    from openpi.models.motion_planner import MotionPlanner
    return MotionPlanner(config)
