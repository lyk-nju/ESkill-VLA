"""Planner architectures available on the motion-planner branch."""
GROUNDING = ('visual_primitive', 'motion_planner')
PLANNERS = ('v1', 'v2', 'v2_joint')
SUBJECT = False
ACTION_ABLATIONS = False
DEFAULT_GROUNDING = 'visual_primitive'
DEFAULT_PLANNER = 'v1'


def build_planner(config):
    from openpi.models.motion_planner import MotionPlanner
    from openpi.models.planner_variants import EarlyBoundaryPlanner, EightSlotPlanner
    return {'v1': EightSlotPlanner, 'v2': EarlyBoundaryPlanner, 'v2_joint': MotionPlanner}[config.planner_version](config)
