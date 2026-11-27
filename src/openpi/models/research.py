"""Planner architectures available on the motion-planner branch."""
GROUNDING = ('visual_primitive', 'motion_planner')
PLANNERS = ('trajectory', 'boundary_first', 'joint_decode')
SUBJECT = False
ACTION_ABLATIONS = False
DEFAULT_GROUNDING = 'visual_primitive'
DEFAULT_PLANNER = 'trajectory'


def build_planner(config):
    from openpi.models.motion_planner import MotionPlanner
    from openpi.models.planner_variants import EarlyBoundaryPlanner, EightSlotPlanner
    return {'trajectory': EightSlotPlanner, 'boundary_first': EarlyBoundaryPlanner, 'joint_decode': MotionPlanner}[config.planner_version](config)
