import torch


def batch(config):
    b, m, s = 2, getattr(config, 'motion_queries', 8), getattr(config, 'canvas_size', 32)
    inputs = dict(images=torch.randn(b, 2, 3, config.image_size, config.image_size),
                  image_mask=torch.ones(b, 2, dtype=torch.bool), tokens=torch.ones(b, 5, dtype=torch.long),
                  token_mask=torch.ones(b, 5, dtype=torch.bool),
                  arm_presence=torch.tensor([[1, 0], [1, 1]], dtype=torch.bool))
    targets = dict(actions=torch.randn(b, config.action_horizon, config.action_dim),
                   action_valid=torch.ones(b, config.action_horizon, config.action_dim, dtype=torch.bool),
                   skill_labels=torch.tensor([[0, -1], [1, 5]]), skill_valid=torch.ones(b, 2, dtype=torch.bool),
                   role_images=torch.randn(b, 2, 2, 3, s, s), role_weights=torch.ones(b, 2, 2, s, s),
                   role_valid=torch.ones(b, 2, 2, dtype=torch.bool),
                   trace=torch.rand(b, 2, m, 2), trace_valid=torch.ones(b, 2, m, dtype=torch.bool),
                   gripper=torch.randint(0, 2, (b, 2, m)).float(),
                   gripper_valid=torch.ones(b, 2, m, dtype=torch.bool),
                   future_padded=torch.zeros(b, 2, m, dtype=torch.bool))
    return inputs, targets
