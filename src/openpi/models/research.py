"""Visual supervision methods available on this research branch."""
GROUNDING = ('vae', 'bbox', 'vae_gac', 'bbox_gac', 'visual_primitive')
PLANNERS = ()
SUBJECT = True
ACTION_ABLATIONS = False
DEFAULT_GROUNDING = 'visual_primitive'
DEFAULT_PLANNER = 'v2_joint'


def training_vae(config):
    from openpi.models.grounding import load_vae

    if config.vae_path:
        return load_vae(config.vae_path)
    if not config.smoke:
        raise ValueError('Set vae_path to a local pretrained AutoencoderKL directory.')
    from diffusers import AutoencoderKL
    return AutoencoderKL(
        in_channels=3, out_channels=3, down_block_types=('DownEncoderBlock2D',) * 2,
        up_block_types=('UpDecoderBlock2D',) * 2, block_out_channels=(16, 32),
        layers_per_block=1, latent_channels=4, norm_num_groups=8, sample_size=config.model.canvas_size)
