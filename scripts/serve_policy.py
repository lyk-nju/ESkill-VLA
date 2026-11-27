"""Serve official OpenPI checkpoints or local π0.5 SFT / ESkill checkpoints."""
import argparse
import os
from pathlib import Path
import sys


def command(args):
    runtime = args.openpi_root.resolve()
    return [sys.executable, str(runtime/'scripts/serve_policy.py'), '--env', 'LIBERO',
            '--port', str(args.port), 'policy:checkpoint',
            '--policy.config', args.config, '--policy.dir', args.checkpoint]


class EpisodeSeededPolicy:
    """Apply the sampled LIBERO protocol's seed at the first request of each episode."""

    def __init__(self, delegate, seed_rng):
        self.delegate = delegate
        self.seed_rng = seed_rng

    def infer(self, observation):
        observation = dict(observation)
        if 'evaluation/episode_seed' in observation:
            seed = observation.pop('evaluation/episode_seed')
            if type(seed) is not int or seed != 7:
                raise ValueError('Sampled LIBERO evaluation requires episode seed 7')
            self.seed_rng(seed)
        return self.delegate.infer(observation)


def serve_local(args):
    import importlib.util
    import json
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.openpi.revision import git_revision

    project = args.policy_source.resolve()
    sys.path.insert(0, str(project / 'src'))
    sys.path.insert(1, str(args.openpi_root / 'packages/openpi-client/src'))
    sys.path.insert(2, str(project))
    from openpi.policies.policy_config import create_trained_policy
    from openpi.policies.libero_policy import LiberoWirePolicy
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    policy = create_trained_policy(args.checkpoint, device=args.device,
                                   inference_precision=args.inference_precision, tokenizer_path=args.tokenizer_path)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if args.record_predictions is not None:
        import numpy as np
        from itertools import count
        directory = args.record_predictions
        directory.mkdir(parents=True, exist_ok=True)
        sequence = count()
        def record(observation, prediction, actions):
            keys = ('current_map', 'skill_id', 'start', 'end', 'trace')
            data = {key: prediction[key][0].detach().float().cpu().numpy() for key in keys if key in prediction}
            data.update(actions=actions, task=observation['prompt'], image=observation['images'][0])
            with (directory / f'call_{next(sequence):06d}.npz').open('xb') as stream:
                np.savez_compressed(stream, **data)
        policy.prediction_sink = record
    wire_policy = LiberoWirePolicy(policy, saved['config'], steps=args.fm_steps)
    metadata = dict(model_revision=git_revision(project), model_type=saved['config']['model_type'], checkpoint=str(Path(args.checkpoint).resolve()),
                    checkpoint_step=saved['step'], lora_rank=saved['config']['lora_rank'],
                    cameras=wire_policy.cameras,
                    requires_camera_calibration=wire_policy.requires_camera_calibration,
                    image_resize_on_server=wire_policy.image_resize_on_server,
                    requires_state=wire_policy.requires_state,
                    state_dim=8 if wire_policy.requires_state else 0,
                    action_horizon=policy.model.config.action_horizon,
                    fm_steps=args.fm_steps, seed=args.seed, technical_only=args.technical,
                    inference_precision=policy.precision,
                    norm_stats=saved['norm_stats']['provenance'])
    if args.reset_per_episode:
        wire_policy = EpisodeSeededPolicy(wire_policy, torch.manual_seed)
        metadata.update(evaluation_protocol='sample_episode_seed7_dynamic_v1', policy_rng_scope='episode',
                        policy_device=args.device)
    if args.record_predictions is not None:
        metadata['prediction_recording'] = dict(directory=str(args.record_predictions), schema_version=2,
                                                source='actual_policy_calls')
    del saved

    # Reuse the upstream wire protocol without importing its separate openpi namespace.
    source = args.openpi_root / 'src/openpi/serving/websocket_policy_server.py'
    spec = importlib.util.spec_from_file_location('openpi_wire_server', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(json.dumps(metadata), flush=True)
    module.WebsocketPolicyServer(wire_policy, host='127.0.0.1', port=args.port, metadata=metadata).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Fine-tuned checkpoint directory, including its assets')
    parser.add_argument('--config', default='pi05_libero', help='The OpenPi config used for this checkpoint')
    parser.add_argument('--policy-source', type=Path, default=Path(__file__).resolve().parents[1],
                        help='Model checkout for local checkpoint inference')
    parser.add_argument('--reset-per-episode', action='store_true',
                        help='Use the sampled LIBERO seed-7 request protocol')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--openpi-root', type=Path, default=Path(__file__).resolve().parents[2]/'openpi')
    parser.add_argument('--lerobot-root', type=Path, help='Checkout of the LeRobot revision pinned by OpenPi')
    parser.add_argument('--print-command', action='store_true', help='Show the command without loading weights')
    parser.add_argument('--runtime', choices=('official', 'local'), default='official')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--fm-steps', type=int, default=10)
    parser.add_argument('--inference-precision', choices=('float32', 'bfloat16'),
                        help='Local inference compute precision; defaults to checkpoint training precision')
    parser.add_argument('--technical', action='store_true', help='Mark a connectivity check, not a scored evaluation')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--tokenizer-path', type=Path, help='Local equivalent of the saved tokenizer')
    parser.add_argument('--record-predictions', type=Path, help='Save actual per-request actions and available Current/plan predictions')
    args = parser.parse_args()
    if args.reset_per_episode and (args.runtime != 'local' or args.seed != 7):
        parser.error('--reset-per-episode requires local inference with seed 7')
    if args.runtime != 'local' and args.inference_precision is not None:
        parser.error('--inference-precision requires --runtime local')
    if args.runtime != 'local' and (args.tokenizer_path is not None or args.record_predictions is not None):
        parser.error('Prediction recording and tokenizer overrides require --runtime local')
    if args.runtime == 'local':
        if args.print_command:
            print(f'Local checkpoint: {args.checkpoint} on {args.device}, port {args.port}')
            return
        serve_local(args)
        return
    invocation = command(args)
    if args.print_command:
        import shlex
        print(shlex.join(invocation))
        return
    runtime = args.openpi_root.resolve()
    env = dict(os.environ)
    # The policy runtime owns its openpi namespace; the simulator imports only openpi_client.
    lerobot = (args.lerobot_root or runtime.parent/'lerobot-openpi').resolve()
    env['PYTHONPATH'] = os.pathsep.join(map(str, [runtime/'src', runtime/'packages/openpi-client/src', lerobot]))
    os.chdir(runtime)
    os.execve(sys.executable, invocation, env)


if __name__ == '__main__':
    main()
