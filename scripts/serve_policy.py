"""Serve official OpenPI checkpoints or this project's π0.5 SFT LoRA checkpoint."""
import argparse
import os
from pathlib import Path
import sys


def command(args):
    runtime = args.openpi_root.resolve()
    return [sys.executable, str(runtime/'scripts/serve_policy.py'), '--env', 'LIBERO',
            '--port', str(args.port), 'policy:checkpoint',
            '--policy.config', args.config, '--policy.dir', args.checkpoint]


def serve_local(args):
    import importlib.util
    import json
    import torch
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / 'src'))
    sys.path.insert(1, str(args.openpi_root / 'packages/openpi-client/src'))
    from openpi.policies.policy_config import create_trained_policy
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    policy = create_trained_policy(args.checkpoint, device=args.device)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if saved['config']['data_format'] != 'libero_sft':
        raise ValueError('Local LIBERO service currently requires the dual-camera libero_sft input adapter')
    metadata = dict(model_type=saved['config']['model_type'], checkpoint=str(Path(args.checkpoint).resolve()),
                    checkpoint_step=saved['step'], lora_rank=saved['config']['lora_rank'],
                    cameras=['image', 'wrist_image'], action_horizon=policy.model.config.action_horizon,
                    fm_steps=args.fm_steps, seed=args.seed, technical_only=args.technical,
                    norm_stats=saved['norm_stats']['provenance'])
    del saved

    class WirePolicy:
        def infer(self, observation):
            return policy.infer(dict(image=observation['observation/image'],
                                     wrist_image=observation['observation/wrist_image'],
                                     state=observation['observation/state'], prompt=observation['prompt']),
                                steps=args.fm_steps)

    # Reuse the upstream wire protocol without importing its separate openpi namespace.
    source = args.openpi_root / 'src/openpi/serving/websocket_policy_server.py'
    spec = importlib.util.spec_from_file_location('openpi_wire_server', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(json.dumps(metadata), flush=True)
    module.WebsocketPolicyServer(WirePolicy(), host='127.0.0.1', port=args.port, metadata=metadata).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Fine-tuned checkpoint directory, including its assets')
    parser.add_argument('--config', default='pi05_libero', help='The OpenPi config used for this checkpoint')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--openpi-root', type=Path, default=Path(__file__).resolve().parents[2]/'openpi')
    parser.add_argument('--lerobot-root', type=Path, help='Checkout of the LeRobot revision pinned by OpenPi')
    parser.add_argument('--print-command', action='store_true', help='Show the command without loading weights')
    parser.add_argument('--runtime', choices=('official', 'local'), default='official')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--fm-steps', type=int, default=10)
    parser.add_argument('--technical', action='store_true', help='Mark a connectivity check, not a scored evaluation')
    args = parser.parse_args()
    if args.runtime == 'local':
        if args.print_command:
            print(f'Local π0.5 checkpoint: {args.checkpoint} on {args.device}, port {args.port}')
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
