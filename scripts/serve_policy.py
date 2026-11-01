"""Serve a LIBERO fine-tuned pi0.5 checkpoint using the workspace's official OpenPi runtime."""
import argparse
import os
from pathlib import Path
import sys


def command(args):
    runtime = args.openpi_root.resolve()
    return [sys.executable, str(runtime/'scripts/serve_policy.py'), '--env', 'LIBERO',
            '--port', str(args.port), 'policy:checkpoint',
            '--policy.config', args.config, '--policy.dir', args.checkpoint]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Fine-tuned checkpoint directory, including its assets')
    parser.add_argument('--config', default='pi05_libero', help='The OpenPi config used for this checkpoint')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--openpi-root', type=Path, default=Path(__file__).resolve().parents[2]/'openpi')
    parser.add_argument('--lerobot-root', type=Path, help='Checkout of the LeRobot revision pinned by OpenPi')
    parser.add_argument('--print-command', action='store_true', help='Show the command without loading weights')
    args = parser.parse_args()
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
