"""Dispatch official benchmark commands; the launcher selects the model checkout."""
import argparse
import importlib
import os
from pathlib import Path
import sys

from eskill_bench import ROOT

PROTOCOL_COMMANDS = ('unseen-composition', 'video', 'unpack',
                     'audit', 'relocate', 'restore-dev', 'replay')
COMMANDS = ('visual', 'planner', 'compare', 'grounding-pairs', 'libero', 'libero-plus', 'late-place',
            *PROTOCOL_COMMANDS)


def implementation(command):
    name = 'libero' if command == 'libero-plus' else command.replace('-', '_')
    return importlib.import_module('eskill_bench.' + name)


def late_place(argv):
    paths = [str(ROOT / 'src'), str(ROOT), os.environ.get('PYTHONPATH', '')]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in paths if p))
    # Sample selection and simulator replay both belong to the benchmark checkout.
    script = ROOT / 'scripts/preprocess/libero/late_place.py'
    os.execve(sys.executable, [sys.executable, str(script), *argv], env)


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog='Use visual infer or planner infer for locked checkpoint evaluation.')
    parser.add_argument('--config', type=Path, help='Host paths and environment for locked protocols')
    parser.add_argument('command', choices=COMMANDS)
    parser.add_argument('options', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    scheduled_libero = command in ('libero', 'libero-plus') and (
        args.config is not None or (args.options and (
            not args.options[0].startswith('-') or args.options[0] in ('--full', '--with-plus')
        ))
    )
    if scheduled_libero or command in PROTOCOL_COMMANDS or (command in ('visual', 'planner') and args.options[:1] == ['infer']):
        from eskill_bench import protocols
        if command in ('visual', 'planner'):
            variants = ('frozen', 'joint', 'boundary') if command == 'planner' else ('bbox', 'bbox_gac', 'legacy_vae', 'vae_gac', 'klal')
            return protocols.main(args.options, args.config, variants, prog=f'{parser.prog} {command}')
        return protocols.main([command, *args.options], args.config, prog=parser.prog)
    if args.config:
        parser.error('--config applies to locked evaluation protocols')
    sys.argv[1:] = args.options
    if command == 'late-place':
        late_place(sys.argv[1:])
    module = implementation(command)
    if command in ('libero', 'libero-plus'):
        module.main(command.replace('-', '_'))
    else:
        module.main()
