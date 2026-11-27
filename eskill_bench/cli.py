"""Dispatch official benchmark commands; the launcher selects the model checkout."""
import argparse
import importlib
import os
import sys

from eskill_bench import ROOT

COMMANDS = ('visual', 'planner', 'compare', 'probe', 'grounding-pairs', 'libero', 'libero-plus', 'late-place', 'handoff')


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
    parser = argparse.ArgumentParser(description=__doc__, usage='%(prog)s COMMAND [benchmark options]')
    parser.add_argument('command', choices=COMMANDS)
    command = parser.parse_args(sys.argv[1:2]).command
    del sys.argv[1]
    if command == 'late-place':
        late_place(sys.argv[1:])
    module = implementation(command)
    if command in ('libero', 'libero-plus'):
        module.main(command.replace('-', '_'))
    else:
        module.main()
