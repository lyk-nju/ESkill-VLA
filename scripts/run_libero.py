"""Train an explicit LIBERO recipe and evaluate at milestones; no implicit experiment config."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

PROJECT = Path(__file__).resolve().parents[1]
SETTINGS = DATA = None
SIM = None
OPENPI = PROJECT.parent/'openpi'
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def environment(gpus):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, PYTHONPATH=f'{PROJECT/"src"}:{PROJECT}',
                OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                HF_HUB_OFFLINE='1', NCCL_DEBUG='WARN', PYTHONUNBUFFERED='1',
                NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def job(name, command, env, timeout):
    record = dict(command=command, started_unix=time.time())
    with (DATA/f'{name}.log').open('x') as log:
        process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        record['pid'] = process.pid
        write(DATA/f'{name}.json', record)
        try:
            code = process.wait(timeout=timeout)
        finally:
            stop(process)
            record.update(returncode=process.returncode, seconds=time.time()-record['started_unix'])
            write(DATA/f'{name}.json', record)
        if code:
            raise RuntimeError(f'{name} exited with {code}; inspect its log')


def evaluate(gpu, suite, step, checkpoint):
    name = f'eval_{step:06d}_{suite}'
    port = 8160+gpu
    output = DATA/'evaluation'/f'checkpoint_{step:06d}'/suite
    env = environment(str(gpu))
    command = [sys.executable, str(PROJECT/'scripts/serve_policy.py'), '--runtime', 'local',
               '--checkpoint', str(checkpoint), '--port', str(port), '--seed', '7', '--fm-steps', '10',
               '--openpi-root', str(OPENPI)]
    with (DATA/f'{name}_server.log').open('x') as log:
        server = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        write(DATA/f'{name}_server.json', dict(command=command, pid=server.pid, gpu=gpu))
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            for _ in range(150):
                if server.poll() is not None:
                    raise RuntimeError(f'{suite} policy server exited')
                try:
                    with opener.open(f'http://127.0.0.1:{port}/healthz', timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError(f'{suite} server startup timed out')
            client = [SIM, str(PROJECT/'examples/libero/main.py'), '--port', str(port),
                      '--task-suite-name', suite, '--num-trials-per-task', '5', '--seed', '7',
                      '--replan-steps', '5', '--num-steps-wait', '10', '--output', str(output),
                      '--checkpoint-tag', str(checkpoint), '--video', 'all']
            simenv = dict(env, PYTHONPATH=str(PROJECT), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                          OPENBLAS_NUM_THREADS='2', MUJOCO_EGL_DEVICE_ID=str(gpu))
            job(name, client, simenv, 4*3600)
            result = json.loads((output/'summary.json').read_text())
            if result['errors'] or result['episodes'] != 50:
                raise RuntimeError(f'Incomplete evaluation: {suite}')
            return result
        finally:
            stop(server)


def main():
    global SETTINGS, DATA, SIM, OPENPI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--eval-every', type=int, default=10000)
    parser.add_argument('--gpus', type=int, nargs='+', required=True, help='Training GPU IDs; first four also run evaluation')
    parser.add_argument('--sim-python', type=Path, required=True, help='LIBERO evaluation Python interpreter')
    parser.add_argument('--resume-step', type=int, default=0,
                        help='Completed evaluation milestone: evaluate it first, then resume training')
    parser.add_argument('--openpi-root', type=Path, default=OPENPI, help='Official OpenPI checkout providing the serving/client code')
    args = parser.parse_args()
    if len(args.gpus) < len(SUITES) or len(set(args.gpus)) != len(args.gpus):
        parser.error('Specify at least four distinct GPUs for the four concurrent evaluation suites')
    # Resolving a venv Python symlink selects the base interpreter and loses its packages.
    SIM, OPENPI = str(args.sim_python.expanduser().absolute()), args.openpi_root.resolve()
    SETTINGS = json.loads(args.config.read_text())
    from openpi.training.config import TrainConfig
    config = TrainConfig.from_dict(SETTINGS)
    if config.model_type != 'pi05' and config.training_stage != 'action_execution':
        parser.error('Closed-loop evaluation needs an action policy; train perception or planning with scripts.train')
    DATA = Path(SETTINGS['output_dir']).parent
    budget = SETTINGS['num_steps']
    if args.eval_every <= 0 or budget <= 0:
        parser.error('Training budget and --eval-every must be positive')
    if (args.resume_step < 0 or args.resume_step > budget
            or (args.resume_step != budget and args.resume_step % args.eval_every)):
        parser.error('--resume-step must be a completed evaluation milestone within the training budget')
    if args.resume_step and not (Path(SETTINGS['output_dir'])/f'checkpoint_{args.resume_step:06d}.pt').is_file():
        parser.error('Requested milestone checkpoint does not exist')
    # Exercise the real evaluation interpreter and task setup before spending GPU hours.
    subprocess.run([SIM, str(PROJECT/'examples/libero/main.py'), '--list-tasks', '--task-suite-name', 'all'],
                   cwd=PROJECT, env=dict(environment(str(args.gpus[0])), PYTHONPATH=str(PROJECT)),
                   stdout=subprocess.DEVNULL, check=True)
    DATA.mkdir(parents=True, exist_ok=True)
    write(DATA/'runtime.json', dict(
        command=sys.argv, python=sys.executable, gpus=args.gpus,
        world_size=len(args.gpus), global_batch_size=len(args.gpus)*SETTINGS['batch_size'],
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--short'], cwd=PROJECT, text=True),
        sim_python=SIM, openpi_root=str(OPENPI)))
    status = dict(phase='training', cumulative_start=args.resume_step, cumulative_target=SETTINGS['num_steps'],
                  started_unix=time.time(), pipeline_pid=os.getpid())
    write(DATA/'status.json', status)
    checkpoint = Path(SETTINGS['output_dir'])/'checkpoint.pt'
    try:
        targets = list(range(args.eval_every, budget, args.eval_every)) + [budget]
        for target in (t for t in targets if t >= args.resume_step):
            if target != args.resume_step:
                settings = dict(SETTINGS, num_steps=target)
                stage_config = DATA/f'config_{target:06d}.json'
                write(stage_config, settings)
                command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={len(args.gpus)}',
                           str(PROJECT/'scripts/train.py'), '--config', str(stage_config)]
                if target != targets[0] or args.resume_step:
                    command += ['--resume', str(checkpoint)]
                status.update(phase='training', stage_target=target, cumulative_stage_target=target)
                write(DATA/'status.json', status)
                job(f'train_{target:06d}', command, environment(','.join(map(str, args.gpus))), 24*3600)
            proof = json.loads((checkpoint.parent/'training_check.json').read_text())
            if proof['step'] != target or not proof['all_ranks_equal']:
                raise RuntimeError('Training completion check failed')
            milestone = checkpoint.parent/f'checkpoint_{target:06d}.pt'
            if not milestone.exists():
                milestone.hardlink_to(checkpoint)  # Short technical runs also retain their evaluated checkpoint.
            cumulative = target
            status.update(phase='evaluating', cumulative_step=cumulative, checkpoint=str(milestone))
            write(DATA/'status.json', status)
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {suite: pool.submit(evaluate, gpu, suite, cumulative, milestone)
                           for gpu, suite in zip(args.gpus, SUITES)}
                summaries = {suite: result.result() for suite, result in futures.items()}
            write(DATA/f'evaluation_{cumulative:06d}.json', dict(cumulative_step=cumulative, suites=summaries))
        status.update(phase='complete', finished_unix=time.time())
    except Exception as error:
        status.update(phase='failed', error=f'{type(error).__name__}: {error}', finished_unix=time.time())
        raise
    finally:
        write(DATA/'status.json', status)


if __name__ == '__main__':
    main()
