"""Resume the converted 30k checkpoint to 40k, then evaluate the four LIBERO suites."""
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT.parent
DATA = Path('/data1/yuankai/robot-research/workspace/experiments/eskill_libero_unified/run_002')
CONFIG = Path(__file__).with_name('train.json')
SIM = '/data1/yuankai/robot-research/workspace/envs/libero-eval/bin/python'
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def environment(gpus):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, PYTHONPATH=f'{PROJECT / "src"}:{PROJECT}',
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
    started = time.time()
    with (DATA / f'{name}.log').open('x') as log:
        process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(command=command, pid=process.pid, started_unix=started)
        write(DATA / f'{name}.json', record)
        print(f'{name}: pid={process.pid}', flush=True)
        try:
            code = process.wait(timeout=timeout)
        finally:
            stop(process)
            record.update(returncode=process.returncode, seconds=time.time()-started)
            write(DATA / f'{name}.json', record)
        if code:
            raise RuntimeError(f'{name} exited with {code}; inspect its log')


def evaluate(gpu, suite, checkpoint):
    name = f'eval_{suite}'
    port = 8150 + gpu
    output = DATA / 'evaluation' / 'checkpoint_040000' / suite
    env = environment(str(gpu))
    command = [sys.executable, str(PROJECT/'scripts/serve_policy.py'), '--runtime', 'local',
               '--checkpoint', str(checkpoint), '--port', str(port), '--seed', '7',
               '--fm-steps', '10', '--openpi-root', str(WORKSPACE/'openpi')]
    with (DATA/f'{name}_server.log').open('x') as log:
        server = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        write(DATA/f'{name}_server.json', dict(command=command, pid=server.pid, gpu=gpu))
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            for _ in range(120):
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
                raise TimeoutError(f'{suite} policy server startup timed out')
            client = [SIM, str(PROJECT/'examples/libero/main.py'), '--port', str(port),
                      '--task-suite-name', suite, '--num-trials-per-task', '5', '--seed', '7',
                      '--replan-steps', '5', '--num-steps-wait', '10', '--output', str(output),
                      '--checkpoint-tag', str(checkpoint), '--video', 'all']
            simenv = dict(env, PYTHONPATH=str(PROJECT), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                          OPENBLAS_NUM_THREADS='2', MUJOCO_EGL_DEVICE_ID=str(gpu))
            job(name, client, simenv, 4*3600)
            summary = json.loads((output/'summary.json').read_text())
            if summary['errors'] or summary['episodes'] != 50:
                raise RuntimeError(f'{suite} evaluation incomplete: {summary}')
            return summary
        finally:
            stop(server)


def main():
    status = dict(phase='training', cumulative_start=30000, cumulative_target=40000,
                  started_unix=time.time(), pipeline_pid=os.getpid())
    write(DATA/'status.json', status)
    try:
        job('train', [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                      str(PROJECT/'scripts/train.py'), '--config', str(CONFIG),
                      '--resume', str(DATA/'initial.pt')], environment('0,1,2,3'), 24*3600)
        checkpoint = DATA/'train/checkpoint_020000.pt'
        proof = json.loads((DATA/'train/training_check.json').read_text())
        if not checkpoint.is_file() or proof['step'] != 20000 or not proof['all_ranks_equal']:
            raise RuntimeError('40k training completion proof missing')
        status.update(phase='evaluating', checkpoint=str(checkpoint))
        write(DATA/'status.json', status)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {suite: pool.submit(evaluate, gpu, suite, checkpoint)
                       for gpu, suite in enumerate(SUITES)}
            summaries = {suite: future.result() for suite, future in futures.items()}
        write(DATA/'evaluation.json', dict(cumulative_checkpoint_step=40000, suites=summaries))
        status.update(phase='complete', finished_unix=time.time())
    except Exception as error:
        status.update(phase='failed', error=f'{type(error).__name__}: {error}', finished_unix=time.time())
        raise
    finally:
        write(DATA/'status.json', status)


if __name__ == '__main__':
    main()
