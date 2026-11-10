"""Per-node supervisor for one Delta-flow production run on 4 nodes x 8 GPUs (adapted from launch_supervised.py).

Usage (called by deltaflow_task.sh on every node, DLC env RANK/WORLD_SIZE/MASTER_ADDR present):
    python deltaflow_supervise.py <run_root R> <session> <port_base>
Up to 3 attempts. Before each attempt node 0 writes a plan: resume from the newest complete training state in
R/run/production/checkpoints/state (saved at epoch 10/20/30), or, when none exists, move any partial output aside
and start fresh. A node that sees a peer fail stops its own launcher; a 30-minute all-idle stall stops all.
NAS barriers live in R/run/retry/s<session>/attempt-<n>/. Exit 0 only when all four launchers exited 0.
"""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
R = Path(sys.argv[1]).resolve()
SESSION = int(sys.argv[2])
PORT_BASE = int(sys.argv[3])
NODE = int(os.environ['RANK'])
NODES, WORLD, ATTEMPTS = 4, 32, 3
RUN = R / 'run'
OUT = RUN / 'production'
STATE = OUT / 'checkpoints' / 'state'
CURRENT = {}


def log(*args):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), f'[supervise node {NODE}]', *args, flush=True)


def write(path, text):
    pending = path.with_name(f'{path.name}.tmp{os.getpid()}')
    pending.write_text(text)
    pending.replace(path)


def wait_for_files(paths, seconds=900):
    deadline = time.monotonic() + seconds
    while not all(path.is_file() for path in paths):
        if time.monotonic() >= deadline:
            missing = [str(p) for p in paths if not p.is_file()]
            raise TimeoutError(f'Four nodes did not reach a shared barrier; missing {missing}')
        time.sleep(2)


def gpu_apps():
    return subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                   text=True, timeout=20).strip()


def gpu_idle():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                         text=True, timeout=8)
        utilization = [int(line.strip()) for line in result.splitlines()]
        return len(utilization) == 8 and max(utilization) == 0
    except (subprocess.SubprocessError, ValueError):
        return False


def complete_state(path):
    return ((path / 'trainer_state.json').is_file()
            and (path / 'pytorch_model' / 'mp_rank_00_model_states.pt').is_file()
            and len(list((path / 'pytorch_model').glob('*_optim_states.pt'))) == WORLD)


def newest_state():
    if not STATE.is_dir():
        return None
    found = []
    for path in STATE.iterdir():
        match = re.fullmatch(r'step_(\d+)', path.name)
        if match and path.is_dir() and complete_state(path):
            found.append((int(match.group(1)), path))
    return max(found)[1] if found else None


def training_idle(attempt_dir, now):
    threshold = now - 1800
    for rank in range(NODES):
        path = attempt_dir / f'gpu-idle-{rank}.json'
        if not path.is_file():
            return False
        report = json.loads(path.read_text())
        if report['sampled'] < now - 90 or report['idle_since'] is None or report['idle_since'] > threshold:
            return False
    activity = list(attempt_dir.glob('node-*.log')) + list((OUT / 'checkpoints').rglob('*'))
    if not activity:
        return False
    for path in activity:
        try:
            if path.stat().st_mtime > threshold:
                return False
        except FileNotFoundError:
            return False
    return True


def wait_for_process(process, peers, attempt_dir):
    next_sample, idle_since = 0.0, None
    stalled = attempt_dir / 'training-stalled'
    while process.poll() is None:
        if time.monotonic() >= next_sample:
            now = time.time()
            idle_since = (idle_since if idle_since is not None else now) if gpu_idle() else None
            write(attempt_dir / f'gpu-idle-{NODE}.json', json.dumps({'sampled': now, 'idle_since': idle_since}))
            next_sample = time.monotonic() + 30
            if NODE == 0 and training_idle(attempt_dir, now):
                write(stalled, 'All four nodes idle and no log/checkpoint progress for 30 minutes.\n')
        if stalled.exists() or any(path.is_file() and path.read_text().strip() != '0' for path in peers):
            log('peer failure or stall detected; stopping local launcher')
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                return process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                return process.wait(timeout=60)
        time.sleep(2)
    return process.returncode


def release_gpus():
    deadline = time.monotonic() + 300
    while gpu_apps():
        if time.monotonic() >= deadline:
            raise TimeoutError('GPU processes did not exit after a failed attempt')
        if time.monotonic() >= deadline - 180:
            subprocess.call(['pkill', '-9', '-f', 'scripts/wholebody/train_deltaflow.py'])
        time.sleep(5)


def plan_attempt(attempt, attempt_dir):
    """Node 0 only: decide resume vs fresh start (all nodes are idle at this point)."""
    state = newest_state()
    if state is not None:
        plan = {'resume': str(state), 'step': int(state.name.split('_')[1])}
    else:
        if OUT.exists():
            aside = RUN / f'discarded-s{SESSION}-a{attempt}-{time.strftime("%Y%m%dT%H%M%S")}'
            OUT.rename(aside)
            log('no complete training state; partial output moved to', aside)
        plan = {'resume': None, 'step': 0}
    write(attempt_dir / 'plan.json', json.dumps(plan) + '\n')


def run():
    if NODE not in range(NODES):
        raise ValueError('Unexpected node rank')
    session_root = RUN / 'retry' / f's{SESSION}'
    for attempt in range(1, ATTEMPTS + 1):
        attempt_dir = session_root / f'attempt-{attempt}'
        attempt_dir.mkdir(parents=True, exist_ok=True)
        ready = [attempt_dir / f'ready-{rank}' for rank in range(NODES)]
        done = [attempt_dir / f'exit-{rank}' for rank in range(NODES)]
        CURRENT['exit'] = done[NODE]
        if ready[NODE].exists():
            raise RuntimeError(f'This node already joined {attempt_dir} (session counters out of sync?)')
        deadline = time.monotonic() + 180
        while gpu_apps():
            if time.monotonic() >= deadline:
                raise RuntimeError('GPUs busy before launch')
            time.sleep(5)
        write(ready[NODE], f'{os.getpid()}\n')
        wait_for_files(ready)
        if NODE == 0:
            plan_attempt(attempt, attempt_dir)
        wait_for_files([attempt_dir / 'plan.json'], seconds=300)
        plan = json.loads((attempt_dir / 'plan.json').read_text())
        if any(path.exists() for path in done):
            raise RuntimeError('A result exists before the attempt began')
        port = PORT_BASE + 3 * ((SESSION - 1) % 3) + attempt - 1
        log(f'attempt {attempt} port {port} resume={plan["resume"]}')
        with (attempt_dir / f'node-{NODE}.log').open('ab') as stream:
            process = subprocess.Popen(['bash', str(HERE / 'deltaflow_attempt.sh'), str(R), str(port),
                                        plan['resume'] or 'none'],
                                       stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            status = wait_for_process(process, [path for i, path in enumerate(done) if i != NODE], attempt_dir)
        log(f'attempt {attempt} local launcher exit {status}')
        write(done[NODE], f'{status}\n')
        wait_for_files(done, seconds=1800)
        if all(path.read_text().strip() == '0' for path in done):
            return 0
        if attempt < ATTEMPTS:
            release_gpus()
            drained = [attempt_dir / f'gpu-released-{rank}' for rank in range(NODES)]
            write(drained[NODE], 'idle\n')
            wait_for_files(drained, seconds=600)
    return 1


def main():
    log_dir = RUN / 'node-logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = run()
    except Exception as error:  # noqa: BLE001 - report every failure to the NAS and exit nonzero
        write(log_dir / f's{SESSION}-supervisor-error-{NODE}.json',
              json.dumps({'type': type(error).__name__, 'detail': str(error)}) + '\n')
        log('ERROR', type(error).__name__, error)
        result = 1
        exit_file = CURRENT.get('exit')
        if exit_file is not None and not exit_file.exists():
            write(exit_file, '1\n')      # lets the peers stop their launchers instead of waiting for timeouts
    write(log_dir / f's{SESSION}-node-{NODE}.exit', f'{result}\n')
    return result


if __name__ == '__main__':
    sys.exit(main())
