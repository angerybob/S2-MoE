#!/usr/bin/env python3
"""Fixed post-fix GPU/SSD speed matrix; no downloads or machine setting changes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import time

PROMPTS = {
    'math': 'Solve step by step: A shop sells 12 notebooks at 3 dollars each. What is the total price?',
    'science': 'Explain why the sky looks blue in two short sentences.',
}
CONFIGS = [(1, 1), (1, 4), (4, 1), (4, 2), (8, 2), (8, 4)]


def stamp(path):
    s = path.stat()
    return {'size': s.st_size, 'mtime_ns': s.st_mtime_ns}


def machine():
    paths = ['/proc/meminfo', '/proc/loadavg', '/proc/diskstats',
             '/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq']
    paths += [str(p) for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')]
    result = {}
    for path in paths:
        try:
            result[path] = Path(path).read_text().strip()
        except (OSError, TypeError):
            result[path] = None
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--frozen-math', required=True, type=Path)
    ap.add_argument('--timeout', type=int, default=21600)
    args = ap.parse_args()
    repo, output, model = args.repo.resolve(), args.output.resolve(), args.model.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = repo / 'build-next/bin/llama-speculative-qwen35'
    runtime = [binary, *sorted(binary.parent.glob('lib*.so'))]
    stamps = {str(p): stamp(p) for p in runtime + [model]}
    runtime_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime}
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo), 'source must be clean'
    frozen = json.loads(args.frozen_math.read_text())[:32]
    assert len(frozen) == 32
    order = list(CONFIGS)
    random.Random(351018).shuffle(order)
    suites = []

    def schedule(name, prompt, reserve, pruning, configs, repetition):
        rows = [{'id': 'warmup-ar', 'mode': 'AR', 'depth': 0, 'experts': 8, 'warmup': True}]
        rows.append({'id': 'ar-start', 'mode': 'AR', 'depth': 0, 'experts': 8, 'warmup': False})
        for i, (d, k) in enumerate(configs):
            rows.append({'id': f'spec-{i}-d{d}-k{k}', 'mode': 'SPEC', 'depth': d, 'experts': k, 'warmup': False})
            if i + 1 == (len(configs) + 1) // 2:
                rows.append({'id': 'ar-middle', 'mode': 'AR', 'depth': 0, 'experts': 8, 'warmup': False})
        rows.append({'id': 'ar-end', 'mode': 'AR', 'depth': 0, 'experts': 8, 'warmup': False})
        # Capacity controls have two measured AR runs, plus excluded warmup.
        if not configs:
            assert len(rows) == 3
        suites.append({'id': name, 'prompt': prompt, 'reserve_mib': reserve,
                       'pruning': pruning, 'repetition': repetition, 'plan': rows})

    schedule('capacity-before', 'math', 512, False, [], 0)
    schedule('linear-r1', 'math', 3072, False, order, 1)
    schedule('prune-r1', 'math', 3072, True, order, 1)
    schedule('prune-r2', 'math', 3072, True, order[::-1], 2)
    schedule('linear-r2', 'math', 3072, False, order[::-1], 2)
    schedule('science-linear', 'science', 3072, False, [(4, 2), (4, 2)], 0)
    schedule('science-prune', 'science', 3072, True, [(4, 2), (4, 2)], 0)
    schedule('capacity-after', 'math', 512, False, [], 0)
    total = sum(len(s['plan']) for s in suites)
    assert total == 58
    report = {'schema': 1, 'status': 'running', 'source_revision': revision, 'source_dirty': False,
              'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'runtime_sha256': runtime_hashes, 'runtime_and_model_stamps': stamps,
              'frozen_math_sha256': hashlib.sha256(args.frozen_math.read_bytes()).hexdigest(),
              'started_at': datetime.now(timezone.utc).isoformat(), 'planned_cases': total,
              'prompts': PROMPTS, 'configs': CONFIGS, 'tokens_per_case': 32,
              'throughput_definition': '31 post-prefill decode tokens / generation seconds; prefill and model load excluded',
              'order': 'capacity-before; linear/prune/prune/linear (reverse configurations in round 2); science controls; capacity-after',
              'prune_settings': {'tpot': 5000, 'bandwidth_argument': .15, 'score_threshold': 1,
                                 'note': 'fixed heuristic settings, not an automatic calibration'},
              'budget_mib': 8192, 'audit': False, 'approximate_reuse': False, 'branches': 1,
              'initial_machine': machine(), 'suites': suites, 'cases': []}

    def save():
        temp = output / 'report.tmp'
        temp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        temp.replace(output / 'report.json')

    def immutable():
        assert all(stamp(Path(p)) == s for p, s in stamps.items()), 'model or runtime changed'

    references = {}
    telemetry = None
    save()
    start = time.monotonic()
    try:
        with (output / 'tegrastats.log').open('wb') as log:
            telemetry = subprocess.Popen(['tegrastats', '--interval', '1000'], stdout=log, stderr=subprocess.STDOUT)
        for suite in suites:
            name = suite['id']
            folder = output / name
            folder.mkdir()
            lines = []
            for item in suite['plan']:
                item['tokens_file'] = str(folder / (item['id'] + '-tokens.json'))
                item['timing_file'] = str(folder / (item['id'] + '-timing.json'))
                values = (item['tokens_file'], item['timing_file'], PROMPTS[suite['prompt']])
                lines.append(f"{item['mode']} {item['depth']} {item['experts']} " + ' '.join(json.dumps(v) for v in values))
            plan = folder / 'plan.txt'
            plan.write_text('\n'.join(lines) + '\n')
            command = [str(binary), '-m', str(model), '-ngl', '99', '-c', '128', '-ub', '32', '-t', '8',
                       '-n', '32', '--ignore-eos', '--quiet', '--no-mmap', '--ssd-moe',
                       '--memory-budget-mib', '8192', '--runtime-reserve-mib', str(suite['reserve_mib']),
                       '--moe-reuse-expert-cap', '0', '--draft-branches', '1', '--benchmark-plan', str(plan)]
            if suite['pruning']:
                command += ['--prune', '1', '--prune-tpot', '5000', '--prune-bandwidth', '0.15', '--prune-score-thresh', '1']
            suite.update(command=command, start_machine=machine(), started_at=datetime.now(timezone.utc).isoformat())
            save()
            print('START', name, len(suite['plan']), flush=True)
            index = 0

            def harvest():
                nonlocal index
                while index < len(suite['plan']):
                    item = suite['plan'][index]
                    try:
                        tokens = json.loads(Path(item['tokens_file']).read_text())
                        timing = json.loads(Path(item['timing_file']).read_text())
                    except (OSError, json.JSONDecodeError):
                        return
                    immutable()
                    if item['mode'] == 'AR':
                        references.setdefault(suite['prompt'], tokens)
                    phases = sum(timing[k] for k in ('draft_ms', 'verify_with_history_ms', 'snapshot_ms', 'restore_ms', 'commit_ms', 'ar_decode_ms', 'sampling_ms'))
                    valid = (len(tokens) == 32 and tokens == references[suite['prompt']] and timing['generated'] == 32
                             and timing['decode_tokens'] == 31 and not timing['audit'] and not timing['approximate_expert_reuse']
                             and timing['quiet'] and not timing['logits_traced'] and timing['batch_probe_at'] == -1
                             and timing['ssd'] and timing['direct_io'] and timing['ssd_resident_hits'] > 0
                             and timing['ssd_read_bytes'] > 0 and timing['pruning'] == suite['pruning']
                             and timing['model_buffer_bytes'] + timing['runtime_accounted_bytes'] <= 8192 * 2**20
                             and timing['runtime_accounted_bytes'] <= suite['reserve_mib'] * 2**20
                             and 0 <= phases <= timing['generation_ms'] + .1
                             and timing['ssd_fetch_ms'] <= timing['generation_ms'] + .1)
                    if suite['prompt'] == 'math':
                        valid = valid and tokens == frozen
                    record = {**item, 'suite': name, 'prompt_id': suite['prompt'], 'pruning': suite['pruning'],
                              'reserve_mib': suite['reserve_mib'], 'timing': timing, 'tokens': tokens, 'valid': bool(valid),
                              'decode_tps': 31000 / timing['generation_ms'], 'other_ms': timing['generation_ms'] - phases,
                              'completed_at': datetime.now(timezone.utc).isoformat(), 'machine': machine()}
                    report['cases'].append(record)
                    index += 1
                    save()
                    print(f"END {name}/{item['id']} {len(report['cases'])}/{total} valid={valid} tps={record['decode_tps']:.4f}", flush=True)
                    if not valid:
                        raise RuntimeError('token, allocation or timing validation failed: ' + name + '/' + item['id'])

            with (folder / 'native.log').open('wb') as log:
                child = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
                try:
                    while child.poll() is None:
                        mem = {s.split(':')[0]: int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()}
                        if mem['MemAvailable'] < 2.5 * 1024**2 or time.monotonic() - start > args.timeout:
                            raise RuntimeError('own benchmark memory/time guard reached')
                        harvest()
                        time.sleep(.25)
                    harvest()
                    if child.returncode or index != len(suite['plan']):
                        raise RuntimeError(f'{name}: native exit {child.returncode}; cases {index}/{len(suite["plan"])}')
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
            native = (folder / 'native.log').read_text(errors='replace')
            assert 'offloaded 41/41 layers to GPU' in native
            placement = re.search(r'S2MOE_PLACEMENT dense_bytes=(\d+) hot_bytes=(\d+) resident_experts=(\d+) weight_budget=(\d+)', native)
            assert placement
            suite['placement'] = dict(zip(('dense_bytes', 'hot_bytes', 'resident_experts', 'weight_budget'), map(int, placement.groups())))
            suite.update(status='passed', end_machine=machine(), finished_at=datetime.now(timezone.utc).isoformat())
            save()
        assert len(report['cases']) == total and all(c['valid'] for c in report['cases'])
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        if telemetry is not None:
            telemetry.terminate()
            telemetry.wait(timeout=10)
        report.update(finished_at=datetime.now(timezone.utc).isoformat(), final_machine=machine())
        save()


if __name__ == '__main__':
    main()
