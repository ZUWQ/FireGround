'''Canonical, offline baseline evidence and pre-run reuse checks.

No simulator or model clients are imported. Artifacts are registered explicitly;
references keep the original run_id and never count as a new execution.
'''
from __future__ import annotations
import copy
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / 'outputs/none_experiments'
ENV_ROOT = 'EMBODIED_BASELINE_LIBRARY'
REQUIRED = ('status.json', 'task.json', 'scene_config.json', 'evaluation/summary.json',
            'evaluation/metadata.json', 'evaluation/world_states.jsonl',
            'evaluation/models.jsonl', 'evaluation/actions.jsonl')


def root_path(root=None):
    return Path(root or os.environ.get(ENV_ROOT) or DEFAULT_ROOT).expanduser().absolute()


def read(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def normalize(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError('Non-finite baseline condition')
        return float(value)
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in sorted(value.items())}
    return value


def condition(task):
    scene = task.get('scene_config')
    if not isinstance(scene, dict) or not scene:
        raise ValueError('Baseline lookup requires a complete scene_config')
    duration, burn = task.get('time_limit_sec'), task.get('burn_time', scene.get('FireRateDelayTime'))
    if (isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0
            or isinstance(burn, bool) or not isinstance(burn, (int, float)) or burn < 0):
        raise ValueError('Baseline lookup requires valid duration and preburn')
    # Extinguishing speed has no effect when there are zero agents/actions.
    dynamics = {k:v for k,v in scene.items() if k != 'OutfireMultiply'}
    return normalize(dict(scene_config=dynamics, time_limit_sec=duration, burn_time_sec=burn))


def fingerprint(value):
    payload = json.dumps(normalize(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def catalog(root=None):
    path = root_path(root) / 'catalog.json'
    if not path.exists():
        return dict(schema_version=1, entries={}, retired={}, aliases={})
    data = read(path)
    if data.get('schema_version') != 1 or not all(isinstance(data.get(k), dict) for k in ('entries', 'retired', 'aliases')):
        raise ValueError('Invalid baseline library catalog: ' + str(path))
    return data


def entry_path(entry, root=None):
    base = root_path(root).resolve()
    path = (base / entry['artifact_dir']).resolve()
    if not path.is_relative_to(base / 'runs'):
        raise ValueError('Baseline artifact escapes the library')
    return path


def validate_entry(entry, root=None, *, full=False):
    path = entry_path(entry, root)
    for name in REQUIRED:
        record = entry['files'].get(name)
        target = path / name
        if not record or not target.is_file() or target.stat().st_size != record['size']:
            raise ValueError('Baseline evidence missing or changed: ' + str(target))
        if full or name != 'evaluation/world_states.jsonl' or target.stat().st_mtime_ns != record['mtime_ns']:
            if file_sha(target) != record['sha256']:
                raise ValueError('Baseline evidence hash changed: ' + str(target))
    summary = read(path / 'evaluation/summary.json')
    if summary.get('run_id') != entry['run_id']:
        raise ValueError('Baseline run_id changed')
    return path


def find(task, *, policy='exact', root=None):
    '''Return a single admitted existing baseline; never select by its score.'''
    if (root_path(root) / '.initializing').exists():
        raise ValueError('Baseline library is being initialized; finish offline registration before dispatch')
    requested = condition(task)
    if policy not in ('exact', 'core6_shared_r'):
        raise ValueError('Unknown baseline match policy')
    matches = []
    for entry in catalog(root)['entries'].values():
        recorded = entry.get('dynamics_condition') or normalize(dict(
            scene_config={k:v for k,v in entry['condition']['scene_config'].items() if k != 'OutfireMultiply'},
            time_limit_sec=entry['condition']['time_limit_sec'], burn_time_sec=entry['condition']['burn_time_sec']))
        if any(requested[k] != recorded[k] for k in ('time_limit_sec', 'burn_time_sec')):
            continue
        a, b = requested['scene_config'], recorded['scene_config']
        matches_scene = fingerprint(a) == fingerprint(b)
        if not matches_scene and policy == 'core6_shared_r':
            from .benchmark_metrics import baseline_scenes_match
            raw_a, raw_b = task['scene_config'], entry['scene_config']
            an, bn = raw_a.get('NPCRoom'), raw_b.get('NPCRoom')
            npc_gate = (isinstance(an, list) and isinstance(bn, list) and len(an) == len(bn) == 14
                and all(type(v) is int and v == 0 for v in an)
                and all(type(v) is int and v in (0, 1) for v in bn) and sum(bn) == 2)
            other_fields_match = fingerprint({k:v for k,v in a.items() if k != 'NPCRoom'}) == fingerprint({k:v for k,v in b.items() if k != 'NPCRoom'})
            matches_scene = npc_gate and other_fields_match and baseline_scenes_match(raw_a, raw_b, policy=policy, ignore_outfire=True)
        if not matches_scene:
            continue
        if task.get('development_probe') and not entry.get('development_probe'):
            continue
        if bool(task.get('fire_mapping')) != bool(entry.get('fire_mapping')):
            continue
        matches.append(entry)
    if len(matches) > 1:
        raise ValueError('Multiple baseline executions match; select one explicitly in the library catalog')
    if not matches:
        return None
    entry = copy.deepcopy(matches[0])
    entry['resolved_artifact_dir'] = str(validate_entry(entry, root))
    return entry


def resolve(path, *, root=None):
    '''Route known baseline paths/run_ids to the canonical evidence only.'''
    requested = Path(path).expanduser().absolute()
    data = catalog(root)
    key = data['aliases'].get(str(requested))
    if key is None and (requested / 'evaluation/summary.json').is_file():
        key = read(requested / 'evaluation/summary.json').get('run_id')
    def selected_key(value):
        seen = set()
        while value in data.get('superseded', {}):
            if value in seen:
                raise ValueError('Cycle in explicitly selected baseline replacements')
            seen.add(value)
            value = data['superseded'][value]['replacement_run_id']
        if seen and value not in data['entries']:
            raise ValueError('Selected replacement baseline is missing')
        return value

    key = selected_key(key)
    if key in data['retired']:
        retired = data['retired'][key]
        if retired.get('no_agent_equivalent'):
            replacement = data['entries'].get(selected_key(retired['replacement_run_id']))
            if replacement is None:
                raise ValueError('Equivalent replacement baseline is missing')
            return validate_entry(replacement, root)
        raise ValueError('Baseline deleted by user; use the selected library baseline: ' + str(selected_key(retired['replacement_run_id'])))
    if key in data['entries']:
        return validate_entry(data['entries'][key], root)
    return requested.resolve()


def reuse(task, directory, *, root=None, repeat_index=1, attempt_index=1):
    '''Materialize a reference, without launching UE, ignition, or any model.'''
    if task.get('runner') != 'BASELINE':
        return None
    if task.get('agent_types') != [] or task.get('agent_positions') != [] or task.get('llm_profile') is not None:
        raise ValueError('A reused baseline must remain agent-free and model-free')
    # Explicit repeat experiments require a user-reviewed reason in their task.
    if 'baseline_repeat_reason' in task:
        reason = task['baseline_repeat_reason']
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('baseline_repeat_reason must explain an explicitly authorized repeat')
        return None
    entry = find(task, root=root)
    if entry is None:
        return None
    source = Path(entry['resolved_artifact_dir'])
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    if output.resolve() == source.resolve():
        raise ValueError('Cannot write a reuse reference over canonical evidence')
    existing = read(output / 'status.json') if (output / 'status.json').exists() else {}
    if existing.get('run_id') == entry['run_id'] and existing.get('reused_baseline'):
        return existing
    for child in source.iterdir():
        if child.name in ('status.json', 'task.json', 'scene_config.json', 'baseline_reference.json', 'artifact_manifest.json'):
            continue
        target = output / child.name
        if target.exists() or target.is_symlink():
            raise ValueError('Refusing to overwrite baseline reference output: ' + str(target))
        target.symlink_to(child, target_is_directory=child.is_dir())
    write(output / 'task.json', task)
    write(output / 'scene_config.json', task['scene_config'])
    write(output / 'baseline_reference.json', dict(run_id=entry['run_id'], artifact_dir=str(source),
          condition_sha256=entry['condition_sha256'], executed=False))
    now = datetime.now().astimezone().isoformat()
    record = dict(task_id=task['task_id'], status='completed', stage='baseline_reused',
        run_id=entry['run_id'], termination_reason='time_limit', returncode=0,
        repeat_index=repeat_index, attempt_index=attempt_index, started_at=now, finished_at=now,
        output_dir=str(output), reused_baseline=True, executed=False,
        baseline_artifact_dir=str(source), source_completed_at=entry.get('finished_at'))
    write(output / 'status.json', record)
    return record


def assess(tasks, *, policy='exact', root=None):
    rows = []
    for task in tasks:
        entry = find(task, policy=policy, root=root)
        rows.append(dict(task_id=task.get('task_id'), runner=task.get('runner'),
            decision='reuse_existing' if entry else 'missing_baseline',
            run_id=None if entry is None else entry['run_id'],
            artifact_dir=None if entry is None else entry['resolved_artifact_dir']))
    return dict(library=str(root_path(root)), tasks=rows,
                reusable=sum(r['decision'] == 'reuse_existing' for r in rows),
                missing=sum(r['decision'] == 'missing_baseline' for r in rows), dispatched=False)


def register(source, *, root=None, aliases=(), expected_run_id=None, trusted_gate=False):
    '''Copy one real completed baseline into the independent library and hash it.'''
    source = Path(source).resolve()
    task, status = read(source / 'task.json'), read(source / 'status.json')
    summary, metadata = read(source / 'evaluation/summary.json'), read(source / 'evaluation/metadata.json')
    run_id = summary.get('run_id')
    if not isinstance(run_id, str) or not run_id or '/' in run_id or run_id in ('.', '..'):
        raise ValueError('Invalid baseline run_id')
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError('Baseline differs from the audited run_id')
    if (status.get('status') != 'completed' or task.get('runner') != 'BASELINE'
            or task.get('agent_types') != [] or task.get('agent_positions') != []
            or task.get('llm_profile') is not None or summary.get('structure') != 'BASELINE'
            or metadata.get('structure') != 'BASELINE' or metadata.get('agents') != {}
            or summary.get('model_call_count') != 0 or summary.get('physical_action_count') != 0
            or summary.get('termination_reason') != 'time_limit'
            or status.get('cleanup_error') or status.get('restore_error')):
        raise ValueError('Not a completed, agent-free, model-free baseline')
    cfg = condition(task)
    recorded_scene = metadata.get('config', {}).get('scene_config', {})
    if (normalize({k:v for k,v in recorded_scene.items() if k != 'OutfireMultiply'}) != cfg['scene_config']
            or not math.isclose(summary['deadline_monotonic'] - summary['started_at_monotonic'], cfg['time_limit_sec'], abs_tol=1e-6)):
        raise ValueError('Baseline configuration/clock mismatch')
    if not trusted_gate:
        from ..experiments.development_validation import validate_baseline
        gate = validate_baseline(source, task)
        if not gate.get('passed'):
            raise ValueError('Baseline admission failed: ' + str(gate.get('errors')))
    base = root_path(root)
    base.mkdir(parents=True, exist_ok=True)
    with (base / '.catalog.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = catalog(base)
        if run_id in data['retired'] or run_id in data.get('superseded', {}):
            raise ValueError('Retired or superseded baseline cannot be registered again')
        same = [e for e in data['entries'].values() if e.get('dynamics_sha256', fingerprint(dict(
            scene_config={k:v for k,v in e['condition']['scene_config'].items() if k != 'OutfireMultiply'},
            time_limit_sec=e['condition']['time_limit_sec'], burn_time_sec=e['condition']['burn_time_sec']))) == fingerprint(cfg)]
        if same and same[0]['run_id'] != run_id:
            raise ValueError('Same condition already has a canonical execution; reuse it or review the selection explicitly')
        target = base / 'runs' / run_id
        if run_id not in data['entries']:
            staging = base / 'runs' / ('.' + run_id + '.' + uuid4().hex + '.tmp')
            staging.mkdir(parents=True)
            try:
                for name in REQUIRED:
                    if not (source / name).is_file():
                        raise ValueError('Missing baseline evidence: ' + name)
                # Keep evidence and diagnostics; engine cache is not experiment data.
                for child in source.iterdir():
                    if child.name in ('saved', 'userdata', 'ue', 'code', '.git'):
                        continue
                    if child.is_dir():shutil.copytree(child, staging / child.name)
                    else:shutil.copy2(child, staging / child.name)
                files = {}
                for p in sorted(staging.rglob('*')):
                    if not p.is_file():continue
                    rel = str(p.relative_to(staging)); src = source / rel
                    value = file_sha(p)
                    if file_sha(src) != value:
                        raise ValueError('Baseline copy verification failed: ' + rel)
                    files[rel] = dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns, sha256=value)
                entry = dict(run_id=run_id, task_id=task['task_id'], artifact_dir='runs/' + run_id,
                    condition=normalize(dict(scene_config=task['scene_config'], time_limit_sec=task['time_limit_sec'], burn_time_sec=task.get('burn_time', task['scene_config'].get('FireRateDelayTime')))),
                    condition_sha256=fingerprint(dict(scene_config=task['scene_config'], time_limit_sec=task['time_limit_sec'], burn_time_sec=task.get('burn_time', task['scene_config'].get('FireRateDelayTime')))),
                    dynamics_condition=cfg, dynamics_sha256=fingerprint(cfg), scene_config=task['scene_config'],
                    development_probe=bool(task.get('development_probe')), fire_mapping=bool(task.get('fire_mapping')),
                    finished_at=status.get('finished_at'), source=str(source), files=files)
                write(staging / 'artifact_manifest.json', entry)
                staging.rename(target)
                data['entries'][run_id] = entry
            finally:
                if staging.exists():shutil.rmtree(staging)
        else:
            validate_entry(data['entries'][run_id], base)
        for alias in (str(source), *aliases):
            data['aliases'][str(Path(alias).absolute())] = run_id
        data['updated_at'] = datetime.now().astimezone().isoformat()
        write(base / 'catalog.json', data)
    return copy.deepcopy(data['entries'][run_id])
