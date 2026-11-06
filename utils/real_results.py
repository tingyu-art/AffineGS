"""Immutable real-flow artifacts shared by EXP3 figures and appendix evaluation.

This module only saves/loads results. It never trains a method or changes seeds.
"""
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import random
import pickle
import uuid

import numpy as np
import torch
from .input_validation import validate_events

REAL_METHODS = ('ILK_VG', 'OF_VG', 'INR_EST', 'INR_VG', 'ConstGS', 'AffineGS')


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def as_array(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value, dtype=np.float32)


def array_hash(value):
    a = as_array(value)
    h = hashlib.sha256(str(a.shape).encode() + a.dtype.str.encode())
    h.update(a.tobytes())
    return h.hexdigest()


def source_hashes(root):
    root = Path(root)
    paths = list((root / 'methods').rglob('*.py')) + list((root / 'generator').glob('*.py'))
    paths += [root / 'utils' / n for n in ['real_results.py', 'input_validation.py', 'visualizer.py', 'metrics.py', 'flow_classifier.py']]
    for suffix in ['*.cu', '*.cpp', '*.so']:
        paths += list((root / 'methods' / 'affinegs' / 'accelerators').rglob(suffix))
    paths = [p for p in paths if 'build' not in p.relative_to(root).parts]
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(set(paths)) if p.is_file()}


def rng_fingerprints():
    # Observation only: no new seed is set and no random draw is made.
    result = dict(python=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
                  numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
                  torch_cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
    if torch.cuda.is_available():
        result['torch_cuda'] = [hashlib.sha256(s.cpu().numpy().tobytes()).hexdigest()
                                for s in torch.cuda.get_rng_state_all()]
    return result


def _now_id():
    return datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%dT%H%M%S%f') + '-' + uuid.uuid4().hex[:8]


def _write_json(path, data):
    path = Path(path)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('x', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        tmp.replace(path)
    finally:
        if tmp.exists(): tmp.unlink()


@contextmanager
def _locked(run):
    with (Path(run) / '.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)


def read_manifest(run):
    run = Path(run).resolve()
    with (run / 'manifest.json').open(encoding='utf-8') as f:
        m = json.load(f)
    if m.get('schema_version') != 1 or m.get('run_id') != run.name:
        raise ValueError('Invalid real-data run manifest; legacy files cannot be relabelled as a verified run')
    return m


def create_real_run(root, case, input_path, source_type, events, window_ms):
    root = Path(root).resolve(); input_path = Path(input_path).resolve()
    validate_events(events, 720, 1280)
    if case not in ('ebiv', 'water'):
        raise ValueError('case must be ebiv or water')
    run = root / 'output' / 'EXP3' / 'runs' / case / _now_id()
    run.mkdir(parents=True, exist_ok=False)
    event_array = as_array(events)
    with (run / 'events.npy').open('xb') as f: np.save(f, event_array, allow_pickle=False)
    m = dict(schema_version=1, run_id=run.name, case=case, H=720, W=1280,
             input_file=str(input_path), input_file_sha256=file_hash(input_path), source_type=source_type,
             window_ms=list(map(float, window_ms)), event_count=len(event_array),
             event_sha256=array_hash(event_array), events_file_sha256=file_hash(run / 'events.npy'),
             t_min=float(event_array[:, 2].min()), t_max=float(event_array[:, 2].max()),
             source_sha256=source_hashes(root), environment=dict(python=platform.python_version(),
             numpy=np.__version__, torch=torch.__version__, cuda=torch.version.cuda), methods={}, evaluations=[])
    _write_json(run / 'manifest.json', m)
    return run


def validate_run_input(run, events, case=None, input_path=None):
    m = read_manifest(run)
    if case is not None and m['case'] != case:
        raise ValueError('The selected run belongs to a different scene')
    if array_hash(events) != m['event_sha256']:
        raise ValueError('Event content hash mismatch; event count and time range are insufficient')
    if input_path is not None and file_hash(input_path) != m['input_file_sha256']:
        raise ValueError('Input recording file hash mismatch')
    if file_hash(Path(run) / 'events.npy') != m['events_file_sha256']:
        raise ValueError('Cached event file was modified')
    return m


def ensure_method_available(run, method):
    if method not in REAL_METHODS:
        raise ValueError('Unknown method: ' + str(method))
    m = read_manifest(run)
    if method in m['methods'] or (Path(run) / (method + '.npz')).exists():
        raise FileExistsError(f'{method} already exists in {run}; start a new run to retrain it')


def save_real_flow(run, method, flow, events, root, config, rng_before, elapsed_s, details=None):
    run = Path(run); m = validate_run_input(run, events)
    if source_hashes(root) != m['source_sha256']:
        raise ValueError('Computation sources changed during this run; start a new run')
    flow = as_array(flow)
    if flow.shape != (2, m['H'], m['W']) or not np.isfinite(flow).all():
        raise ValueError('Velocity must be finite with shape [2, H, W]')
    path = run / (method + '.npz')
    with _locked(run):
        ensure_method_available(run, method)
        with path.open('xb') as f:
            np.savez_compressed(f, flow=flow, run_id=run.name, flow_name=m['case'], method_name=method,
                                event_sha256=m['event_sha256'], flow_sha256=array_hash(flow), H=m['H'], W=m['W'])
        m = read_manifest(run)
        row = dict(file=path.name, file_sha256=file_hash(path), flow_sha256=array_hash(flow),
                   event_sha256=m['event_sha256'], config=config, rng_before=rng_before,
                   elapsed_s=float(elapsed_s), details=details or {}, figures=[])
        m['methods'][method] = row
        _write_json(run / 'manifest.json', m)
    return path


def load_real_flow(run, method, events=None):
    run = Path(run); m = read_manifest(run)
    if events is not None: validate_run_input(run, events)
    row = m['methods'].get(method)
    if row is None: raise FileNotFoundError(f'Missing {method} in run {run}')
    path = run / row['file']
    if path.parent.resolve() != run.resolve() or file_hash(path) != row['file_sha256']:
        raise ValueError('Saved velocity file changed or has an invalid path')
    with np.load(path, allow_pickle=False) as z:
        flow = z['flow'].copy()
        if str(z['run_id']) != run.name or str(z['method_name']) != method or str(z['event_sha256']) != m['event_sha256']:
            raise ValueError('Velocity metadata does not match the selected run')
    if array_hash(flow) != row['flow_sha256'] or flow.shape != (2, m['H'], m['W']) or not np.isfinite(flow).all():
        raise ValueError('Velocity content hash/shape mismatch')
    return flow, row


def figure_path(run, method):
    ensure = read_manifest(run)['methods'].get(method)
    if ensure is None: raise FileNotFoundError('Save the velocity field before plotting')
    directory = Path(run) / 'figures'; directory.mkdir(exist_ok=True)
    return directory / (method + '-' + _now_id() + '.pdf')


def register_real_figure(run, method, path, plotted_flow):
    run = Path(run).resolve(); path = Path(path).resolve()
    flow, row = load_real_flow(run, method)
    if array_hash(plotted_flow) != row['flow_sha256']:
        raise ValueError('The plotted flow differs from the saved velocity')
    if path.parent != run / 'figures' or not path.is_file():
        raise ValueError('Figure must be saved inside this run figures directory')
    with _locked(run):
        m = read_manifest(run)
        m['methods'][method]['figures'].append(dict(file=str(path.relative_to(run)),
            file_sha256=file_hash(path), flow_sha256=row['flow_sha256']))
        _write_json(run / 'manifest.json', m)


def select_real_run(root, case, run_dir=None):
    if run_dir is not None: run = Path(run_dir).resolve()
    else:
        candidates = sorted((Path(root) / 'output' / 'EXP3' / 'runs' / case).glob('*/manifest.json'))
        if not candidates: raise FileNotFoundError('No verified run. Generate the six fields and figures in EXP3 first.')
        run = candidates[-1].parent
    if read_manifest(run)['case'] != case: raise ValueError('Selected run belongs to a different scene')
    return run


def load_real_case(run, events, input_path, case):
    m = validate_run_input(run, events, case, input_path)
    flows = {}; provenance = {}
    for method in REAL_METHODS:
        flow, row = load_real_flow(run, method, events)
        if not row['figures']:
            raise ValueError(f'{method} has no registered EXP3 figure; save its figure before appendix evaluation')
        for figure in row['figures']:
            if figure['flow_sha256'] != row['flow_sha256'] or file_hash(Path(run) / figure['file']) != figure['file_sha256']:
                raise ValueError('EXP3 figure hash mismatch')
        flows[method] = flow
        provenance[method] = dict(flow_sha256=row['flow_sha256'], file_sha256=row['file_sha256'],
                                  figures=row['figures'])
    return flows, dict(run_id=m['run_id'], case=m['case'], event_sha256=m['event_sha256'],
                       input_file_sha256=m['input_file_sha256'], source_sha256=m['source_sha256'], methods=provenance)


def save_real_metrics(run, metric, summary, provenance):
    m = read_manifest(run)
    if provenance['run_id'] != m['run_id'] or provenance['event_sha256'] != m['event_sha256']:
        raise ValueError('Metric provenance belongs to a different run')
    for method in REAL_METHODS:
        _, row = load_real_flow(run, method)
        if provenance['methods'][method]['flow_sha256'] != row['flow_sha256']:
            raise ValueError('Metric flow differs from the EXP3 flow')
        if provenance['methods'][method]['figures'] != row['figures']:
            raise ValueError('Metric figure references differ from the EXP3 figure references')
        for figure in row['figures']:
            if file_hash(Path(run) / figure['file']) != figure['file_sha256']:
                raise ValueError('EXP3 figure was modified before metric saving')
    folder = Path(run) / 'metrics'; folder.mkdir(exist_ok=True)
    path = folder / (metric + '-' + _now_id() + '.json')
    payload = dict(summary, provenance=provenance)
    with path.open('x', encoding='utf-8') as f: json.dump(payload, f, indent=2, allow_nan=False)
    with _locked(run):
        m = read_manifest(run); m['evaluations'].append(dict(metric=metric, file=str(path.relative_to(run)), file_sha256=file_hash(path)))
        _write_json(Path(run) / 'manifest.json', m)
    return path
