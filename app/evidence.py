"""Read-only dashboard access to pipeline data, provenance and fail-closed analysis."""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, load_experiments
from src.data.loaders import load_niteroi
from src.data.validate import validate_panel
from src.pipeline.predict import analyze

ABOUT = ('This is a research decision-support prototype. It is not a clinical '
         'diagnostic system. Intervention-effect estimates are conditional on '
         'the data and causal assumptions and should receive domain review.')
PAGES = ['Overview', 'Data Audit', 'Counterfactual Analysis', 'Model Comparison',
         'Reliability', 'Robustness', 'Export']
STRATEGIES = {'cits': 'CITS', 'synthetic_control': 'SCM', 'equal_weight': 'Equal-weight ensemble',
              'best_historical': 'Fixed historically-best model', 'dynamic': 'Reliability-weighted ensemble'}


class EvidenceUnavailable(ValueError):
    """Missing or inconsistent evidence must never become an example result."""


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError('Expected a JSON object')
        return value
    except (OSError, ValueError) as exc:
        raise EvidenceUnavailable(f'{path.name}: {exc}') from exc


def read_table(path: Path, required=(), numeric=()) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
        if frame.empty or not set(required) <= set(frame):
            raise ValueError('No complete result table is available')
        for name in numeric:
            if not np.isfinite(pd.to_numeric(frame[name], errors='coerce')).all():
                raise ValueError(f'Invalid numeric values in {name}')
        return frame
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        raise EvidenceUnavailable(f'{path.name}: {exc}') from exc


def run_directory(root: Path, manifest: dict) -> Path:
    try:
        directory = (root/manifest['directory']).resolve()
        if not directory.is_relative_to((root/'artifacts/runs').resolve()):
            raise ValueError('Run path must be within artifacts/runs')
        recorded = read_json(directory/'manifest.json')
        if directory.name != manifest['run_id'] or recorded != manifest:
            raise ValueError('Run manifest and latest pointer disagree')
        return directory
    except (KeyError, ValueError) as exc:
        raise EvidenceUnavailable(f'Invalid run provenance: {exc}') from exc


@dataclass
class Snapshot:
    config: object = None
    manifest: dict = field(default_factory=dict)
    directory: Path | None = None
    panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    audit: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    retrieval: dict = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    data_issues: list[str] = field(default_factory=list)

    @property
    def ready(self):
        return bool(self.directory and self.audit.get('valid') and not self.issues and not self.data_issues)


def load_snapshot(root: Path) -> Snapshot:
    snapshot = Snapshot()
    try:
        snapshot.config = load_config(root/'configs/niteroi.yaml')
        snapshot.panel = load_niteroi(root, snapshot.config.data)
        snapshot.audit = validate_panel(snapshot.panel, snapshot.config.models.min_pre_months,
                                       snapshot.config.models.min_post_months).to_dict()
        snapshot.data_issues.extend(snapshot.audit['errors'])
        snapshot.metadata = read_json(root/snapshot.config.data.raw_dir/'source_metadata.json')
        snapshot.retrieval = read_json(root/snapshot.config.data.raw_dir/'retrieval.json')
    except Exception as exc:
        snapshot.data_issues.append(str(exc))
    try:
        snapshot.manifest = read_json(root/'artifacts/latest_reliability.json')
        snapshot.directory = run_directory(root, snapshot.manifest)
        if snapshot.config is None:
            raise EvidenceUnavailable('Project configuration is unavailable')
        raw = root/snapshot.config.data.raw_dir/snapshot.config.data.filename
        checksum = hashlib.sha256(raw.read_bytes()).hexdigest()
        if checksum != snapshot.manifest['dataset_sha256']:
            raise EvidenceUnavailable('Dataset differs from the trained run; regenerate pipeline artifacts')
        if snapshot.manifest['config'] != snapshot.config.model_dump(mode='json'):
            raise EvidenceUnavailable('Project configuration differs from the trained run')
        if snapshot.manifest['experiment_config'] != load_experiments(root/'configs/experiments.yaml').model_dump():
            raise EvidenceUnavailable('Experiment configuration differs from the trained run')
        if not (snapshot.directory/'bundle.joblib').is_file():
            raise EvidenceUnavailable('Trained analysis bundle is missing')
    except Exception as exc:
        snapshot.issues.append(str(exc))
    return snapshot


def strategy_results(snapshot: Snapshot) -> pd.DataFrame:
    if not snapshot.ready:
        raise EvidenceUnavailable('; '.join(snapshot.issues + snapshot.data_issues) or 'No validated research run')
    frame = read_table(snapshot.directory/'strategy_results.csv',
        ['experiment_id', 'strategy', 'split', 'treated_location', 'horizon', 'mae', 'rmse', 'normalized_mae'],
        ['mae', 'rmse', 'normalized_mae'])
    if frame.duplicated(['experiment_id','strategy']).any():
        raise EvidenceUnavailable('Duplicate experiment/strategy records')
    if not frame.groupby('experiment_id').strategy.apply(lambda x:set(x)==set(STRATEGIES)).all():
        raise EvidenceUnavailable('Missing comparison strategies in an experiment')
    return frame


def selection_key(snapshot: Snapshot, area: str, horizon: int) -> tuple:
    return (snapshot.manifest.get('run_id'), snapshot.manifest.get('dataset_sha256'), area, horizon)


def abstention(study: str, area: str, reason: str) -> dict:
    return {'study_id':study, 'location_id':area, 'decision':'INSUFFICIENT_EVIDENCE',
            'effect_absolute':None, 'effect_percent':None, 'interval_lower':None, 'interval_upper':None,
            'reliability_score':None, 'model_weights':{}, 'diagnostics':{}, 'trajectories':[],
            'exploratory_contrast':None, 'warnings':[reason], 'assumptions':[],
            'explanation':{'reasons':[reason]}}


def run_analysis(root: Path, snapshot: Snapshot, area: str, horizon: int) -> dict:
    study = snapshot.config.data.study_id if snapshot.config else 'unavailable'
    if not snapshot.ready:
        return abstention(study, area, '; '.join(snapshot.issues + snapshot.data_issues))
    try:
        result = analyze(root, study, area, horizon_months=horizon)
        # Strict JSON serialization also rejects NaN/Infinity instead of exposing bogus scores.
        json.dumps(result, allow_nan=False)
        return result
    except Exception as exc:
        # UI boundary: an unreadable/unloadable model must leave a visible abstention, not a traceback.
        return abstention(study, area, f'Analysis unavailable ({type(exc).__name__}): {exc}')


def archived_robustness(root: Path, snapshot: Snapshot) -> list[dict]:
    found = []
    for path in sorted((root/'artifacts/runs').glob('*/robustness.json'), reverse=True):
        try:
            manifest = read_json(path.parent/'manifest.json')
            robustness = read_json(path)
            if (manifest['dataset_sha256'] == snapshot.manifest.get('dataset_sha256') and
                    robustness['reliability_run'] == manifest['run_id'] and
                    manifest['run_id'] != snapshot.manifest.get('run_id')):
                run_directory(root, manifest)
                found.append(manifest)
        except (EvidenceUnavailable, KeyError):
            continue
    return found


def export_payload(result: dict, snapshot: Snapshot) -> str:
    payload = dict(result)
    payload['dashboard_context'] = {'run_id':snapshot.manifest.get('run_id'),
        'dataset_sha256':snapshot.manifest.get('dataset_sha256'), 'limitations':ABOUT}
    return json.dumps(payload, indent=2, allow_nan=False)


def result_csv(result: dict) -> str:
    """Include a status row when no trajectory exists, keeping abstentions exportable."""
    common = {key:result.get(key) for key in ['study_id','location_id','decision','reliability_score',
                                            'effect_absolute','effect_percent']}
    common['warnings'] = ' | '.join(result.get('warnings', []))
    return pd.DataFrame([common | row for row in result.get('trajectories', [])] or [common]).to_csv(index=False)


def result_report(result: dict, snapshot: Snapshot) -> str:
    return (f"# RACE-Dengue analysis\n\nStudy: {result['study_id']}\n\nArea: {result['location_id']}\n\n"
            f"Evidence status: {result['decision']}\n\n{ABOUT}\n\n"
            + '\n'.join('- '+w for w in result.get('warnings', []))
            + '\n\n## Full pipeline result and provenance\n\n```json\n'
            + export_payload(result, snapshot) + '\n```\n')
