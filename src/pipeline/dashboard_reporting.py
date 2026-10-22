"""Append only recorded, current dashboard verification to the PP1 report."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

MARKER='## PP1 demonstration interface'


def update(root: Path) -> dict:
    summary=root/'reports/pp1_summary.md'
    verification_path=root/'artifacts/figures/pp1_dashboard/verification.json'
    if not verification_path.exists():
        return {'status':'No browser verification recorded'}
    verification=json.loads(verification_path.read_text())
    manifest=json.loads((root/'artifacts/latest_reliability.json').read_text())
    current=(verification.get('reliability_run')==manifest['run_id'] and
        all((root/path).is_file() and hashlib.sha256((root/path).read_bytes()).hexdigest()==digest
            for path,digest in verification.get('source_sha256',{}).items()) and
        bool(verification.get('source_sha256')))
    screenshots=[path for path in verification.get('screenshots',[]) if (root/path).is_file()]
    lines=[MARKER,'',f"Browser verification recorded: {verification.get('timestamp_utc','unavailable')}.",
        f"Verified research run: `{verification.get('reliability_run','unavailable')}`.",
        f"Recorded browser outcome: **{verification.get('status','unavailable')}**. "
        f"Source files and active research run still match verification: **{current}**.",'']
    if verification.get('status')=='passed' and current:
        lines += ['Implemented and exercised in the running Streamlit application:', '',
            '- Overview: research problem, question, component boundary and artifact-derived comparisons.',
            '- Data Audit: original metadata, observations, locations, period, missingness and intervention chronology; pipeline validation is recomputed.',
            '- Counterfactual Analysis: source study/area, configured horizons, locked source dates and real observed/CITS/SCM/ensemble trajectories.',
            '- Model Comparison: paired experiment metrics, validation diagnostics and learned weights from the current run.',
            '- Reliability: actual score, prominent evidence status, measured factors, warnings and the frozen error-versus-coverage operating point.',
            '- Robustness: current-run availability and explicitly selected archived placebo, donor and stress artifacts. Older results are never presented as current-run validation.',
            '- Export: actual analysis JSON, status-bearing CSV and Markdown report; unsupported effects remain null.',
            '- The requested About/Limitations text is displayed on every page.', '',
            f"Browser pages checked: {', '.join(verification['pages_checked'])}.",
            f"Download contents compared with fresh pipeline analysis: **{verification.get('export_matches_pipeline',False)}**.",
            f"Uncaught browser errors: **{len(verification.get('browser_errors',[]))}**.",'',
            'The real recorded-intervention demonstration returned `INSUFFICIENT_EVIDENCE`; '
            'it displays warnings and exploratory trajectories while withholding a supported effect. '
            'No research models were retrained or methods added for the UI.']
    else:
        lines += ['Browser verification is incomplete or stale. Do not treat old screenshots as verification of the current interface.',
                  verification.get('failure','')]
    tests=root/'artifacts/metrics/dashboard_tests.xml'
    if tests.exists():
        suites=list(ET.parse(tests).getroot().iter('testsuite'))
        counts={key:sum(int(s.get(key,'0')) for s in suites) for key in ['tests','failures','errors','skipped']}
        counts['passed']=counts['tests']-counts['failures']-counts['errors']-counts['skipped']
        lines += ['', 'Recorded dashboard tests: `'+json.dumps(counts)+'`.',
            'These include artifact-to-UI equality, source/configuration mismatch, missing artifacts, '
            'stale selection clearing, archived-run labeling and null-preserving abstention exports.']
    lines += ['',f'Saved browser screenshots: **{len(screenshots)}**.',
              *['- `'+path+'`' for path in screenshots], '',
              'Verification record: `artifacts/figures/pp1_dashboard/verification.json`.',
              'Launch locally: `.venv/bin/python -m streamlit run app/streamlit_app.py`.',
              'Recheck the running UI: `.venv/bin/python scripts/capture_pp1_dashboard.py`.',
              'The browser tooling is optional; Streamlit does not require Playwright to run.']
    original=summary.read_text().split(MARKER)[0].rstrip()
    summary.write_text(original+'\n\n'+'\n'.join(lines)+'\n')
    return {'status':verification.get('status'),'current':current,'screenshots':len(screenshots)}


if __name__=='__main__':
    print(json.dumps(update(Path(__file__).resolve().parents[2]),indent=2))
