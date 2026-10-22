"""UI tests compare rendered results with actual artifacts and exercise abstentions."""
import copy
import io
import json
from pathlib import Path
import pandas as pd
import pytest
from app import evidence

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def snapshot():
    result=evidence.load_snapshot(ROOT)
    if not result.ready: pytest.skip('Requires verified source and trained artifacts')
    return result


def start_app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=30).run()


def assert_rendered(app):
    assert not app.exception
    assert not any('This view cannot be validated' in w.value for w in app.warning)
    assert any(evidence.ABOUT in m.value for m in app.markdown)


@pytest.mark.real_data
def test_dashboard_pages_real_results_and_analysis(snapshot):
    app=start_app();assert_rendered(app)
    metrics={m.label:m.value for m in app.metric}
    assert metrics['Source observations']==str(len(snapshot.panel))
    assert metrics['Valid pseudo experiments']==str(evidence.strategy_results(snapshot).experiment_id.nunique())
    for page in evidence.PAGES:
        app.sidebar.radio[0].set_value(page).run();assert_rendered(app)
    app.sidebar.radio[0].set_value('Model Comparison').run()
    expected=evidence.strategy_results(snapshot).query('split == "test"').groupby('strategy').normalized_mae.mean()
    table=app.dataframe[0].value
    for code,label in evidence.STRATEGIES.items():
        assert table.loc[table.Method.eq(label),'normalized_mae'].iloc[0]==pytest.approx(expected[code])
    app.sidebar.radio[0].set_value('Counterfactual Analysis').run()
    app.selectbox(key='analysis_area').set_value(snapshot.config.treated_units[1]).run()
    app.button[0].click().run();assert_rendered(app)
    result=app.session_state['analysis']
    assert result['decision']=='INSUFFICIENT_EVIDENCE'
    assert result['effect_absolute'] is None
    assert len(result['trajectories'])==app.session_state['selected_horizon']
    assert any('Insufficient Evidence' in m.value for m in app.markdown)
    for page in ['Reliability','Export','Counterfactual Analysis']:
        app.sidebar.radio[0].set_value(page).run();assert_rendered(app)
        assert app.session_state['analysis']['location_id']==result['location_id']
    # A changed selection must not export the previous area's score.
    app.selectbox(key='analysis_area').set_value(snapshot.config.treated_units[0]).run()
    assert 'analysis' not in app.session_state
    app.sidebar.radio[0].set_value('Export').run();assert_rendered(app)
    assert any('Run Counterfactual Analysis first' in x.value for x in app.info)


@pytest.mark.real_data
def test_archived_robustness_is_explicit(snapshot):
    app=start_app();app.sidebar.radio[0].set_value('Robustness').run();assert_rendered(app)
    if not (snapshot.directory/'robustness.json').exists():
        assert any('Not run in the selected artifact run' in w.value for w in app.warning)
    archives=evidence.archived_robustness(ROOT,snapshot)
    if archives:
        app.selectbox[0].select_index(1).run();assert_rendered(app)
        assert any('do not validate the current run' in w.value for w in app.warning)
        expected=pd.read_csv(ROOT/archives[0]['directory']/'placebos.csv')
        pd.testing.assert_frame_equal(app.dataframe[0].value,expected)


@pytest.mark.real_data
def test_missing_run_shows_insufficient_evidence_on_every_page(snapshot,monkeypatch):
    blocked=copy.deepcopy(snapshot);blocked.issues=['UNIT TEST: required model artifact is missing']
    monkeypatch.setattr(evidence,'load_snapshot',lambda root:blocked)
    app=start_app()
    for page in evidence.PAGES:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
        assert any('Insufficient Evidence' in m.value for m in app.markdown)
        assert 'analysis' not in app.session_state


@pytest.mark.real_data
def test_missing_comparison_does_not_crash(snapshot,monkeypatch):
    def missing(_): raise evidence.EvidenceUnavailable('UNIT TEST: missing comparison table')
    monkeypatch.setattr(evidence,'strategy_results',missing)
    app=start_app()
    assert not app.exception
    assert any('Insufficient Evidence' in m.value for m in app.markdown)
    assert any('missing comparison table' in w.value for w in app.warning)


@pytest.mark.real_data
def test_analysis_failure_and_export_preserve_nulls(snapshot,monkeypatch):
    def unavailable(*args,**kwargs): raise FileNotFoundError('UNIT TEST: unreadable bundle')
    monkeypatch.setattr(evidence,'analyze',unavailable)
    result=evidence.run_analysis(ROOT,snapshot,snapshot.config.treated_units[0],6)
    assert result['decision']=='INSUFFICIENT_EVIDENCE'
    assert result['reliability_score'] is None and result['effect_absolute'] is None
    payload=json.loads(evidence.export_payload(result,snapshot))
    assert payload['effect_absolute'] is None
    assert payload['dashboard_context']['run_id']==snapshot.manifest['run_id']
    table=pd.read_csv(io.StringIO(evidence.result_csv(result)))
    assert len(table)==1 and table.effect_absolute.isna().all()
    assert table.decision.iloc[0]=='INSUFFICIENT_EVIDENCE'
    assert evidence.ABOUT in evidence.result_report(result,snapshot)


def test_malformed_metadata_and_tables_fail_closed(tmp_path):
    (tmp_path/'bad.json').write_text('{broken')
    with pytest.raises(evidence.EvidenceUnavailable): evidence.read_json(tmp_path/'bad.json')
    (tmp_path/'bad.csv').write_text('mae\ninvalid\n')
    with pytest.raises(evidence.EvidenceUnavailable): evidence.read_table(tmp_path/'bad.csv',['mae'],['mae'])
    with pytest.raises(evidence.EvidenceUnavailable):
        evidence.run_directory(tmp_path,{'directory':'../outside','run_id':'outside'})
    snapshot=evidence.load_snapshot(tmp_path)
    assert not snapshot.ready and snapshot.data_issues and snapshot.issues


@pytest.mark.real_data
def test_changed_source_configuration_blocks_analysis(snapshot,monkeypatch):
    changed=snapshot.config.model_copy(deep=True);changed.models.nb_alpha+=1
    monkeypatch.setattr(evidence,'load_config',lambda _:changed)
    actual=evidence.load_snapshot(ROOT)
    assert not actual.ready
    assert any('configuration differs' in issue for issue in actual.issues)
    result=evidence.run_analysis(ROOT,actual,changed.treated_units[0],6)
    assert result['decision']=='INSUFFICIENT_EVIDENCE' and not result['trajectories']
