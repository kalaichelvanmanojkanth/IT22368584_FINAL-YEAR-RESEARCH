"""PP1 demonstration: every research result comes from the pipeline."""
from pathlib import Path
import json
import logging
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from app.evidence import (ABOUT, PAGES, STRATEGIES, EvidenceUnavailable, load_snapshot,
    read_json, read_table, strategy_results, selection_key, run_analysis, archived_robustness,
    export_payload, result_csv, result_report)
from src.pipeline.reporting import test_summary

st.set_page_config(page_title='RACE-Dengue · PP1 evidence workspace',page_icon='🔬',layout='wide')
st.markdown('''<style>
.block-container{max-width:1440px;padding-top:4.5rem;padding-bottom:3rem}
h1,h2,h3{letter-spacing:-.035em}h1{font-size:2.65rem!important;font-weight:750!important}
[data-testid="stSidebar"]{border-right:1px solid #d6e2e9}
[data-testid="stMetric"]{background:white;border:1px solid #dce6ec;border-radius:12px;padding:16px}
[data-testid="stMetricValue"]{font-size:1.8rem}
.eyebrow{font-size:.73rem;letter-spacing:.17em;font-weight:700;color:#087e83;margin-bottom:8px}
.lede{font-size:1.1rem;max-width:900px;color:#4e6476;line-height:1.65}
.evidence-banner{border-left:5px solid #c64e3c;background:#fff1eb;padding:18px 22px;border-radius:8px;margin:16px 0;color:#752d21}
.evidence-banner strong{font-size:1.4rem}
.about-box{background:#eaf0f5;border:1px solid #d7e1e8;border-radius:12px;padding:18px 22px;color:#40586b;font-size:.9rem;line-height:1.6;margin-top:28px}
</style>''',unsafe_allow_html=True)
snapshot=load_snapshot(ROOT)
with st.sidebar:
    st.markdown('### RACE-Dengue')
    st.caption('PP1 · Research evidence workspace')
    page=st.radio('Workspace',PAGES,label_visibility='collapsed',key='workspace')
    st.divider()
    st.caption('PIPELINE PROVENANCE')
    st.caption('Run: '+snapshot.manifest.get('run_id','Unavailable'))
    st.caption('Data: '+(snapshot.config.data.study_id if snapshot.config else 'Unavailable'))
    try:
        checks=test_summary(ROOT)
        if checks.get('source'):
            st.caption(f"Saved tests: {checks['passed']} passed · {checks['failures']} failed · {checks['errors']} errors · {checks['skipped']} skipped")
        else: st.caption('Saved test results unavailable')
    except Exception: st.caption('Saved test results unreadable')
    st.caption('Saved verification is not a guarantee of causal validity.')
st.markdown('<div class="eyebrow">RACE-DENGUE / RESEARCH DEMONSTRATION</div>',unsafe_allow_html=True)
st.title(page)
st.caption('Component C03 · Reliability-aware counterfactual intervention evaluation')


def insufficient(reasons):
    st.markdown('<div class="evidence-banner"><strong>Insufficient Evidence</strong><br>A supported intervention-effect estimate is withheld.</div>',unsafe_allow_html=True)
    for reason in reasons: st.warning(str(reason))


def chart(fig):
    fig.update_layout(template='plotly_white',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font={'color':'#172d42'},margin={'l':15,'r':15,'t':110,'b':30},
        title={'y':.98,'yanchor':'top'},legend={'orientation':'h','y':1.02,'yanchor':'bottom','x':0},
        colorway=['#087e83','#df9560','#527db8','#9f6eaf','#c64e3c'])
    st.plotly_chart(fig,width='stretch')


def provenance(directory=None):
    directory=directory or snapshot.directory
    if directory: st.caption('Source: '+str(directory.relative_to(ROOT))+' · Pipeline artifacts only.')


def current_result():
    result=st.session_state.get('analysis');key=st.session_state.get('analysis_key')
    if not snapshot.ready or not result or not key or key[:2]!=(snapshot.manifest.get('run_id'),snapshot.manifest.get('dataset_sha256')):
        return None
    return result


def evidence_status(result):
    status=result['decision']
    reasons=result.get('explanation',{}).get('reasons') or result.get('warnings',[])
    if status=='INSUFFICIENT_EVIDENCE': insufficient(reasons)
    elif status in ['UNCERTAIN_REVIEW_REQUIRED','SUPPORTED_ESTIMATE']:
        if status=='UNCERTAIN_REVIEW_REQUIRED': st.warning('Uncertain — Review Required. A definitive effect estimate is not supported.')
        else: st.success('Supported Estimate — conditional on the stated data and causal assumptions.')
        for warning in reasons: st.warning(warning)
    else: raise EvidenceUnavailable('Unknown pipeline evidence status')
    additional=[warning for warning in result.get('warnings',[]) if warning not in reasons]
    if additional:
        with st.expander('Additional model warnings'):
            for warning in additional: st.warning(warning)


def overview():
    st.markdown('<p class="lede">Observed dengue counts alone cannot tell us what would have happened without an intervention. RACE-Dengue compares counterfactual estimates and makes their reliability visible.</p>',unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader('The research question')
        st.write('Can reliability-aware counterfactual AI improve the dependability of intervention estimates while abstaining when evidence is insufficient?')
    if snapshot.ready:
        results=strategy_results(snapshot);test=results[results.split.eq('test')]
        if test.empty: raise EvidenceUnavailable('No held-out comparison is available')
        stats=snapshot.audit['statistics']
        for col,label,value in zip(st.columns(4),['Source observations','Locations','Valid pseudo experiments','Test experiments'],
                [stats['rows'],stats['locations'],results.experiment_id.nunique(),test.experiment_id.nunique()]): col.metric(label,f'{value:,}')
        comparison=test.groupby('strategy',as_index=False).normalized_mae.mean()
        dynamic=comparison.loc[comparison.strategy.eq('dynamic'),'normalized_mae'].iloc[0]
        better=comparison[(comparison.strategy!='dynamic') & (comparison.normalized_mae<dynamic)].strategy.map(STRATEGIES).tolist()
        if better: st.warning('Current finding: the proposed ensemble has higher mean test error than '+', '.join(better)+'.')
        else: st.info('The proposed ensemble has the lowest mean test error in this saved run; this alone does not establish superiority.')
        comparison['Method']=comparison.strategy.map(STRATEGIES)
        chart(px.bar(comparison,x='normalized_mae',y='Method',orientation='h',title='Actual test comparison',labels={'normalized_mae':'Mean normalized MAE · lower is better'}))
        st.caption('Normalized MAE uses each experiment’s pre-period mean cases, floored at one. Overlapping windows share information; these are dependent exploratory results.')
        provenance()
    st.subheader('A standalone component')
    for col,title,text in zip(st.columns(3),['Inputs','C03 responsibility','Boundary'],[
        'Source-reviewed surveillance, intervention chronology, valid untreated controls and the frozen experiment design.',
        'Counterfactual comparison, expected error, evidence status and explicit abstention.',
        'C01 outbreak prediction, C02 community priorities and C04 resource allocation remain separate. C03 runs independently.']):
        with col,st.container(border=True): st.markdown('**'+title+'**');st.write(text)


def data_audit():
    if not snapshot.audit or snapshot.panel.empty: raise EvidenceUnavailable('The source dataset cannot be audited')
    stats=snapshot.audit['statistics']
    if snapshot.audit['valid']: st.success('Dataset validation passed — recomputed using the pipeline validator.')
    else: insufficient(snapshot.audit['errors'])
    for col,label,value in zip(st.columns(4),['Observations','Locations','Missing outcomes','Duplicate rows'],
        [stats['rows'],stats['locations'],stats['missing_outcomes'],stats['duplicate_rows']]): col.metric(label,str(value))
    st.write(f"**Observation period:** {stats['date_min']} → {stats['date_max']}")
    metadata=snapshot.metadata
    with st.expander('Original dataset and retrieval metadata',expanded=True):
        st.write(metadata.get('title','Source title unavailable'))
        st.write('Source version:',metadata.get('version','Unavailable'))
        st.write('License:',metadata.get('license',{}).get('name','Unavailable'))
        st.write('Retrieved:',snapshot.retrieval.get('retrieval_date','Unavailable'))
        st.write('DOI:',metadata.get('doi','Unavailable'))
        st.caption('SHA-256: '+snapshot.retrieval.get('sha256','Unavailable'))
    panel=snapshot.panel
    chronology=panel.groupby(['location_id','location_name'],as_index=False).agg(first_observation=('date','min'),
        last_observation=('date','max'),zone_wide_release=('intervention_start','first'),
        monitoring_start=('monitoring_start','first'),untreated_control=('control_status','first'))
    chronology['clean_history_end']=chronology.location_id.map(snapshot.config.clean_pre_end)
    st.subheader('Intervention chronology');st.dataframe(chronology,hide_index=True,width='stretch')
    st.caption('Zone-wide releases and monitoring start are different source events. Clean-history cutoffs come from the configuration. A control has no intervention date.')
    chart(px.line(panel,x='date',y='dengue_cases',color='location_name',title='Original monthly dengue notifications',labels={'date':'Month','dengue_cases':'Notified cases','location_name':'Area'}))
    st.subheader('Missingness by field')
    st.dataframe(pd.DataFrame(list(stats['column_missing_percent'].items()),columns=['Field','Missing (%)']),hide_index=True,width='stretch')
    for warning in snapshot.audit['warnings']: st.warning(warning)
    with st.expander('Per-location validation and original observations'):
        st.dataframe(pd.DataFrame(snapshot.audit['locations']),hide_index=True);st.dataframe(panel,hide_index=True)


def counterfactual():
    if not snapshot.config or snapshot.panel.empty: raise EvidenceUnavailable('Study data and configuration are required')
    st.write('Evaluate the source-recorded intervention using the unchanged pipeline. Unsupported analyses remain explicit.')
    cols=st.columns([2,2,1]);cols[0].selectbox('Study',[snapshot.config.data.study_id])
    if 'analysis_area' not in st.session_state:
        previous=st.session_state.get('selected_area')
        st.session_state['analysis_area']=previous if previous in snapshot.config.treated_units else snapshot.config.treated_units[0]
    area=cols[1].selectbox('Area',snapshot.config.treated_units,key='analysis_area')
    horizons=snapshot.manifest.get('experiment_config',{}).get('horizons',[])
    if not horizons: raise EvidenceUnavailable('No configured evaluation horizons are available')
    if 'analysis_horizon' not in st.session_state:
        previous=st.session_state.get('selected_horizon')
        st.session_state['analysis_horizon']=previous if previous in horizons else horizons[-1]
    horizon=cols[2].selectbox('Horizon (months)',horizons,key='analysis_horizon')
    st.session_state['selected_area']=area;st.session_state['selected_horizon']=horizon
    target=snapshot.panel[snapshot.panel.location_id.eq(area)]
    if target.empty: raise EvidenceUnavailable('Selected area is absent from the source')
    start=pd.Timestamp(target.monitoring_start.iloc[0]);cols=st.columns(3)
    cols[0].metric('Zone-wide release',str(pd.Timestamp(target.intervention_start.iloc[0]).date()))
    cols[1].metric('Analysis / monitoring start',str(start.date()))
    cols[2].metric('Clean history ends',snapshot.config.clean_pre_end[area])
    st.caption('Dates are locked to the source chronology. Arbitrary dates would change the causal design.')
    key=selection_key(snapshot,area,horizon)
    if st.session_state.get('analysis_key')!=key:
        st.session_state.pop('analysis',None);st.session_state.pop('analysis_key',None)
    if st.button('Run counterfactual analysis',type='primary',disabled=not snapshot.ready):
        with st.spinner('Running the real counterfactual and reliability pipeline…'):
            st.session_state['analysis']=run_analysis(ROOT,snapshot,area,horizon);st.session_state['analysis_key']=key
    result=current_result()
    if not result: st.info('Choose an area and run the analysis. No result has been substituted.');return
    evidence_status(result);score=result.get('reliability_score');effect=result.get('effect_absolute');cols=st.columns(2)
    cols[0].metric('Model Reliability Score',f'{score:.4f}' if score is not None else 'Unavailable')
    cols[1].metric('Supported effect · observed minus counterfactual',f'{effect:.2f} cases' if effect is not None else 'Withheld')
    if result['trajectories']:
        data=pd.DataFrame(result['trajectories']);fig=go.Figure()
        fig.add_trace(go.Scatter(x=data.date,y=data.upper,line={'width':0},name='Empirical upper bound',showlegend=False))
        fig.add_trace(go.Scatter(x=data.date,y=data.lower,line={'width':0},fill='tonexty',fillcolor='rgba(8,126,131,.10)',name='Empirical interval'))
        for name,label,color in [('observed','Observed','#172d42'),('counterfactual','Reliability-weighted ensemble','#087e83'),('cits','CITS','#527db8'),('synthetic_control','SCM','#df9560')]:
            fig.add_trace(go.Scatter(x=data.date,y=data[name],mode='lines+markers',name=label,line={'color':color}))
        fig.add_vline(x=str(start.date()),line_dash='dash',line_color='#8b9ba7')
        fig.update_layout(title='Observed vs counterfactual · exploratory when evidence is insufficient',xaxis_title='Month',yaxis_title='Monthly notified dengue cases',height=450)
        chart(fig)
        blocks=result.get('explanation',{}).get('calibration_calendar_years')
        st.caption(f'Calibration calendar years: {blocks if blocks is not None else "Unavailable"}. Intervals are empirical; exact coverage is not established. Dashed line: monitoring start.')
    with st.expander('Exploratory contrast — not a supported causal conclusion'): st.json(result.get('exploratory_contrast'))
    with st.expander('Assumptions and result provenance'): st.write(result.get('assumptions',[]));st.json(result.get('provenance',{}))
    st.info('Continue to Model Comparison, Reliability or Export to inspect this same analysis.')


def model_comparison():
    results=strategy_results(snapshot);cols=st.columns(2)
    split=cols[0].selectbox('Evaluation partition',[s for s in ['test','validation','calibration','train'] if s in set(results.split)])
    area=cols[1].selectbox('Comparison area',['All areas',*sorted(results.treated_location.unique())])
    selected=results[results.split.eq(split)]
    if area!='All areas': selected=selected[selected.treated_location.eq(area)]
    if selected.empty: st.warning('No eligible experiments exist for this area and partition. No comparison is available.');return
    comparison=selected.groupby('strategy')[['mae','rmse','normalized_mae']].mean().reindex(STRATEGIES).reset_index()
    comparison['Method']=comparison.strategy.map(STRATEGIES)
    st.caption(f'{selected.experiment_id.nunique()} paired pseudo-intervention experiments · {split} partition. Values are means of per-experiment metrics, not pooled RMSE.')
    chart(px.bar(comparison,x='Method',y='normalized_mae',title='Counterfactual accuracy',labels={'normalized_mae':'Mean normalized MAE · lower is better'}))
    st.dataframe(comparison[['Method','mae','rmse','normalized_mae']],hide_index=True,width='stretch')
    st.subheader('Learned reliability weights')
    dynamic=selected[selected.strategy.eq('dynamic')]
    weights=pd.DataFrame([json.loads(x) for x in dynamic.model_weights],index=dynamic.experiment_id);weights.index.name='experiment_id'
    long=weights.reset_index().melt(id_vars='experiment_id',var_name='Estimator',value_name='Weight')
    chart(px.bar(long,x='experiment_id',y='Weight',color='Estimator',barmode='stack',title='Weights from the frozen error learner',labels={'experiment_id':'Pseudo intervention'}))
    result=current_result()
    if result and result['model_weights']:
        st.write('Recorded-intervention weights for the current analysis:',result['location_id'])
        st.dataframe(pd.DataFrame(list(result['model_weights'].items()),columns=['Estimator','Weight']),hide_index=True)
    st.subheader('Validation decisions and diagnostics')
    st.dataframe(read_table(snapshot.directory/'temperature_validation.csv',['model_kind','temperature','validation_normalized_mae']),hide_index=True,width='stretch')
    st.caption('Selected family: '+snapshot.manifest['selected_model']+' · Temperature: '+str(snapshot.manifest['temperature']))
    errors=read_table(snapshot.directory/'expected_errors.csv',['split','experiment_id','model_name','predicted_error','target_error'])
    validation=errors[errors.split.eq('validation')]
    columns=[c for c in ['experiment_id','model_name','target_error','predicted_error',*snapshot.manifest.get('numeric_features',[])] if c in validation]
    with st.expander('Validation diagnostic records'): st.dataframe(validation[columns],hide_index=True)
    with st.expander('Expected-error prediction quality · both learners and historical average'): st.dataframe(read_table(snapshot.directory/'meta_metrics.csv'),hide_index=True)
    provenance()


def reliability():
    result=current_result()
    if result:
        st.write('Current recorded-intervention analysis:',result['location_id']);evidence_status(result)
        score=result.get('reliability_score');st.metric('Model Reliability Score',f'{score:.4f}' if score is not None else 'Unavailable')
        st.caption('Relative expected-error score, not a probability of causality or clinical risk.')
        factors=result.get('explanation',{}).get('factors',{})
        if factors:
            st.subheader('Main reliability factors');st.dataframe(pd.DataFrame(list(factors.items()),columns=['Diagnostic','Value']),hide_index=True,width='stretch')
            st.caption('Measured diagnostic signals and pipeline rules; these are not causal feature attributions.')
        for reason in result.get('explanation',{}).get('unavailable_signals',[]): st.warning('Unavailable signal: '+reason)
    else: st.info('Run Counterfactual Analysis to see an area-specific score, evidence status and warnings.')
    if not snapshot.ready: return
    st.subheader('Frozen abstention policy');design=read_json(snapshot.directory/'frozen_design.json');selector=design['selector'];cols=st.columns(3)
    cols[0].metric('Selected score threshold',f"{selector['threshold']:.6f}")
    cols[1].metric('Validation retained coverage',f"{selector['validation_coverage']:.2%}")
    cols[2].metric('Calibration calendar years',str(design['calibration']['block_count']))
    evaluation=read_table(snapshot.directory/'selective_evaluation.csv',['split','location','coverage','retained_count','analyses','retained_error'])
    st.dataframe(evaluation[evaluation.location.eq('ALL')],hide_index=True,width='stretch')
    risk=read_table(snapshot.directory/'risk_coverage.csv',['coverage','retained_error'])
    fig=px.line(risk,x='coverage',y='retained_error',markers=True,title='Test error versus retained coverage',labels={'coverage':'Retained fraction','retained_error':'Mean retained normalized MAE'})
    point=evaluation[(evaluation.split=='test') & (evaluation.location=='ALL')]
    if not point.empty and point.retained_error.notna().all():
        fig.add_trace(go.Scatter(x=point.coverage,y=point.retained_error,mode='markers',marker={'size':16,'symbol':'star','color':'#c64e3c'},name='Validation-selected operating point'))
    chart(fig)
    st.caption('The test curve is descriptive; it does not tune the threshold. Score retention is separate from the full evidence policy, including gap, donor and calibration requirements.')
    with st.expander('Exact saved policy and feature availability'):
        st.json(design['policy']);st.write('Omitted diagnostics:',snapshot.manifest.get('omitted_features',[]));st.write('Constant measured diagnostics:',snapshot.manifest.get('constant_numeric_features',[]))
    provenance()


def robustness():
    if not snapshot.ready: raise EvidenceUnavailable('A validated run is required to establish robustness provenance')
    archived=archived_robustness(ROOT,snapshot);labels=['Current run']+['Archived run · '+m['run_id'] for m in archived]
    label=st.selectbox('Robustness artifact source',labels)
    if label=='Current run': directory=snapshot.directory
    else:
        directory=ROOT/archived[labels.index(label)-1]['directory']
        st.warning('Archived evidence: these tests belong to an earlier frozen model and do not validate the current run.')
    provenance(directory)
    for name,title in [('placebos','In-time placebo tests'),('in_space_placebos','In-space placebo eligibility'),('donor_sensitivity','Donor sensitivity'),('stress_tests','Artificial stress tests')]:
        st.subheader(title);path=directory/f'{name}.csv'
        if path.exists():
            table=read_table(path);st.dataframe(table,hide_index=True,width='stretch')
            if name=='placebos' and {'model_name','apparent_effect'}<=set(table):
                if 'split' in table: table=table[table.split.eq('test')]
                chart(px.box(table,x='model_name',y='apparent_effect',points='all',title='Saved test-period placebo contrasts',labels={'apparent_effect':'Apparent case difference','model_name':'Estimator'}))
        else: st.warning('Not run in the selected artifact run. No result is substituted.')
        if name=='donor_sensitivity':
            donors=snapshot.panel[snapshot.panel.control_status.eq(True)].location_id.nunique()
            st.caption(f'Permanent untreated donors in the source: {donors}.')
            if donors<=1: st.info('Removing the sole donor leaves no comparator. Donor-removal stability cannot be established.')
    st.caption('Artificial stress scenarios are methodological checks, not additional observed interventions. Unavailable and failed tests remain visible in the source tables.')


def export():
    result=current_result()
    if result:
        st.write('Selected analysis:',result['location_id']);evidence_status(result);cols=st.columns(3)
        cols[0].download_button('Download result JSON',export_payload(result,snapshot),'race_dengue_analysis.json','application/json')
        cols[1].download_button('Download result CSV',result_csv(result),'race_dengue_analysis.csv','text/csv')
        cols[2].download_button('Download analysis report',result_report(result,snapshot),'race_dengue_analysis.md','text/markdown')
        st.caption('Exports retain evidence status, warnings and null unsupported effects. JSON includes run and dataset provenance.')
    else: st.info('Run Counterfactual Analysis first to export its result, including an abstention.')
    if snapshot.ready:
        st.subheader('Saved research artifacts')
        for name in ['strategy_results.csv','expected_errors.csv','selective_evaluation.csv','frozen_design.json']:
            path=snapshot.directory/name
            if path.exists(): st.download_button('Download '+name,path.read_bytes(),name)
        summary=ROOT/'reports/pp1_summary.md'
        if summary.exists() and snapshot.manifest['run_id'] in summary.read_text(): st.download_button('Download PP1 summary',summary.read_bytes(),'pp1_summary.md','text/markdown')
        st.download_button('Download current data audit',json.dumps(snapshot.audit,indent=2),'data_audit.json','application/json')
        provenance()


if not snapshot.ready:
    st.session_state.pop('analysis',None);st.session_state.pop('analysis_key',None)
    insufficient(snapshot.data_issues+snapshot.issues)
try:
    {'Overview':overview,'Data Audit':data_audit,'Counterfactual Analysis':counterfactual,'Model Comparison':model_comparison,
     'Reliability':reliability,'Robustness':robustness,'Export':export}[page]()
except Exception as exc:
    logging.exception('Dashboard evidence could not be rendered')
    insufficient([f'This view cannot be validated: {exc}'])
st.markdown('<div class="about-box"><strong>About / Limitations</strong><br>'+ABOUT+'</div>',unsafe_allow_html=True)
