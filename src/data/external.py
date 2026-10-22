"""Source-specific external adapters; audit only, never silently pooled for training."""
from pathlib import Path
import hashlib
import json
import pandas as pd
from src.data.loaders import parse_source_month
from src.data.schema import CANONICAL_COLUMNS,DataError,DataUnavailableError

SOURCES={
    'yogyakarta':{'article':21321192,'version':1,'file':'YogyaCity_wMel and dengue case dataset to May22.xlsx',
                  'md5':'a470d9fb5d1be3cc64338933ee220ec2','country':'Indonesia',
                  'publication':'https://doi.org/10.1080/16549716.2023.2166650',
                  'study_design':'Secondary notified-disease analysis around randomized deployments with administrative/cluster boundary mismatch',
                  'outcome_definition':'Hospitalised notified DHF only; DF retained separately, never silently summed across surveillance changes'},
    'colombia':{'article':22724663,'version':1,'file':'AV dengue case and wMel time series data_Figshare.xls',
                'md5':'249059310332f70e06ccda401dd217a1','country':'Colombia',
                'publication':'https://doi.org/10.1371/journal.pntd.0011713',
                'study_design':'City-wide phased deployments; interrupted time series',
                'outcome_definition':'Notified suspected cases excluding IgM test-negative cases in Medellin/Bello; aggregate notifications in Itagui'},
}


def load_external(root: Path, site: str) -> tuple[pd.DataFrame,dict]:
    """Preserve source treatment states; permanent control membership remains unknown."""
    if site not in SOURCES: raise DataError('Unknown external study adapter')
    spec=SOURCES[site]; directory=root/'data/raw'/site; path=directory/spec['file']
    if not path.exists(): raise DataUnavailableError(f'Missing real {site} file: {path}; see docs/EXTERNAL_SITES.md')
    if hashlib.md5(path.read_bytes()).hexdigest()!=spec['md5']: raise DataError('External source checksum mismatch')
    if site=='yogyakarta':
        raw=pd.read_excel(path,sheet_name='Data',engine='openpyxl')
        raw['city']='Yogyakarta';name='kelurahan';outcome='dhf';start='firstrelease';end='lastrelease'
    else:
        frames=[]
        for city in ['Medellin','Bello','Itagui']:
            frame=pd.read_excel(path,sheet_name=f'Data_{city}',engine='xlrd');frame['city']=city;frames.append(frame)
        raw=pd.concat(frames,ignore_index=True);name='reportingarea';outcome='denvnotifiedcase';start='year_month_start';end='year_month_end'
    required={name,outcome,start,end,'year_month','population','wmel_pc','wmel_status'}
    if not required<=set(raw): raise DataError(f'Unexpected source columns: {required-set(raw)}')
    frame=pd.DataFrame(index=raw.index,columns=CANONICAL_COLUMNS)
    frame['study_id']=f'{site}_figshare_{spec["article"]}_v1';frame['country']=spec['country']
    frame['location_name']=raw[name]
    # Preserve exact source spelling, including any original encoding issue; city avoids name collision.
    frame['location_id']=raw.city.astype(str)+':'+raw[name].astype(str);frame['city']=raw.city
    frame['date']=parse_source_month(raw.year_month);frame['year']=frame.date.dt.year;frame['month']=frame.date.dt.month
    frame['dengue_cases']=pd.to_numeric(raw[outcome],errors='raise');frame['population']=pd.to_numeric(raw.population,errors='raise')
    frame['dengue_incidence']=frame.dengue_cases/frame.population*100_000
    frame['intervention_start']=parse_source_month(raw[start],nullable=True)
    frame['release_end']=parse_source_month(raw[end],nullable=True)
    frame['source_exposure_status']=raw.wmel_status
    frame['intervention_status']=pd.Series(pd.NA,index=raw.index,dtype='Int64')
    known=frame.intervention_start.notna()
    frame.loc[known,'intervention_status']=frame.loc[known,'date'].ge(frame.loc[known,'intervention_start']).astype(int)
    # Neither source design establishes a permanent untreated donor set for the Niteroi estimator interface.
    frame['control_status']=pd.Series(pd.NA,index=raw.index,dtype='boolean')
    frame['wolbachia_prevalence']=pd.to_numeric(raw.wmel_pc,errors='raise')/100
    frame['source_file']=spec['file'];frame['source_version']=f"figshare.{spec['article']}.v1"
    retrieval=directory/'retrieval.json'
    if retrieval.exists():
        manifest=json.loads(retrieval.read_text())
        if manifest['sha256']!=hashlib.sha256(path.read_bytes()).hexdigest(): raise DataError('External retrieval manifest mismatch')
        frame['retrieval_date']=manifest['retrieval_date']
    frame['missing_flag']=frame.dengue_cases.isna()
    if site=='yogyakarta':
        frame['df_cases_separate']=raw.df
        frame['contamination_start']=parse_source_month(raw.wmel_contamination,nullable=True)
        # avg_wmelpc is a CENTRED average; deliberately excluded, including from the canonical feature set.
    metadata={'study_id':frame.study_id.iloc[0],'study_design':spec['study_design'],
              'intervention_type':'wMel Wolbachia mosquito deployments','description':spec['outcome_definition'],
              'data_source':f"https://doi.org/10.6084/m9.figshare.{spec['article']}.v1",'publication':spec['publication'],
              'licence':'CC BY 4.0','geographic_resolution':'kelurahan' if site=='yogyakarta' else 'city/reporting area',
              'time_resolution':'monthly','treatment_definition':'Source first release date plus preserved multi-state exposure status; requires design-specific review',
              'control_definition':'Not established as a permanent donor set; control_status remains null',
              'known_limitations':['Outcome definition differs from Niteroi','No verified permanent control set',
                                  'Release indicator alone does not capture full/partial exposure or contamination',
                                  'No cross-site model training/evaluation performed'],
              'excluded_predictors':['avg_wmelpc','avg_wmel_quint'] if site=='yogyakarta' else ['wmel_ipol'],
              'exclusion_reason':'Centred averaging uses next month' if site=='yogyakarta' else 'Source interpolation may use future monitoring; not used as predictor'}
    return frame.sort_values(['study_id','location_id','date']).reset_index(drop=True),metadata


def audit_external(root: Path, site: str) -> dict:
    frame,metadata=load_external(root,site)
    duplicates=int(frame.duplicated(['study_id','location_id','date']).sum())
    if duplicates or (frame.dengue_cases.dropna()<0).any() or (frame.population.dropna()<=0).any():
        raise DataError('Invalid external duplicates/count/population values')
    audit={'study_id':metadata['study_id'],'rows':len(frame),'locations':int(frame.location_id.nunique()),
           'date_min':str(frame.date.min().date()),'date_max':str(frame.date.max().date()),
           'duplicate_months':duplicates,'missing_outcomes':int(frame.dengue_cases.isna().sum()),
           'unknown_control_rows':int(frame.control_status.isna().sum()),
           'outcome_definition':metadata['description'],'training_eligible':False,
           'reason':'Noncomparable outcomes/intervention designs; permanent controls and clean windows require separate review'}
    processed=root/'data/processed';processed.mkdir(parents=True,exist_ok=True)
    frame.to_csv(processed/f'{site}.csv',index=False)
    (processed/f'{site}_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (processed/f'{site}_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return audit


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2]
    audits=[audit_external(root,site) for site in SOURCES]
    (root/'reports/external_data_audit.md').write_text('# External source audit — no pooled evaluation\n\n'+
        '\n\n'.join('```json\n'+json.dumps(a,indent=2)+'\n```' for a in audits)+'\n')
    print(json.dumps(audits,indent=2))
