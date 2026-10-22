"""Exercise a running local app and save unedited browser screenshots and exports.

Optional tooling: pip install playwright; install Chrome or playwright install chromium.
Run from the project root: python scripts/capture_pp1_dashboard.py --url http://127.0.0.1:8501
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright, expect
from app.evidence import ABOUT, PAGES
from src.pipeline.predict import analyze


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8501')
    parser.add_argument('--channel',default='chrome',help='Use chromium for Playwright-managed Chromium')
    args=parser.parse_args()
    output=ROOT/'artifacts/figures/pp1_dashboard';output.mkdir(parents=True,exist_ok=True)
    downloads=ROOT/'artifacts/exports/pp1_dashboard';downloads.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'artifacts/latest_reliability.json').read_text())
    verification={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'url':args.url,
        'reliability_run':manifest['run_id'],'dataset_sha256':manifest['dataset_sha256'],
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                         ['app/streamlit_app.py','app/evidence.py','.streamlit/config.toml']},
        'screenshots':[],'pages_checked':[],'downloads':[],'browser_errors':[],'status':'running'}
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,**({'channel':args.channel} if args.channel!='chromium' else {}))
        verification['browser_version']=browser.version
        context=browser.new_context(viewport={'width':1440,'height':1100},device_scale_factor=1,accept_downloads=True)
        page=context.new_page()
        page.on('pageerror',lambda error:verification['browser_errors'].append(str(error)))

        def settled():
            # Streamlit can briefly retain stale elements during a rerun/responsive resize.
            page.wait_for_timeout(250)
            expect(page.locator('.about-box')).to_have_count(1,timeout=30000)
            expect(page.locator('.about-box')).to_contain_text(ABOUT,timeout=30000)
            page.wait_for_function("document.querySelector('[data-testid=stApp]')?.getAttribute('data-test-script-state') !== 'running'",timeout=30000)
            expect(page.locator('[data-testid=stException]')).to_have_count(0)
            assert 'This view cannot be validated' not in page.locator('body').inner_text()
            page.evaluate('document.fonts.ready')

        def navigate(name):
            page.locator('[data-testid=stSidebar]').get_by_text(name,exact=True).click()
            expect(page.get_by_role('heading',name=name,exact=True,level=1)).to_be_visible(timeout=30000)
            settled()
            if name not in verification['pages_checked']: verification['pages_checked'].append(name)

        def screenshot(name):
            page.set_viewport_size({'width':1440,'height':1100})
            settled()
            height=page.locator('[data-testid=stMain]').evaluate('(el)=>el.scrollHeight')
            page.set_viewport_size({'width':1440,'height':min(6000,max(1100,height+100))})
            page.locator('[data-testid=stMain]').evaluate('(el)=>el.scrollTo(0,0)')
            page.screenshot(path=str(output/name),full_page=True,animations='disabled')
            verification['screenshots'].append(str((output/name).relative_to(ROOT)))

        try:
            response=page.goto(args.url,wait_until='domcontentloaded')
            assert response and response.status==200
            settled();verification['pages_checked'].append('Overview')
            expect(page.get_by_role('heading',name='Overview',exact=True,level=1)).to_be_visible()
            screenshot('01_overview.png')
            navigate('Data Audit');screenshot('02_data_audit.png')
            navigate('Counterfactual Analysis')
            page.get_by_role('button',name='Run counterfactual analysis',exact=True).click()
            expect(page.locator('.evidence-banner')).to_contain_text('Insufficient Evidence',timeout=30000)
            settled();expect(page.locator('[data-testid=stPlotlyChart]')).to_have_count(1)
            screenshot('03_counterfactual_insufficient_evidence.png')
            navigate('Model Comparison');screenshot('04_model_comparison.png')
            navigate('Reliability');screenshot('05_reliability.png')
            navigate('Robustness');screenshot('06_robustness_current.png')
            page.get_by_role('combobox').click()
            archives=page.get_by_role('option').filter(has_text='Archived run')
            if archives.count():
                archives.first.click();settled()
                expect(page.get_by_text('Archived evidence:',exact=False)).to_be_visible()
                screenshot('06b_robustness_archived.png')
                verification['archived_robustness_checked']=True
            else: page.keyboard.press('Escape')
            navigate('Export');screenshot('07_export.png')
            for label in ['Download result JSON','Download result CSV','Download analysis report']:
                with page.expect_download() as download_info:
                    page.get_by_role('button',name=label,exact=True).click()
                download=download_info.value;path=downloads/download.suggested_filename;download.save_as(path)
                assert path.stat().st_size>0
                verification['downloads'].append(str(path.relative_to(ROOT)))
            exported=json.loads((downloads/'race_dengue_analysis.json').read_text())
            assert exported['dashboard_context']['run_id']==manifest['run_id']
            actual=analyze(ROOT,exported['study_id'],exported['location_id'],horizon_months=len(exported['trajectories']))
            assert exported['decision']==actual['decision']=='INSUFFICIENT_EVIDENCE'
            assert exported['effect_absolute'] is None and exported['effect_percent'] is None
            assert exported['reliability_score']==actual['reliability_score']
            pd.testing.assert_frame_equal(pd.DataFrame(exported['trajectories']),pd.DataFrame(actual['trajectories']))
            csv=pd.read_csv(downloads/'race_dengue_analysis.csv')
            np.testing.assert_allclose(csv.counterfactual,[r['counterfactual'] for r in actual['trajectories']])
            assert csv.effect_absolute.isna().all() and csv.decision.eq(actual['decision']).all()
            assert ABOUT in (downloads/'race_dengue_analysis.md').read_text()
            verification['export_matches_pipeline']=True
            navigate('Overview')
            page.set_viewport_size({'width':390,'height':844});settled()
            page.screenshot(path=str(output/'08_mobile_overview.png'),full_page=True,animations='disabled')
            verification['screenshots'].append(str((output/'08_mobile_overview.png').relative_to(ROOT)))
            assert set(verification['pages_checked'])==set(PAGES)
            assert not verification['browser_errors']
            verification['status']='passed'
            (output/'failure.png').unlink(missing_ok=True)
        except Exception as exc:
            verification['status']='failed';verification['failure']=str(exc)
            page.screenshot(path=str(output/'failure.png'),full_page=True)
            raise
        finally:
            (output/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
            browser.close()
    print(json.dumps(verification,indent=2))


if __name__=='__main__': main()
