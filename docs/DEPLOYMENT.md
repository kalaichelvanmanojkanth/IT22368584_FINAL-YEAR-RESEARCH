# Running and hosting the PP1 dashboard

The local dashboard and an online deployment are separate. Opening Streamlit's
Deploy menu does not publish this local folder. An online deployment has not yet
been configured or verified for this project.

## Local demonstration

From `race-dengue`:

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501` on the same computer. Keep the server process running.
This loopback address does not provide access from another computer or phone.

## Streamlit Community Cloud

Community Cloud requires a GitHub repository containing the application and its
dependencies. The designated repository is
`kalaichelvanmanojkanth/IT22368584_FINAL-YEAR-RESEARCH`, branch `main`.

Use `race-dengue` as the repository root and `app/streamlit_app.py` as the app entry
point. `requirements.txt` includes the existing `requirements.lock.txt`; retain
both. Select Python 3.13 to match the saved local model environment. Keep the
pinned scikit-learn and joblib versions when loading these artifacts.

The repository/deployment must include the actual evidence, not only the UI:

- `app/`, `src/`, `configs/`, and `.streamlit/config.toml`.
- The original checksum-verified Niteroi workbook named in `configs/niteroi.yaml`,
  plus its `source_metadata.json` and `retrieval.json`.
- `artifacts/latest_reliability.json` and its referenced run directory, including
  `manifest.json`, `bundle.joblib`, frozen design and the saved evaluation tables.
- `reports/pp1_summary.md` and the saved test XML files if their results should
  be available in the UI.
- Earlier run directories only if archived robustness results should be available.

The `.gitignore` excludes raw XLS/XLSX files by default, with explicit exceptions
for the three reviewed public CC BY 4.0 study workbooks. These originals and their
source metadata are included for reproducibility. Preserve their attribution and
license. Do not substitute sample data or retrain silently during app startup.

The shared configuration does not force a loopback address, allowing the hosting
platform to choose its listener. The local launch command above explicitly binds
to `127.0.0.1`. Use the port assigned by the host for a hosted deployment.

After deployment, verify all seven pages, a recorded-intervention analysis and
the JSON/CSV exports at the actual hosted URL. Local browser checks do not establish
that a cloud deployment succeeded. Missing data/models must remain an explicit
Insufficient Evidence state.

Official instructions:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
