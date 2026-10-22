# Optional integration

C03 trains and runs without C01/C02/C04 outputs. C02 may supply location/context;
C04 may inspect evidence. Abstention must not become a supported benefit downstream.

```bash
.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

`GET /health` reports model availability. `POST /analyze` calls the same pipeline as
the dashboard and CLI. Use the source monitoring date, not an arbitrary change point:

```json
{"study_id":"niteroi_zone_v3","location_id":"zone_2","intervention_date":"2017-09-01","horizon_months":6}
```

Unsupported evidence retains null main effect and interval fields. A separately
labelled `exploratory_contrast` is inspectable, not a supported causal conclusion.
Responses preserve warnings, reasons, assumptions, source hash, model run, weights
and trajectories. Model Reliability Score is not clinical risk or causal probability.

The service is local and synchronous. Public hosting/access controls and clinical
deployment are outside this research implementation.
