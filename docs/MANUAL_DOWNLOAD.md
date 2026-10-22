# Original source download

No generated-data fallback exists. A missing or altered source stops the pipeline.

1. Open the [original publication](https://doi.org/10.1371/journal.pntd.0009556).
2. Follow its zone-level data link to [Figshare version 3](https://doi.org/10.6084/m9.figshare.13662203.v3).
3. Download `Niteroi_ZoneData_May2021_Shared.xls` (199168 bytes). The versioned
   metadata endpoint is https://api.figshare.com/v2/articles/13662203/versions/3;
   the original file URL is https://ndownloader.figshare.com/files/28008525.
4. Place the unchanged XLS under `data/raw/niteroi/`. Do not export to CSV or
   change cell values. The loader expects sheets `Data` and `Metadata`.
5. Verify MD5 `a31b89e0f2bc0a4d17271b3a386defb8`. The loader also verifies it.
6. Run `.venv/bin/python -m src.pipeline.evaluate audit` from the repository.

Automated alternative: `.venv/bin/python -m src.pipeline.evaluate download`.
TLS verification remains enabled. On a restricted network, perform the manual
download in a browser or on a network that permits Figshare access.

Automated retrieval saves source metadata, retrieval time and SHA256. For a manual
download with no retrieval manifest, retrieval_date remains null, rather than
inventing a download time from the file modification time. Record your actual
retrieval time and SHA256 in `retrieval.json` using the downloaded manifest format
if known. Source files are never overwritten by the download command.

Dataset attribution: Anders, Katie (2021), Zone-level wMel, dengue, chik and Zika
data. Monash University, Figshare v3, CC BY 4.0. Processed columns are transformations
by this project. Neighbourhood v2 (not loaded in Phase A/B):
https://doi.org/10.6084/m9.figshare.13662230.v2.
