# External adapters and comparability review

External raw files are separate, checksum verified and excluded from Niteroi
training/selection. The initial adapters produce source-specific canonical panels
and metadata. They do not claim external validation of the reliability learner.

## Yogyakarta

- Original data: https://doi.org/10.6084/m9.figshare.21321192.v1
- Publication: https://doi.org/10.1080/16549716.2023.2166650
- File: `YogyaCity_wMel and dengue case dataset to May22.xlsx`
- Direct source: https://ndownloader.figshare.com/files/37842813
- MD5: `a470d9fb5d1be3cc64338933ee220ec2`; licence CC BY 4.0.
- Manual location: `data/raw/yogyakarta/`; preserve workbook unchanged.

The workbook defines DHF and DF separately; the adapter maps **DHF only** to the
canonical outcome and preserves DF separately. It does not backfill absent DF with
zero or sum series across their reporting changes. Treatment is multi-state and
includes contamination; randomized deployment boundaries differ from administrative
surveillance units. `avg_wmelpc` and `avg_wmel_quint` are centred averages using the
previous/current/NEXT month and are excluded from predictor inputs.

## Aburra Valley, Colombia

- Original data: https://doi.org/10.6084/m9.figshare.22724663.v1
- Publication: https://doi.org/10.1371/journal.pntd.0011713
- File: `AV dengue case and wMel time series data_Figshare.xls`
- Direct source: https://ndownloader.figshare.com/files/42267318
- MD5: `249059310332f70e06ccda401dd217a1`; licence CC BY 4.0.
- Manual location: `data/raw/colombia/`; preserve workbook unchanged.

The three city sheets retain city/reporting area identifiers. Medellin/Bello exclude
IgM-negative notifications; Itagui supplies aggregate cases. Preserve exact source
names rather than guessing corrections to original encoding. Raw `wmel_pc` is kept;
source-interpolated `wmel_ipol` is excluded from predictors. Both files contain
source-coded prevalence zeros; adapters preserve source values without generating
zeros for missing cells.

## Decision

Outcomes and intervention/control definitions are not directly comparable with
Niteroi. Do not pool them. Source status and release dates are preserved; permanent
control eligibility stays null until a study-specific design is validated. The
current CITS/SCM interfaces will reject these unknown donor labels. Audit outputs
are not evidence that those panels are eligible for the current estimators.

Run adapters:

```bash
.venv/bin/python -m src.data.external
```

Next research work requires design-specific control sets (or an appropriate
uncontrolled ITS design), contamination-aware clean windows and separate outcome
normalization. An external final-test designation must remain frozen before model
selection. No external performance numbers are currently available.
