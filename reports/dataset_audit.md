# Dataset audit

Validation: PASS

Source: https://doi.org/10.6084/m9.figshare.13662203.v3
Publication: https://doi.org/10.1371/journal.pntd.0009556
Licence: CC BY 4.0

## Computed statistics

- rows: 810
- studies: 1
- locations: 5
- date_min: 2007-01-01
- date_max: 2020-06-01
- duplicate_rows: 0
- missing_outcomes: 0
- missing_outcome_percent: 0.0

## Per-location audit

| Location | Rows | Control | Source pre | Source post | Missing outcomes | Gaps |
|---|---:|---|---:|---:|---:|---:|
| control | 162 | True | 162 | 0 | 0 | 0 |
| zone_1 | 162 | False | 123 | 39 | 0 | 0 |
| zone_2 | 162 | False | 127 | 35 | 0 | 0 |
| zone_3 | 162 | False | 130 | 32 | 0 | 0 |
| zone_4 | 162 | False | 152 | 10 | 0 | 0 |

Source pre labels are not guaranteed unexposed: conservative training cutoffs are recorded in configs/niteroi.yaml and each run manifest.

## Column missingness (%)

- study_id: 0.0000
- country: 0.0000
- location_id: 0.0000
- location_name: 0.0000
- date: 0.0000
- year: 0.0000
- month: 0.0000
- dengue_cases: 0.0000
- population: 0.0000
- dengue_incidence: 0.0000
- intervention_status: 0.0000
- intervention_start: 20.0000
- intervention_exposure: 100.0000
- wolbachia_prevalence: 87.6543
- control_status: 0.0000
- latitude: 100.0000
- longitude: 100.0000
- source_file: 0.0000
- source_version: 0.0000
- retrieval_date: 0.0000
- missing_flag: 0.0000

## Errors

None detected.

## Warnings and study limitations

One permanent control zone
Zone 1 includes 2015-16 pilot
Partial releases can precede zone-wide start
Suspected and confirmed notifications
Population uses 2000 census in 2007-09, 2010 census subsequently
No verified unit centroids or independent exposure dose
