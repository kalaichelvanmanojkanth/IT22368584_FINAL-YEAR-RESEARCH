# Source-to-canonical mapping

The exact workbook `Metadata` sheet is authoritative for variable definitions.
No rows are fabricated, pooled across studies, resampled, or imputed.

| Source | Canonical | Transformation |
|---|---|---|
| yearmonth | date, year, month | YYYY_MM to first day of calendar month |
| zone | location_name, location_id | Explicit Zone 1–4 / Untreated control map |
| dengue_cases | dengue_cases | Notified counts, preserving missing values |
| pop | population | Source census denominator, including change in 2010 |
| dengue_cases / pop | dengue_incidence | Cases per 100,000 population |
| group | control_status | group == 0 |
| yearmonth_start | intervention_start | Zone-wide release start, not earliest local release |
| start_release | intervention_status | Source binary zone-wide release indicator |
| yearmonth_monitor | monitoring_start | Source post-intervention monitoring start |
| epi_monitor | monitoring_status | Source monitoring indicator |
| wmel_pc | wolbachia_prevalence | Percentage divided by 100; absent remains null |
| no source column | intervention_exposure, latitude, longitude | Null |
| versioned repository | study_id, country, source_file, source_version | Explicit provenance |
| retrieval.json | retrieval_date | Recorded retrieval time; null if unknown |
| dengue_cases is null | missing_flag | Target missingness only |

The complete source remains available, including chikungunya, Zika and
entomological counts that are not used as dengue model predictors.
The loader sorts by study/location/month; validation checks chronological order
of any input it receives. Gaps are reported, never turned into zero counts.

## Clean analysis windows

`intervention_status=0` does not guarantee absence of local Wolbachia releases.
The conservative default training windows end in December 2014 for Zone 1 and
December 2016 for Zones 2–4. These are analyst-defined exclusions, not inferred
biological exposure dates. The source monitoring date determines the six-month
post-analysis window. All intervening treated rows are excluded from training.
The untreated control is the sole default donor; treated zones are never relabelled
as permanent controls to increase the donor count.

Later neighbourhood/site adapters require their own schema, intervention design,
licence and clean-window review. The external raw directories are reserved only.
