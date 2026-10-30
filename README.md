# Post-pandemic shifts in physical fitness among Korean preschool children — analysis code

Analysis code for the manuscript *Post-Pandemic Shifts in Physical Fitness Among Korean Preschool Children: Age-Specific and Distributional Evidence from a Population Surveillance Study* (Oh M-Y, Han Y-O, Yoo IH, Kim SH; Scandinavian Journal of Medicine & Science in Sports, under revision).

## Contents

| File | Purpose |
|---|---|
| `analysis_v2.py` | Complete analysis: data processing, age–sex-adjusted regressions, age-group and sex × BMI subgroups with interaction tests, percentiles within 6-month age bands, quantile regression (complementary analysis), sensitivity analyses, missingness audit, figures |
| `ref/kdca2017_bmi_lms.csv` | Monthly BMI L/M/S parameters of the 2017 Korean National Growth Charts (KDCA, public) |
| `ref/iotf_bmi_lms.csv`, `ref/who0607_bmi_lms.csv`, `ref/iotf_z_cutoffs.csv` | IOTF and WHO 2006/2007 BMI references, exported from the R package `sitar` by `ref/export_lms.R` |

## Data

The surveillance records are held by the Seoul Metropolitan Government and were released to the authors under conditions that do not permit redistribution. They are **not** included. The script expects a CSV with the columns
`covid, time, year, sex, age, height, weight, endu, flex, equi, agil, quick`
(see the docstring of `analysis_v2.py` for the variable definitions; note that `agil` is the 5-metre shuttle run in seconds and `quick` the standing long jump in centimetres).

Give the path in one of three ways (checked in this order): an environment variable, a `local_paths.json` file next to the script (`{"PFN_DATA": "...", "PFN_PRIOR": "...", "PFN_REPEAT": "..."}`; ignored by git), or by placing the file at `data/data3.csv`.

```bash
export PFN_DATA=/path/to/data3.csv          # required
export PFN_PRIOR=/path/to/prior_extract.csv # optional: input file of the earlier report (2017–2019 sit-and-reach zeros — required to reproduce the SRT results)
export PFN_REPEAT=/path/to/file              # optional: programme's 2017-2019 repeat-measurement file (repeat-measurement sensitivity analysis; required to reproduce that column of Table 5)
python analysis_v2.py                        # -> output_v2/{tables,supplementary,figures}
Rscript gamlss_compare.R                     # -> GAMLSS centiles for Table S7 (reads output_v2/private/, which holds
                                             #    record-level data for local use only; never distribute)
```

## Output files and manuscript tables

| File written by the script (`output_v2/tables/` or `output_v2/supplementary/`) | Manuscript |
|---|---|
| `table1_baseline.csv` | Table 2 (characteristics of records) |
| `table2_overall.csv` | Table 3A (overall adjusted differences) and the months-of-development equivalents in Results 3.2 |
| `table3_age_group.csv` | Table 3B and Supplementary Table S5A (age groups) |
| `table3_interaction.csv` | Table 3B, interaction *p* (period × age group; robust Wald χ², 3 df) |
| `table4_band_percentiles.csv` | Table 4 (percentile differences within 6-month age bands, averaged across bands; `pre_value`/`post_value` are the band-averaged percentiles, `diff`/`pct` are post − pre, and the `deficit` columns shown in the paper are oriented so that positive = post-pandemic deficit) |
| `table5_robustness.csv` | Table 5 (sensitivity analyses) |
| `S_audit_cleaning.csv` | Supplementary Table S2 (processing audit) |
| `S_srt_zero_restore.csv` | Methods 2.4 (0 cm is a valid sit-and-reach score; 2017–2019 zeros recovered from the earlier report's input file; aggregate counts) |
| `S_repeat_advantage.csv` | Results 3.7 and Supplementary Table S3 (later vs first measurements within each period, age–sex adjusted) |
| `S_recording_checks.csv` | Supplementary Table S2 (recorded precision by period; standardised differences with a common precision and a common 60-s cap) |
| `S_percentile_sensitivity.csv` | Supplementary Table S4B (band-averaged differences under alternative cleaning rules) |
| `S_quantile_regression.csv` | Supplementary Table S4C (quantile regression adjusted for age and sex; check) |
| `S_low_range_share.csv` | Supplementary Table S4D (share of records below the pre-pandemic 10th fitness percentile of the age band) |
| `S_sex_bmi_interactions.csv` | Supplementary Table S5C (period × sex and period × BMI interactions) |
| `S_vse_empirical_vs_published.csv` | Supplementary Table S7 (empirical and published percentiles) |
| `S_vse_gamlss_present.csv` | Supplementary Table S7 (GAMLSS centiles of the same records; from `gamlss_compare.R`) |
| `S_pre_repeat_link.csv` | Methods 2.4, Table 5 and Supplementary Tables S1/S3 (2017–2019 records identified as later measurements of the same child by linkage to the programme's repeat-measurement file; aggregate counts, overall and by year, and the interval between a child's first and second measurements in that file) |

The script also writes the remaining supplementary tables (missingness, measurement index by year, annual means, empirical percentiles, sex × BMI subgroups, BMI by sex and age) and Figures 1–3 and S1 to `output_v2/`.

## Environment

Python 3.12 — pandas 2.2, numpy 2.2, scipy 1.16, statsmodels 0.14, patsy, matplotlib 3.10; openpyxl for the optional repeat-measurement file (see `requirements.txt`).
R — package `gamlss` for `gamlss_compare.R` (GAMLSS centiles in Supplementary Table S7); package `sitar` (optional) only to regenerate the IOTF/WHO reference CSVs with `ref/export_lms.R`.

## Analysis specification (summary)

* Unit of analysis: measurement record (no child identifier available).
* Processing, identical in both periods: plausibility limits (SRT −30 to 50 cm; VSE ≤ 200 s; SLS ≤ 150 s; SLJ ≤ 200 cm; 5SR ≤ 50 s) → zero values set to missing (sit-and-reach: only in records with all five tests 0; 2017–2019 sit-and-reach zeros recovered from the earlier report's input file given by `PFN_PRIOR`) → 120-s ceiling for VSE and SLS → values > 3 SD from the mean for age in months (periods pooled) set to missing.
* Primary model: OLS of each test on period, sex and a cubic B-spline of age in months (4 df), HC3 standard errors; standardised difference = adjusted difference / pre-period SD, oriented so that positive = post-pandemic deficit (percentile and quantile-regression differences are reported in the same orientation).
* Distribution: P10–P90 computed directly within eight 6-month age bands for each period (sexes pooled; fitness percentiles, P10 = least fit, i.e. slowest for the 5SR); post–pre differences averaged across bands, weighted by records, with bootstrap 95 % CIs; quantile regression with the primary-model covariates as a complementary analysis.
* BMI: z-scores from the LMS parameters of the 2017 Korean National Growth Charts (primary; not overweight < 85th centile, including thinness; overweight 85th to < 95th; obese ≥ 95th centile; the script stops if this reference table is missing), IOTF and WHO 2006/2007 (sensitivity). Age in completed months is converted to the mid-month (age + 0.5 months) and the L, M and S values are linearly interpolated.
* Input checks: required columns, period code consistent with calendar year, sex coded 1/2; the script stops otherwise. Library deprecation notices are suppressed; other warnings are printed.
* Sensitivity analyses: BMI-complete subset with and without BMI z-score; identified repeat measurements excluded (post-pandemic index > 1; pre-pandemic repeats from the programme's repeat-measurement file — without `PFN_REPEAT` only post-pandemic repeats are excluded, so this column of Table 5 differs from the published one); no ceiling; no trimming; common recording precision; VSE and SLS capped at 60 s; illustrative clustering scenario; band-averaged percentile differences under each cleaning rule.

## Citation

Oh M-Y, Han Y-O, Yoo IH, Kim SH. Post-Pandemic Shifts in Physical Fitness Among Korean Preschool Children: Age-Specific and Distributional Evidence from a Population Surveillance Study. *Scandinavian Journal of Medicine & Science in Sports* (under revision). See `CITATION.cff`.

## License

MIT (see `LICENSE`). The reference tables in `ref/` are derived from public sources (KDCA; Cole & Lobstein 2012; WHO) and remain subject to their original terms.
