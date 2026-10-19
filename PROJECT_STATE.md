# Project State: Cervical Cancer Risk Harmonized Master Dataset

**Schema version:** 1.1
**Last updated:** 2026-10-19
**Status:** Completed
**Current phase:** Harmonized Master Dataset Exported (Option 3 Complete)
**Project type:** PREDICTIVE
**Decision owner:** User
**Primary notebook:** notebooks/01_data_harmonization.ipynb
**Last completed cell:** Cell 3
**Data snapshot:** data/harmonized_master_real.csv (3,567 rows, 8 cols)
**Code version:** b410b97

## Objective and Success Criteria

**Problem or decision:** Construct, validate, and export `data/cervical_cancer_master_cleaned.csv` as a dual-purpose master dataset (ready for Data Analyst exploration and Machine Learning modeling with pre-baked 5-fold CV). Scope concludes upon master dataset export.

**Unit of analysis:** Individual real patient examination record.

**Outcome or target:** `target` (0 = Negative / Healthy, 1 = Positive confirmed cervical cancer or CIN histology/diagnosis).

**Primary success metric:** PR-AUC (Precision-Recall Area Under Curve / Average Precision) evaluated under pre-baked 5-Fold Stratified Cross-Validation.

**Constraints:** 100% genuine biological data from actual human patients (Option B); dual representation (descriptive labels for DA, numeric codes for MLE); zero target leakage across folds; pre-baked fixed CV folds.

## Data Inventory

| Source | Metadata / Grain | Time Coverage | Sensitivity | Exploration Status |
|---|---|---|---|---|
| data/uci_cervical_cancer_858.csv | 100 KB, 858 rows, 36 cols, patient record grain | 2017 (Caracas) | Public / De-identified | ACQUIRED |
| data/nhanes_cervical_harmonized.csv | 101 KB, 2,709 rows, 8 cols, adult female survey/lab grain | 2015-2016 (US) | Public / De-identified | ACQUIRED |
| data/harmonized_master_real.csv | 136 KB, 3,567 rows, 8 cols, unified real cohort grain | Multi-center | Public / De-identified | NOT OPENED |

## Progress

| Stage | Status | Evidence / Cell | Notes |
|---|---|---|---|
| Planning | Completed | Initial setup | Option 3 dual-purpose master dataset locked |
| Data quality and preparation | Completed | Cell 2 | Dual-representation schema constructed; fold_5cv assigned |
| EDA or statistical analysis | Not Started | None | Pending downstream DA exploration |
| Feature engineering | Completed | Cell 2 | DA human-readable binned categories + MLE numeric mapping |
| Modeling and validation | Completed | Cell 2 | Pre-baked Stratified 5-Fold split locked (15 pos/fold) |
| Explanation and error analysis | Not Started | None | Pending downstream modeling |
| Methodology design | Completed | Cell 3 | Option 3 harmonization protocol executed |
| Delivery and export QA | Completed | Cell 3 | Exported and verified cervical_cancer_master_cleaned.csv |
| Operational monitoring | Not Applicable | None | Offline analysis and research pipeline |

## Assumptions Register

| ID | Assumption | Category | Confidence | Impact if Wrong | Validation Plan | Status |
|---|---|---|---|---|---|---|
| A-001 | Harmonized lifestyle features have compatible physiological ranges across UCI and NHANES | DATA | HIGH | HIGH | Verify distributions, min/max, and null rates in Cell 1 | OPEN |
| A-002 | Pre-baked Stratified 5-Fold split preserves positive class ratio (~2.1% overall) equally across all folds | STATISTICAL | HIGH | HIGH | Verify fold counts in Cell 2 | OPEN |

## Decisions Log

### D-001 — Reset to Harmonized Master Dataset (Option 3)

- **Chosen:** Clean slate interactive workflow constructing `cervical_cancer_master_cleaned.csv` for both Data Analysts and Machine Learning Engineers.
- **Alternatives considered:** Continuing single-center UCI Caracas 835-row file with severe EPV deficit.
- **Why:** Scales genuine patient sample from 835 to 3,567 (75 positives), eliminates synthetic data artifacts, and synchronizes DA and MLE workflows with pre-baked 5-fold CV.
- **Evidence:** User approval for Option 3 interactive execution.
- **Revisit when:** Not applicable

### D-002 — Corpus-Guided Feature Enhancement & Tri-Family Ensemble

- **Chosen:** Incorporate domain exposure interactions (`years_active`, `partner_rate`, `smoke_hormone`), missing indicator flags (`_isna`), and Tri-Family Ensemble (`RF_Unw_FE` + `RF_Bal_d3_FE` + `CatBoost_Auto_FE`) with F2 cost-sensitive medical thresholding.
- **Alternatives considered:** Naive 2-model blend without feature engineering (PR-AUC 0.1048).
- **Why:** Domain interactions capture biological HPV exposure duration and smoking/hormone synergy; missing flags prevent information loss from structured survey missingness; tri-family blending combines bagging, shallow tree, and symmetric GBDT to achieve new peak PR-AUC 0.1111.
- **Evidence:** Cells 13-15 in notebooks/02_model_ablation_ensemble.ipynb.
- **Revisit when:** External clinical validation cohort becomes available.

## Evidence Ledger

| ID | Claim or Result | Value | Evidence Source | Evaluation Context | Status |
|---|---|---|---|---|---|
| E-001 | Harmonized real cohort inventory | 3,567 patients (858 UCI Caracas, 2,709 CDC NHANES), 75 positive cases | data/harmonized_master_real.csv | DESCRIPTIVE | OBSERVED |
| E-002 | Harmonized raw profile audit | total_rows=3567, UCI pos=55 (6.41%), NHANES pos=20 (0.74%), top missing: pregnancies (850), partners (791) | Cell 1 | DESCRIPTIVE | OBSERVED |
| E-003 | Pre-baked Stratified 5-Fold balance | 5 folds, exactly 15 positives per fold (~713 samples/fold) | Cell 2 | CV | OBSERVED |
| E-004 | Master Cleaned QA export | 3,567 rows, 16 cols (6 DA, 8 MLE, 2 ID/source), target=75, file=319 KB | Cell 3 | EXPORT_QA | PASSED |
| E-005 | Enhanced Tri-Family Champion Ensemble OOF | RF_Unw_FE + RF_Bal_d3_FE + CatBoost_Auto_FE: PR-AUC 0.1111, ROC-AUC 0.7682, F2-Threshold 0.4547 (Recall 48.0%) | notebooks/02_model_ablation_ensemble.ipynb | CV | OBSERVED |

## Methodology Design

**Method ID and version:** M-002 v0.1 (Harmonized Master Clinical Risk Dataset)

**Maturity:** PROPOSAL

**Method spec:** In-notebook interactive pipeline constructing dual-purpose master dataset

**Case signature and baseline gap:** Small single-center clinical dataset (N=835, EPV=1.93) expanded into multi-cohort real master (N=3,567, 75 positive cases). Requires unified schema, human-readable labels for DA, and pre-baked stratified CV folds for MLE.

**Exact next experiment:** Cell 1: Load `data/harmonized_master_real.csv`, audit raw schema and missingness across both sources (UCI vs NHANES).

## Model Drivers and Error Analysis

NOT APPLICABLE (Dataset preparation phase).

## Artifacts

| Artifact | Purpose | Validation Status |
|---|---|---|
| data/uci_cervical_cancer_858.csv | Full official UCI 383 dataset (858 rows) | PASSED |
| data/nhanes_cervical_harmonized.csv | Standardized CDC NHANES real cohort (2,709 rows) | PASSED |
| data/uci_cervical_harmonized.csv | Standardized UCI Caracas cohort (858 rows) | PASSED |
| data/harmonized_master_real.csv | Master merged real dataset (3,567 rows, 75 positives) | PASSED |
| data/cervical_cancer_master_cleaned.csv | Dual-purpose master dataset (DA & MLE ready, 3,567 rows, 16 cols) | PASSED |
| notebooks/01_data_harmonization.ipynb | Data harmonization notebook (Cells 1-3) | PASSED |
| notebooks/02_model_ablation_ensemble.ipynb | Multi-family ablation & champion ensemble notebook (Cells 1-15) | PASSED |
| scripts/build_harmonized_dataset.py | Reproducible download and ETL script | PASSED |
| requirements.txt | Python package dependencies specification | PASSED |
| sumber.md | Comprehensive data source inventory and harmonization map | PASSED |

## Operational Monitoring

**Status:** NOT APPLICABLE

**Owner and cadence:** NOT APPLICABLE

**Signals and intervention thresholds:** NOT APPLICABLE

**Fallback or rollback:** NOT APPLICABLE

## Limitations and Risks

- Cross-cohort demographic differences between Venezuela hospital outpatients and US national survey respondents.
- Missingness exists in specific behavioral questions (e.g. lifetime partners, age at first intercourse).

## Open Questions

- None (scoped objective achieved).

## Exact Next Action

Selesai. Dataset `data/cervical_cancer_master_cleaned.csv` siap digunakan oleh Data Analyst untuk eksplorasi visual dan Machine Learning Engineer untuk pemodelan pada partisi `fold_5cv`.
