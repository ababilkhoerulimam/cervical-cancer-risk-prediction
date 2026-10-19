# Project State: Cervical Cancer Risk Prediction

**Schema version:** 1.1
**Last updated:** 2026-10-19
**Status:** Planning
**Current phase:** Problem formulation and data inventory
**Project type:** PREDICTIVE
**Decision owner:** User
**Primary notebook:** cervical-cancer-notebook.ipynb
**Last completed cell:** Cell 5
**Data snapshot:** cervical-cancer_csv.csv (94 KB, shape: 835x36, git 9badb2e)
**Code version:** 9badb2e

## Objective and Success Criteria

**Problem or decision:** Predict cervical cancer risk from patient demographic, behavioral, and clinical records.

**Unit of analysis:** Individual patient examination record.

**Outcome or target:** Biopsy (definitive histological diagnosis, binary 0/1).

**Primary success metric:** PR-AUC (Precision-Recall Area Under Curve / Average Precision). Secondary check: ROC-AUC.

**Constraints:** Severe positive class imbalance (~6.5%), medical screening context, zero data leakage across CV splits, honest missing value handling (? / NaN representations).

## Data Inventory

| Source | Metadata / Grain | Time Coverage | Sensitivity | Exploration Status |
|---|---|---|---|---|
| cervical-cancer_csv.csv | 94 KB, patient record grain | Unknown | Public / De-identified | NOT OPENED |

## Progress

| Stage | Status | Evidence / Cell | Notes |
|---|---|---|---|
| Planning | Completed | Cell 1 | Problem framing, target Biopsy, and PR-AUC metric locked |
| Data quality and preparation | Completed | Cell 2 | Dropped 3 diagnostic leakage columns + 2 sparse duration cols; 30 features retained |
| EDA or statistical analysis | Completed | Cell 3 | 2 zero-var cols dropped; 28 features remaining; EPV=1.93; top signals identified |
| Feature engineering | Not Started | None | Pending clean baseline |
| Modeling and validation | In Progress | Cell 5 | Family 1 (Linear) and Family 2 (Forests) benchmarked; ExtraTrees_Balanced_d3 leads |
| Explanation and error analysis | Not Started | None | Pending trained model artifacts |
| Methodology design | In Progress | Cell 5 | Ablation matrix active (Families 1 & 2 complete; Families 3, 4, 5 pending teammate) |
| Delivery and export QA | Not Started | None | Pending final artifact evaluation |
| Operational monitoring | Not Applicable | None | Offline analysis and research pipeline |

## Assumptions Register

| ID | Assumption | Category | Confidence | Impact if Wrong | Validation Plan | Status |
|---|---|---|---|---|---|---|
| A-001 | Target columns follow standard UCI cervical cancer risk indicators | DATA | HIGH | HIGH | Confirmed in Cell 1 (Biopsy binary 0/1 with 54 positive cases) | VALIDATED |
| A-002 | Diagnostic tests (Hinselmann, Schiller, Citology) must be excluded from lifestyle predictors | DATA | HIGH | HIGH | Excluded in Cell 2 to prevent clinical leakage | VALIDATED |
| A-003 | Full 28-feature model risks overfitting due to EPV=1.93 | STATISTICAL | HIGH | HIGH | Confirmed in Cell 4: L1 sparsity (14.8 feats) improved ROC-AUC from 0.572 to 0.625 | VALIDATED |

## Decisions Log

### D-001 — Initialize Project State Tracking

- **Chosen:** Track progress and evidence via PROJECT_STATE.md schema 1.1
- **Alternatives considered:** Ad-hoc unrecorded experimentation
- **Why:** Preserves reproducibility, auditability, and leakage-safe state tracking across sessions
- **Evidence:** User approval to create state tracker
- **Revisit when:** Not applicable

### D-002 — Target and Metric Selection

- **Chosen:** Target = `Biopsy`, Primary Metric = PR-AUC (Average Precision)
- **Alternatives considered:** Composite target (Biopsy | Schiller | Citology | Hinselmann), ROC-AUC
- **Why:** Biopsy is clinical gold standard; PR-AUC is sensitive to positive class precision/recall under severe ~6.5% imbalance without ROC-AUC true-negative inflation
- **Evidence:** Dataset diagnostic check (Biopsy 54 pos / 835 total = 6.47%) and user decision
- **Revisit when:** User requests multi-target or composite risk screening

### D-003 — Feature Triage and Leakage Shielding

- **Chosen:** Drop diagnostic tests (`Hinselmann`, `Schiller`, `Citology`) and sparse duration columns (`STDs: Time since first/last diagnosis`). Retain 30 lifestyle, demographic, and anamnesis features.
- **Alternatives considered:** Keep diagnostic tests as features (creates target leakage / unrealistic screening setting).
- **Why:** Objective is lifestyle-based risk prediction prior to invasive diagnostic examinations.
- **Evidence:** Observed 91.5% missingness on STD duration and diagnostic label correlation.
- **Revisit when:** Clinical setting changes to post-examination multi-modal triaging.

### D-004 — Validation Strategy

- **Chosen:** 5-fold StratifiedKFold (seed=42).
- **Alternatives considered:** Simple train/test split, unstratified K-fold.
- **Why:** Ensures stable representation of rare positive class (~10-11 positives per fold) without fold starvation.
- **Evidence:** Cell 2 fold distribution audit.
- **Revisit when:** Dataset size changes materially.

### D-005 — EPV-Constrained Model Architecture

- **Chosen:** Enforce strict regularization (L1/L2 penalties, shallow tree depth 2-3) to counter EPV=1.93 deficit.
- **Alternatives considered:** Deep unconstrained trees, unregularized neural networks.
- **Why:** EPV=1.93 violates standard biostatistical guideline (EPV>=10) and guarantees severe overfit if unregularized.
- **Evidence:** Cell 3 observed 54 events across 28 candidate predictors.
- **Revisit when:** Sample size increases or high-dimensional embeddings are considered.

### D-006 — Family 1 Linear Ablation Outcome

- **Chosen:** Retain L1 Lasso and ElasticNet configurations as sparse interpretable linear baselines.
- **Alternatives considered:** Unweighted unregularized logistic regression.
- **Why:** L1 Lasso reduced active features from 27.2 to 14.8 while lifting ROC-AUC to 0.6252, demonstrating effective noise culling on small sample size.
- **Evidence:** Cell 4 OOF benchmark table.
- **Revisit when:** Non-linear interactions are modeled in Family 2.

### D-007 — Family 2 Decision Forests Ablation Outcome

- **Chosen:** ExtraTrees with shallow depth 3 (`ExtraTrees_Balanced_d3`) establishes new benchmark champion (PR-AUC 0.1692 vs random 0.0647, ROC-AUC 0.6233).
- **Alternatives considered:** Deep unconstrained Random Forest (PR-AUC 0.1160 - 0.1341).
- **Why:** Extreme randomization at each split dramatically reduces model variance on small tabular data ($N=835$, EPV=1.93) without overfitting to individual samples.
- **Evidence:** Cell 5 OOF benchmark table.
- **Revisit when:** Family 3 (GBDT) and Family 4 (Kernel/SVM) models are benchmarked.

### D-008 — Data Harmonization Strategy (Option B)

- **Chosen:** Pursue Option B (Dual-Real Harmonization: UCI Caracas + CDC NHANES real patient datasets). Exclude purely synthetic generation from primary evaluation.
- **Alternatives considered:** Total naive merge of synthetic data (Option A), Pre-training prior transfer (Option C).
- **Why:** Guarantees 100% genuine biological data from actual human subjects; aligns features (`Age`, `Pregnancies`, `Smokes`, `Oral_Contraceptives`, `First_Sex_Age`, `Partners`) while maintaining honest clinical ground-truth.
- **Evidence:** Documented in `sumber.md` with complete URLs, cycle identifiers, and feature alignment mapping.
- **Revisit when:** Cross-domain validation results are analyzed.

## Evidence Ledger

| ID | Claim or Result | Value | Evidence Source | Evaluation Context | Status |
|---|---|---|---|---|---|
| E-001 | Target Biopsy distribution and shape | shape=(835, 36), pos=54 (6.47%), neg=781 (93.53%) | Cell 1 | DESCRIPTIVE | OBSERVED |
| E-002 | Missingness profile in raw data | 26/36 cols with nulls; top missing: STDs: Time since first/last dx (91.5%), IUD (13.4%) | Cell 1 | DESCRIPTIVE | OBSERVED |
| E-003 | StratifiedKFold partition balance | 5 folds, 668 train / 167 val, 10-11 pos per val fold | Cell 2 | CV | OBSERVED |
| E-004 | Zero-variance feature elimination | Dropped STDs:cervical condylomatosis and STDs:AIDS; remaining features=28 | Cell 3 | DESCRIPTIVE | OBSERVED |
| E-005 | Events Per Variable (EPV) diagnostic | 54 positive events / 28 features = 1.93 | Cell 3 | STATISTICAL | OBSERVED |
| E-006 | Top univariate risk signals | Dx:HPV (0.1621), Dx:Cancer (0.1621), Dx (0.1588), STDs:genital herpes (0.1338) | Cell 3 | DESCRIPTIVE | OBSERVED |
| E-007 | Family 1 Linear Ablation OOF | LR_L2_Unw: PR 0.1234, ROC 0.5922; LR_L1_Bal: PR 0.1187, ROC 0.6252 (14.8 feats); LR_EN_Bal: PR 0.1193, ROC 0.6190 | Cell 4 | CV | OBSERVED |
| E-008 | Family 2 Decision Forests Ablation OOF | RF_Unw_Deep: PR 0.1160, ROC 0.6183; RF_Bal_Deep: PR 0.1341, ROC 0.6013; RF_Bal_d3: PR 0.1154, ROC 0.5932; ExtraTrees_Bal_d3: PR 0.1692, ROC 0.6233 | Cell 5 | CV | OBSERVED |

## Methodology Design

**Method ID and version:** M-001 v0.1 (Clinical Risk Research Pipeline)

**Maturity:** PROPOSAL

**Method spec:** In-notebook research protocol guided by corpus cards

**Case signature and baseline gap:** Tabular clinical dataset (N=835, D=30 lifestyle features), severe class imbalance (positive rate 6.47%), missingness up to 13.4% in lifestyle variables. Standard naive models with threshold 0.5 fail to detect rare positive cases.

**Exact next experiment:** Execute EDA & missingness pattern analysis (Cell 3) followed by leakage-safe Multi-Family Model Grand Prix (Linear, Bagging, Boosting) under 5-Fold Stratified OOF.

## Model Drivers and Error Analysis

NOT APPLICABLE (No models trained yet).

## Artifacts

| Artifact | Purpose | Validation Status |
|---|---|---|
| cervical-cancer_csv.csv | Raw tabular input data | PASSED |
| cervical-cancer-notebook.ipynb | Primary notebook with Cells 1-5 | PASSED |
| sumber.md | Comprehensive data source inventory and harmonization map | PASSED |

## Operational Monitoring

**Status:** NOT APPLICABLE

**Owner and cadence:** NOT APPLICABLE

**Signals and intervention thresholds:** NOT APPLICABLE

**Fallback or rollback:** NOT APPLICABLE

## Limitations and Risks

- Missingness ekstrem (>90%) pada fitur `STDs: Time since first/last diagnosis` (764/835 null).
- Imbalance rasio 1:14 (54 positif / 781 negatif).

## Open Questions

- [ ] Fitur diagnosa medis (`Hinselmann`, `Schiller`, `Citology`) dikeluarkan agar murni faktor risiko gaya hidup / anamnesis?
- [ ] Fitur riwayat medis (`Dx:Cancer`, `Dx:CIN`, `Dx:HPV`, `Dx`) dipertahankan sebagai riwayat medis atau dikeluarkan?

## Exact Next Action

Handover ke rekan tim: implementasi Cell 6 untuk Keluarga 3 (GBDT: HistGradientBoosting, LightGBM, CatBoost) dan Keluarga 4/5 (Kernel SVM RBF, TabPFN/MLP) di bawah protokol 5-Fold Stratified OOF yang sama.
