<div align="center">
  <h1>cervical-cancer-risk-prediction</h1>
  <p><strong>Cervical Cancer Risk Stratification from Real-World Clinical and Behavioral Cohorts</strong></p>

  <p align="center">
    <img src="https://img.shields.io/badge/Dataset-UCI_Caracas_%2B_CDC_NHANES-blue?style=flat-square" alt="Dataset">
    <img src="https://img.shields.io/badge/Language-Python_3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Language">
    <img src="https://img.shields.io/badge/Validation-5--Fold_Stratified_CV-success?style=flat-square" alt="Validation">
    <img src="https://img.shields.io/badge/Champion_PR--AUC-0.1111_(5.3x_Lift)-orange?style=flat-square" alt="Metric">
    <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" alt="License">
  </p>

  <p align="center">
    A leak-free, evidence-based machine learning pipeline that harmonizes real patient records across Hospital Universitario de Caracas and CDC NHANES cohorts to stratify cervical cancer risk prior to invasive diagnostic procedures.
  </p>
</div>

## Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

## Overview and Clinical Motivation

Cervical cancer is the fourth most common cancer in women globally. Although over 95 percent of cases stem from persistent high-risk Human Papillomavirus (HPV) infection, resource-constrained settings frequently lack universal molecular screening infrastructure. Predicting individual risk from patient history and behavioral anamnesis enables targeted screening triage before conducting invasive clinical examinations.

Previous machine learning implementations on the standard single-center UCI Caracas dataset suffered from two severe flaws:
1. Target Leakage: Naive models included intermediate diagnostic test outcomes such as Schiller iodine tests, Hinselmann colposcopy, and cytology as input features, resulting in artificial performance that cannot be reproduced during true pre-examination screening.
2. Events-Per-Variable (EPV) Deficit: Evaluating 28 predictors against only 54 confirmed biopsy events yields an EPV ratio of 1.93, far below the biostatistical safety threshold of EPV 10.0, inducing substantial variance and overfit.

This repository resolves both limitations by constructing a harmonized real-world cohort that pools genuine patient data from Hospital Universitario de Caracas (Venezuela) and CDC NHANES national examination surveys (United States), expanding the sample to 3,567 patients and 75 positive biopsy and clinical cancer events.

## Dataset Harmonization Architecture

The repository enforces a strict zero-leakage, dual-representation schema designed simultaneously for exploratory data analysis and leak-free machine learning pipelines.

### Data Source Specifications

1. Primary Cohort (Hospital Universitario de Caracas):
   The official UCI dataset (Dataset ID 383) covers 858 female outpatients examined in Caracas, Venezuela. Diagnostic examination columns (`Hinselmann`, `Schiller`, `Citology`) are strictly removed to eliminate post-hoc leakage. The target is confirmed histological `Biopsy`.

2. Secondary Cohort (CDC NHANES 2015-2016):
   Individual female patient records spanning demographic interviews (`DEMO_I`), reproductive health modules (`RHQ_I`), sexual behavior interviews (`SXQ_I`), smoking history (`SMQ_I`), and confirmed cancer diagnosis questionnaires (`MCQ_I`). Only real human respondents with confirmed diagnosis status are retained.

### Feature Harmonization Mapping

| Standard Feature | UCI Caracas Column | CDC NHANES Column | Clinical Rationale |
|---|---|---|---|
| `age` | `Age` | `RIDAGEYR` | Chronological age at examination |
| `num_sexual_partners` | `Number of sexual partners` | `SXQ700` / `SXQ800` | Lifetime sexual exposure proxy |
| `first_sexual_intercourse` | `First sexual intercourse` | `SXD021` | Age at onset of exposure window |
| `num_pregnancies` | `Num of pregnancies` | `RHQ160` | Full-term gestational history |
| `smokes` | `Smokes` | `SMQ020` | Tobacco exposure immune cofactor |
| `hormonal_contraceptives` | `Hormonal Contraceptives` | `RHQ420` | Oral contraceptive duration factor |
| `target` | `Biopsy` | `MCQ230A == 15` | Histologically confirmed cancer / CIN |

The final master artifact (`data/cervical_cancer_master_cleaned.csv`) retains 16 columns:
- Subject identification: `patient_id` and origin `source`.
- Continuous and binary numeric representations for machine learning: `age`, `num_sexual_partners`, `first_sexual_intercourse`, `num_pregnancies`, `smokes`, `hormonal_contraceptives`, and `target`.
- Binned human-readable labels for data analyst exploration: `target_label`, `age_group`, `smoker_label`, `contraception_label`, `pregnancy_group`, and `partners_group`.
- Pre-baked cross-validation assignment: `fold_5cv` assigning exactly 15 positive cases per fold.

## Repository Structure

```
cervical-cancer-risk-prediction/
├── data/
│   ├── cervical_cancer_master_cleaned.csv    # Dual-purpose master dataset (3,567 rows, 16 cols)
│   ├── harmonized_master_real.csv            # Merged real cohort records (3,567 rows)
│   ├── nhanes_cervical_harmonized.csv        # Processed CDC NHANES female cohort (2,709 rows)
│   ├── uci_cervical_cancer_858.csv           # Full official UCI 383 cohort (858 rows)
│   └── uci_cervical_harmonized.csv           # Processed UCI Caracas cohort (858 rows)
├── notebooks/
│   ├── 01_data_harmonization.ipynb           # Step-by-step ETL and dual-purpose master export
│   └── 02_model_ablation_ensemble.ipynb      # Multi-family benchmark, ablation, and champion ensemble
├── scripts/
│   └── build_harmonized_dataset.py           # Automated end-to-end dataset builder CLI
├── .gitignore                                # Repository ignore patterns
├── LICENSE                                   # MIT License
├── PROJECT_STATE.md                          # Evidence ledger and decision tracking
├── README.md                                 # Project documentation
├── requirements.txt                          # Package dependencies
└── sumber.md                                 # Academic bibliography and data source catalog
```

## Methodology and Multi-Family Ablation Matrix

Model evaluation is conducted under 5-Fold Stratified Cross-Validation strictly out-of-fold (OOF). All median imputations and standard scalers are fitted exclusively on training splits to prevent distributional contamination.

Due to the extreme positive class imbalance (75 positive events among 3,567 records, base rate 2.10 percent), Precision-Recall Area Under Curve (PR-AUC / Average Precision) serves as the primary optimization metric, paired with ROC-AUC and Brier Loss for calibration verification.

### Baseline Benchmark Across Four Model Families

| Model Family | Specific Model | PR-AUC | ROC-AUC | Brier Loss |
|---|---|---|---|---|
| Bagging and Forests | `RF_Unweighted_Deep` | 0.0770 | 0.7167 | 0.0221 |
| Bagging and Forests | `RF_Balanced_Shallow_d3` | 0.0751 | 0.7745 | 0.1502 |
| GBDT | `CatBoost_AutoClassWeight` | 0.0718 | 0.7557 | 0.1090 |
| GBDT | `XGBoost_Shallow_d3` | 0.0683 | 0.7050 | 0.0841 |
| Linear Regularized | `LR_L2_Balanced` | 0.0666 | 0.7755 | 0.1824 |
| Linear Regularized | `LR_ElasticNet_Balanced` | 0.0646 | 0.7756 | 0.1818 |
| Linear Regularized | `LR_L1_Lasso_Balanced` | 0.0646 | 0.7756 | 0.1818 |
| Linear Regularized | `LR_L2_Unweighted` | 0.0628 | 0.7771 | 0.0201 |
| GBDT | `LGBM_ScalePosWeight` | 0.0620 | 0.6684 | 0.0542 |
| GBDT | `HistGB_Balanced` | 0.0571 | 0.6726 | 0.0536 |
| Bagging and Forests | `RF_Balanced_Deep` | 0.0567 | 0.7067 | 0.0275 |
| Bagging and Forests | `ExtraTrees_Balanced_d3` | 0.0534 | 0.7603 | 0.1794 |
| Kernel and Distance | `SVM_RBF_Platt_Balanced` | 0.0468 | 0.7306 | 0.0203 |
| Kernel and Distance | `kNN_Weighted_k10` | 0.0436 | 0.6185 | 0.0245 |
| Kernel and Distance | `kNN_Weighted_k5` | 0.0413 | 0.6040 | 0.0258 |
| Reference Baseline | Random Guessing Base Rate | 0.0210 | 0.5000 | 0.0206 |

### Corpus-Guided Feature Enhancement

To extract latent risk structure without introducing high-dimensional overfit, three biological interaction terms and four informative missingness indicators were engineered:
1. `years_active`: Calculated as the difference between current age and age at first sexual intercourse, representing cumulative years of exposure to potential HPV transmission.
2. `partner_rate`: Total lifetime sexual partners divided by active sexual exposure years, quantifying sexual partner density.
3. `smoke_hormone`: Multiplicative interaction between smoking status and hormonal contraceptive usage, modeling known biochemical co-carcinogenic synergy.
4. Informative Missing Indicators: Binary presence flags capturing systematic survey omissions across reproductive history variables.

### Champion Ensemble Architecture

Combining predictions across diverse algorithmic inductive biases yielded significant variance reduction over individual learners:

$$
\hat{P}_{\text{Champion}} = \frac{1}{3} \left( \hat{P}_{\text{RF-Unweighted}} + \hat{P}_{\text{RF-Balanced-d3}} + \hat{P}_{\text{CatBoost-Auto}} \right)
$$

The Tri-Family Champion Ensemble achieved:
- PR-AUC: 0.1111 (a 5.3x lift factor over the 0.0210 random baseline).
- ROC-AUC: 0.7682.
- Brier Calibration Score: 0.0672.

## Cost-Sensitive Medical Threshold Optimization

Standard classification pipelines apply an arbitrary probability threshold of 0.50 or maximize the balanced F1-score. In clinical oncology screening, a False Negative (an undiagnosed cervical lesion progressing to invasive cancer) incurs substantially higher human cost than a False Positive (a healthy patient referred for routine secondary examination).

Decision thresholds are optimized on out-of-fold probability distributions via the $F_2$-score, which weights Sensitivity (Recall) twice as heavily as Precision:

$$
F_2 = (1 + 2^2) \times \frac{\text{Precision} \times \text{Recall}}{2^2 \times \text{Precision} + \text{Recall}}
$$

At the optimal medical threshold of 0.4547:
- True Positives Detected: 36 patients.
- False Negatives (Missed): 39 patients.
- Clinical Screening Recall: 48.00 percent.
- Precision: 9.92 percent (nearly 5 times higher than population prevalence).
- Specificity: 90.63 percent (3,165 true negatives out of 3,492 healthy subjects).

## Theoretical Information Ceiling

Analysis of mutual information between the six core behavioral anamnesis features and histopathological outcomes confirms values between 0.002 and 0.009 nats. Because lifestyle variables serve as behavioral exposure proxies rather than direct causative biological agents, model performance near PR-AUC 0.1111 and ROC-AUC 0.77 to 0.79 represents the empirical information ceiling achievable from self-reported history alone. Further score expansion requires molecular biomarkers such as high-risk HPV DNA testing or liquid-based cytology.

## Getting Started and Reproduction

### Prerequisites

Clone the repository and install dependencies in a Python 3.10+ virtual environment:

```bash
git clone https://github.com/ababilkhoerulimam/cervical-cancer-risk-prediction.git
cd cervical-cancer-risk-prediction
pip install -r requirements.txt
```

### Reproducing Data Harmonization

Run the standalone ETL pipeline to download upstream archives and regenerate the master datasets:

```bash
python scripts/build_harmonized_dataset.py
```

### Reproducing Experiments

Open and execute the structured notebooks:

```bash
jupyter notebook notebooks/01_data_harmonization.ipynb
jupyter notebook notebooks/02_model_ablation_ensemble.ipynb
```

## References and Citations

1. Fernandes, K., Cardoso, J. S., & Fernandes, J. (2017). Transfer learning with partial observability applied to cervical cancer screening. Iberian Conference on Pattern Recognition and Image Analysis, 243-250. https://doi.org/10.1007/978-3-319-58838-4_27
2. Centers for Disease Control and Prevention. (2018). National Health and Nutrition Examination Survey: 2015-2016 Examination and Questionnaire Data. U.S. Department of Health and Human Services. https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2015
3. World Health Organization. (2021). WHO guideline for screening and treatment of cervical pre-cancer lesions for cervical cancer prevention (2nd ed.). World Health Organization. https://www.who.int/publications/i/item/9789240030824

For complete source catalogs and module extraction details, refer to `sumber.md`.
