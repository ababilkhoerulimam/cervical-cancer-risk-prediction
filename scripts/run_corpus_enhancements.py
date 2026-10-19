import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve, confusion_matrix
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

SEED = 42
nb_path = Path("C:/Github/cervical-cancer-risk-prediction/cervical-cancer-notebook2.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

df_raw = pd.read_csv("C:/Github/cervical-cancer-risk-prediction/data/cervical_cancer_master_cleaned.csv")
y = df_raw["target"].astype(int)

# Cell 13 execution
df_fe = df_raw.copy()
df_fe["years_active"] = np.maximum(df_fe["age"] - df_fe["first_sexual_intercourse"], 0)
df_fe["partner_rate"] = df_fe["num_sexual_partners"] / np.maximum(df_fe["years_active"], 1.0)
df_fe["smoke_hormone"] = df_fe["smokes"] * df_fe["hormonal_contraceptives"]

missing_indicator_cols = ["num_pregnancies", "num_sexual_partners", "first_sexual_intercourse", "hormonal_contraceptives"]
for c in missing_indicator_cols:
    df_fe[f"{c}_isna"] = df_fe[c].isnull().astype(float)

drop_fe_cols = [
    "patient_id", "target", "Biopsy", "target_label", "smoker_label", 
    "contraception_label", "pregnancy_group", "partners_group", "fold_5cv", "age_group"
]
X_fe = pd.get_dummies(df_fe.drop(columns=[c for c in drop_fe_cols if c in df_fe.columns]), drop_first=True)

out13_text = f"""original_features=10 | enhanced_features={X_fe.shape[1]}
added_interactions=['years_active', 'partner_rate', 'smoke_hormone']
added_missing_indicators={[f'{c}_isna' for c in missing_indicator_cols]}
"""

code13 = """# Cell 13 - Feature Engineering: Domain Interactions and Missing Indicators
import numpy as np
import pandas as pd

# ponytail: domain exposure interactions + informative missing indicators inspired by corpus cards
df_fe = df_raw.copy()
df_fe["years_active"] = np.maximum(df_fe["age"] - df_fe["first_sexual_intercourse"], 0)
df_fe["partner_rate"] = df_fe["num_sexual_partners"] / np.maximum(df_fe["years_active"], 1.0)
df_fe["smoke_hormone"] = df_fe["smokes"] * df_fe["hormonal_contraceptives"]

missing_indicator_cols = ["num_pregnancies", "num_sexual_partners", "first_sexual_intercourse", "hormonal_contraceptives"]
for c in missing_indicator_cols:
    df_fe[f"{c}_isna"] = df_fe[c].isnull().astype(float)

drop_fe_cols = [
    "patient_id", "target", "Biopsy", "target_label", "smoker_label", 
    "contraception_label", "pregnancy_group", "partners_group", "fold_5cv", "age_group"
]
X_fe = pd.get_dummies(df_fe.drop(columns=[c for c in drop_fe_cols if c in df_fe.columns]), drop_first=True)

print(f"original_features=10 | enhanced_features={X_fe.shape[1]}")
print(f"added_interactions=['years_active', 'partner_rate', 'smoke_hormone']")
print(f"added_missing_indicators={[f'{c}_isna' for c in missing_indicator_cols]}")"""

# Cell 14 execution
folds = []
for f_id in sorted(df_raw["fold_5cv"].unique()):
    tr_idx = df_raw[df_raw["fold_5cv"] != f_id].index.values
    va_idx = df_raw[df_raw["fold_5cv"] == f_id].index.values
    folds.append((tr_idx, va_idx))

fe_models = {
    "RF_Unw_FE": lambda: RandomForestClassifier(n_estimators=100, max_depth=None, random_state=SEED),
    "RF_Bal_d3_FE": lambda: RandomForestClassifier(n_estimators=100, max_depth=3, class_weight="balanced", random_state=SEED),
    "CatBoost_Auto_FE": lambda: CatBoostClassifier(iterations=100, auto_class_weights="Balanced", random_seed=SEED, verbose=0),
    "XGB_d3_FE": lambda: XGBClassifier(n_estimators=100, max_depth=3, scale_pos_weight=(y==0).sum()/(y==1).sum(), random_state=SEED, eval_metric="logloss"),
    "HistGB_Bal_FE": lambda: HistGradientBoostingClassifier(max_iter=100, class_weight="balanced", random_state=SEED),
    "LR_EN_Bal_FE": lambda: LogisticRegression(solver="saga", l1_ratio=0.5, C=0.1, class_weight="balanced", random_state=SEED, max_iter=2000),
}

fe_oof_preds = {m: np.zeros(len(y)) for m in fe_models}

for fold, (tr_idx, va_idx) in enumerate(folds):
    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X_fe.iloc[tr_idx])
    X_va = imputer.transform(X_fe.iloc[va_idx])

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)

    for name, m_fn in fe_models.items():
        clf = m_fn()
        if "LR_" in name:
            clf.fit(X_tr_sc, y.iloc[tr_idx])
            fe_oof_preds[name][va_idx] = clf.predict_proba(X_va_sc)[:, 1]
        else:
            clf.fit(X_tr, y.iloc[tr_idx])
            fe_oof_preds[name][va_idx] = clf.predict_proba(X_va)[:, 1]

lines14 = ["enhanced_model_benchmark:"]
for name, preds in fe_oof_preds.items():
    pr = average_precision_score(y, preds)
    roc = roc_auc_score(y, preds)
    brier = brier_score_loss(y, preds)
    lines14.append(f"model={name:18s} | pr_auc={pr:.4f} | roc_auc={roc:.4f} | brier={brier:.4f}")
out14_text = "\n".join(lines14) + "\n"

code14 = """# Cell 14 - Multi-Family Benchmark with Enhanced Feature Space
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

fe_models = {
    "RF_Unw_FE": lambda: RandomForestClassifier(n_estimators=100, max_depth=None, random_state=SEED),
    "RF_Bal_d3_FE": lambda: RandomForestClassifier(n_estimators=100, max_depth=3, class_weight="balanced", random_state=SEED),
    "CatBoost_Auto_FE": lambda: CatBoostClassifier(iterations=100, auto_class_weights="Balanced", random_seed=SEED, verbose=0),
    "XGB_d3_FE": lambda: XGBClassifier(n_estimators=100, max_depth=3, scale_pos_weight=(y==0).sum()/(y==1).sum(), random_state=SEED, eval_metric="logloss"),
    "HistGB_Bal_FE": lambda: HistGradientBoostingClassifier(max_iter=100, class_weight="balanced", random_state=SEED),
    "LR_EN_Bal_FE": lambda: LogisticRegression(solver="saga", l1_ratio=0.5, C=0.1, class_weight="balanced", random_state=SEED, max_iter=2000),
}

fe_oof_preds = {m: np.zeros(len(y)) for m in fe_models}

for fold, (tr_idx, va_idx) in enumerate(folds):
    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X_fe.iloc[tr_idx])
    X_va = imputer.transform(X_fe.iloc[va_idx])

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)

    for name, m_fn in fe_models.items():
        clf = m_fn()
        if "LR_" in name:
            clf.fit(X_tr_sc, y.iloc[tr_idx])
            fe_oof_preds[name][va_idx] = clf.predict_proba(X_va_sc)[:, 1]
        else:
            clf.fit(X_tr, y.iloc[tr_idx])
            fe_oof_preds[name][va_idx] = clf.predict_proba(X_va)[:, 1]

print("enhanced_model_benchmark:")
for name, preds in fe_oof_preds.items():
    pr = average_precision_score(y, preds)
    roc = roc_auc_score(y, preds)
    brier = brier_score_loss(y, preds)
    print(f"model={name:18s} | pr_auc={pr:.4f} | roc_auc={roc:.4f} | brier={brier:.4f}")"""

# Cell 15 execution
champion_blend_preds = (fe_oof_preds["RF_Unw_FE"] + fe_oof_preds["RF_Bal_d3_FE"] + fe_oof_preds["CatBoost_Auto_FE"]) / 3.0
champ_prauc = average_precision_score(y, champion_blend_preds)
champ_rocauc = roc_auc_score(y, champion_blend_preds)
champ_brier = brier_score_loss(y, champion_blend_preds)

precisions, recalls, thresholds = precision_recall_curve(y, champion_blend_preds)
beta = 2.0
f2_scores = (1 + beta**2) * (precisions[:-1] * recalls[:-1]) / ((beta**2 * precisions[:-1]) + recalls[:-1] + 1e-8)
best_f2_idx = np.nanargmax(f2_scores)
opt_thresh_f2 = thresholds[best_f2_idx]

f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
best_f1_idx = np.nanargmax(f1_scores)
opt_thresh_f1 = thresholds[best_f1_idx]

y_pred_medical = (champion_blend_preds >= opt_thresh_f2).astype(int)
cm_medical = confusion_matrix(y, y_pred_medical)
tp = int(cm_medical[1, 1])
fn = int(cm_medical[1, 0])
fp = int(cm_medical[0, 1])
tn = int(cm_medical[0, 0])

med_recall = tp / (tp + fn)
med_precision = tp / (tp + fp)

out15_text = f"""champion_ensemble=RF_Unw_FE + RF_Bal_d3_FE + CatBoost_Auto_FE
pr_auc={champ_prauc:.4f} | roc_auc={champ_rocauc:.4f} | brier={champ_brier:.4f}
standard_f1_thresh={opt_thresh_f1:.4f} | max_f1={f1_scores[best_f1_idx]:.4f}
medical_f2_thresh={opt_thresh_f2:.4f} | max_f2={f2_scores[best_f2_idx]:.4f}
medical_eval_at_f2_thresh:
tp={tp} | fn={fn} | fp={fp} | tn={tn}
sensitivity_recall={med_recall:.4f} | precision={med_precision:.4f}
"""

code15 = """# Cell 15 - Champion Ensemble & Cost-Sensitive Medical Threshold Optimization
import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix

# ponytail: 3-model tri-family blend incorporates bagging, shallow balanced tree, and symmetric GBDT
champion_blend_preds = (fe_oof_preds["RF_Unw_FE"] + fe_oof_preds["RF_Bal_d3_FE"] + fe_oof_preds["CatBoost_Auto_FE"]) / 3.0
champ_prauc = average_precision_score(y, champion_blend_preds)
champ_rocauc = roc_auc_score(y, champion_blend_preds)
champ_brier = brier_score_loss(y, champion_blend_preds)

print(f"champion_ensemble=RF_Unw_FE + RF_Bal_d3_FE + CatBoost_Auto_FE")
print(f"pr_auc={champ_prauc:.4f} | roc_auc={champ_rocauc:.4f} | brier={champ_brier:.4f}")

# Cost-Sensitive Medical Threshold Calibration (F2-score: prioritize recall to minimize fatal false negatives)
precisions, recalls, thresholds = precision_recall_curve(y, champion_blend_preds)
beta = 2.0
f2_scores = (1 + beta**2) * (precisions[:-1] * recalls[:-1]) / ((beta**2 * precisions[:-1]) + recalls[:-1] + 1e-8)
best_f2_idx = np.nanargmax(f2_scores)
opt_thresh_f2 = thresholds[best_f2_idx]

f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
best_f1_idx = np.nanargmax(f1_scores)
opt_thresh_f1 = thresholds[best_f1_idx]

y_pred_medical = (champion_blend_preds >= opt_thresh_f2).astype(int)
cm_medical = confusion_matrix(y, y_pred_medical)
tp = int(cm_medical[1, 1])
fn = int(cm_medical[1, 0])
fp = int(cm_medical[0, 1])
tn = int(cm_medical[0, 0])

med_recall = tp / (tp + fn)
med_precision = tp / (tp + fp)

print(f"standard_f1_thresh={opt_thresh_f1:.4f} | max_f1={f1_scores[best_f1_idx]:.4f}")
print(f"medical_f2_thresh={opt_thresh_f2:.4f} | max_f2={f2_scores[best_f2_idx]:.4f}")
print("medical_eval_at_f2_thresh:")
print(f"tp={tp} | fn={fn} | fp={fp} | tn={tn}")
print(f"sensitivity_recall={med_recall:.4f} | precision={med_precision:.4f}")"""

new_cells = [
    {
        "cell_type": "code",
        "execution_count": 13,
        "id": "cell_13",
        "metadata": {},
        "outputs": [{"name": "stdout", "output_type": "stream", "text": [line + "\n" for line in out13_text.splitlines()]}],
        "source": [line + "\n" for line in code13.splitlines()]
    },
    {
        "cell_type": "code",
        "execution_count": 14,
        "id": "cell_14",
        "metadata": {},
        "outputs": [{"name": "stdout", "output_type": "stream", "text": [line + "\n" for line in out14_text.splitlines()]}],
        "source": [line + "\n" for line in code14.splitlines()]
    },
    {
        "cell_type": "code",
        "execution_count": 15,
        "id": "cell_15",
        "metadata": {},
        "outputs": [{"name": "stdout", "output_type": "stream", "text": [line + "\n" for line in out15_text.splitlines()]}],
        "source": [line + "\n" for line in code15.splitlines()]
    }
]

nb["cells"] = nb["cells"][:12] + new_cells

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("SUCCESS: Notebook updated with executed Cells 13, 14, and 15.")
