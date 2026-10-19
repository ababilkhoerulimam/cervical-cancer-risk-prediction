#!/usr/bin/env python3
"""build_harmonized_dataset.py - End-to-end download and harmonization pipeline.

Downloads:
1. UCI 383 Cervical Cancer Risk Factors (858 rows)
2. CDC NHANES 2015-2016 (DEMO, RHQ, SXQ, SMQ, MCQ modules)
Produces:
- data/uci_cervical_cancer_858.csv
- data/uci_cervical_harmonized.csv
- data/nhanes_cervical_harmonized.csv
- data/harmonized_master_real.csv (3,567 rows)
"""

import io
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "nhanes_raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=45) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def step_1_fetch_uci():
    url_uci = "https://archive.ics.uci.edu/static/public/383/cervical+cancer+risk+factors.zip"
    dest_csv = DATA_DIR / "uci_cervical_cancer_858.csv"
    if not dest_csv.exists():
        req = urllib.request.Request(url_uci, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            with zipfile.ZipFile(io.BytesIO(resp.read())) as z:
                z.extract("risk_factors_cervical_cancer.csv", DATA_DIR)
        (DATA_DIR / "risk_factors_cervical_cancer.csv").rename(dest_csv)

    df_uci = pd.read_csv(dest_csv, na_values="?")
    uci_clean = pd.DataFrame()
    uci_clean["source"] = ["UCI_Caracas"] * len(df_uci)
    uci_clean["age"] = df_uci["Age"]
    uci_clean["num_sexual_partners"] = df_uci["Number of sexual partners"]
    uci_clean["first_sexual_intercourse"] = df_uci["First sexual intercourse"]
    uci_clean["num_pregnancies"] = df_uci["Num of pregnancies"]
    uci_clean["smokes"] = df_uci["Smokes"]
    uci_clean["hormonal_contraceptives"] = df_uci["Hormonal Contraceptives"]
    uci_clean["target"] = df_uci["Biopsy"].astype(int)
    uci_clean.to_csv(DATA_DIR / "uci_cervical_harmonized.csv", index=False)
    return uci_clean


def step_2_fetch_nhanes():
    files = {
        "DEMO_I.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DEMO_I.xpt",
        "RHQ_I.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/RHQ_I.xpt",
        "SXQ_I.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SXQ_I.xpt",
        "SMQ_I.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SMQ_I.xpt",
        "MCQ_I.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/MCQ_I.xpt",
    }
    for name, url in files.items():
        download_file(url, RAW_DIR / name)

    demo = pd.read_sas(RAW_DIR / "DEMO_I.xpt", format="xport")
    rhq = pd.read_sas(RAW_DIR / "RHQ_I.xpt", format="xport")
    sxq = pd.read_sas(RAW_DIR / "SXQ_I.xpt", format="xport")
    smq = pd.read_sas(RAW_DIR / "SMQ_I.xpt", format="xport")
    mcq = pd.read_sas(RAW_DIR / "MCQ_I.xpt", format="xport")

    df_nh = demo[demo["RIAGENDR"] == 2][["SEQN", "RIDAGEYR"]].copy()
    df_nh = df_nh.merge(rhq[["SEQN", "RHQ160", "RHQ420"]], on="SEQN", how="left")
    df_nh = df_nh.merge(sxq[["SEQN", "SXD021", "SXQ700"]], on="SEQN", how="left")
    df_nh = df_nh.merge(smq[["SEQN", "SMQ020"]], on="SEQN", how="left")
    df_nh = df_nh.merge(mcq[["SEQN", "MCQ220", "MCQ230A"]], on="SEQN", how="left")

    df_nh["target"] = np.where(df_nh["MCQ230A"] == 15.0, 1, np.where(df_nh["MCQ220"] == 2.0, 0, np.nan))
    df_nh = df_nh.dropna(subset=["target"]).copy()

    df_nh["age"] = df_nh["RIDAGEYR"]
    df_nh["num_pregnancies"] = df_nh["RHQ160"].replace({77: np.nan, 99: np.nan})
    df_nh["hormonal_contraceptives"] = df_nh["RHQ420"].replace({1: 1, 2: 0, 7: np.nan, 9: np.nan})
    df_nh["first_sexual_intercourse"] = df_nh["SXD021"].replace({777: np.nan, 999: np.nan})
    df_nh["num_sexual_partners"] = df_nh["SXQ700"].replace({7777: np.nan, 9999: np.nan})
    df_nh["smokes"] = df_nh["SMQ020"].replace({1: 1, 2: 0, 7: np.nan, 9: np.nan})
    df_nh["source"] = "CDC_NHANES"

    nhanes_clean = df_nh[
        ["source", "age", "num_sexual_partners", "first_sexual_intercourse", "num_pregnancies", "smokes", "hormonal_contraceptives", "target"]
    ].copy()
    nhanes_clean.to_csv(DATA_DIR / "nhanes_cervical_harmonized.csv", index=False)
    return nhanes_clean


def main():
    print("Building harmonized dataset...")
    uci_clean = step_1_fetch_uci()
    nhanes_clean = step_2_fetch_nhanes()
    master = pd.concat([uci_clean, nhanes_clean], ignore_index=True)
    master.to_csv(DATA_DIR / "harmonized_master_real.csv", index=False)
    print(f"Master dataset generated successfully: {master.shape[0]} rows, {master.shape[1]} columns.")
    print(f"Total positives: {int(master['target'].sum())} / {len(master)} ({master['target'].mean():.4f})")


if __name__ == "__main__":
    main()
