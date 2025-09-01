import numpy as np, pandas as pd
from typing import List

TARGETS = ["Tg","FFV","Tc","Density","Rg"]

def _prep(df: pd.DataFrame, targets):
    df = df.copy()
    df.dropna(subset=["SMILES"], inplace=True)
    for t in targets:
        if t in df.columns:
            df[t] = pd.to_numeric(df[t], errors="coerce")
    keep = ["SMILES"] + [t for t in targets if t in df.columns]
    df = df[keep]
    if len(keep) > 1:
        df = df.groupby("SMILES", as_index=False).mean()
    else:
        df = df.drop_duplicates(subset=["SMILES"])
    return df

def merge_supplements(train: pd.DataFrame, supplement_paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in (supplement_paths or []):
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            pass

    # Attempt to map ds1..ds4 semantics (Tc from TC_mean; Tg; FFV)
    ds1 = frames[0] if len(frames) > 0 else pd.DataFrame(columns=["SMILES"])
    ds2 = frames[1] if len(frames) > 1 else pd.DataFrame(columns=["SMILES"])
    ds3 = frames[2] if len(frames) > 2 else pd.DataFrame(columns=["SMILES"])
    ds4 = frames[3] if len(frames) > 3 else pd.DataFrame(columns=["SMILES"])

    ds1 = ds1[["SMILES"] + [c for c in ds1.columns if c in ("TC_mean","Tc")]].rename(columns={"TC_mean":"Tc"})
    ds2 = ds2[["SMILES"]]
    ds3 = ds3[["SMILES"] + [c for c in ds3.columns if c == "Tg"]]
    ds4 = ds4[["SMILES"] + [c for c in ds4.columns if c == "FFV"]]

    ds1 = _prep(ds1, ["Tc"])
    ds3 = _prep(ds3, ["Tg"])
    ds4 = _prep(ds4, ["FFV"])

    extras = ds1.merge(ds3, on="SMILES", how="outer").merge(ds4, on="SMILES", how="outer")

    merged = train.merge(extras, on="SMILES", how="left", suffixes=("", "_ext"))
    for t in TARGETS:
        if f"{t}_ext" in merged.columns:
            merged[t] = merged[t].where(~merged[t].isna(), merged[f"{t}_ext"])
            merged.drop(columns=[f"{t}_ext"], inplace=True)

    present = set(merged["SMILES"].unique())
    new_rows = extras[~extras["SMILES"].isin(present)].copy()
    if not new_rows.empty:
        for t in TARGETS:
            if t not in new_rows.columns:
                new_rows[t] = np.nan
        for col in merged.columns:
            if col not in new_rows.columns:
                new_rows[col] = np.nan
        new_rows = new_rows[merged.columns]
        merged = pd.concat([merged, new_rows], ignore_index=True)

    return merged
