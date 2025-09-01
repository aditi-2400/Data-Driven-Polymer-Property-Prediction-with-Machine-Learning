from typing import Dict, List, Tuple
import numpy as np, pandas as pd

def _try_import():
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, AllChem
    return Chem, DataStructs, Descriptors, AllChem

def build_features(df: pd.DataFrame, feats_cfg: Dict) -> Tuple[pd.DataFrame, List[str], List[str]]:
    Chem, DataStructs, Descriptors, AllChem = _try_import()
    use_desc = feats_cfg.get("use_rdkit_descriptors", True)
    use_morgan = feats_cfg.get("use_morgan_fp", True)
    radius = feats_cfg.get("morgan_radius", 2)
    nbits = feats_cfg.get("morgan_nbits", 1024)

    desc_names = [n for n, _ in Descriptors._descList] if use_desc else []
    fp_names = [f"FP_{i}" for i in range(nbits)] if use_morgan else []
    names = desc_names + fp_names

    rows, idxs = [], []
    for idx, smi in zip(df.index, df["SMILES"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        parts = []
        if use_desc:
            dvals = []
            for _, fn in Descriptors._descList:
                try:
                    dvals.append(float(fn(mol)))
                except Exception:
                    dvals.append(np.nan)
            parts += dvals
        if use_morgan:
            try:
                bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
                arr = np.zeros((nbits,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(bv, arr)
                parts += arr.astype(float).tolist()
            except Exception:
                parts += [0.0]*nbits
        rows.append(parts)
        idxs.append(idx)

    feat_df = pd.DataFrame(rows, index=idxs, columns=names)
    return feat_df, desc_names, fp_names
