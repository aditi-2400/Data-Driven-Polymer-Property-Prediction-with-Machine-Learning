from typing import Tuple
import pandas as pd
from rdkit import Chem

def load_train_test(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(train_path), pd.read_csv(test_path)

def canonicalize_smiles(df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
    out = df.copy()
    canon = []
    for smi in out[smiles_col].astype(str):
        try:
            mol = Chem.MolFromSmiles(smi)
            canon.append(Chem.MolToSmiles(mol, canonical=True) if mol else None)
        except Exception:
            canon.append(None)
    out[smiles_col] = canon
    return out
