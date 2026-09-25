import pandas as pd
from torch.utils.data import Dataset


class SmilesDataset(Dataset):
    def __init__(self, csv_path, smiles_col):
        df = pd.read_csv(csv_path)
        self.smiles = df[smiles_col].dropna().astype(str).tolist()

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx]


class TgDataset(Dataset):
    def __init__(
        self,
        csv_path,
        smiles_col="repeat_unit_smiles",
        target_col="tg_value",
        unit_col="tg_unit",
    ):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=[smiles_col, target_col]).copy()

        values = df[target_col].astype(float)
        units = df[unit_col].astype(str).str.upper()
        values.loc[units == "C"] += 273.15

        self.smiles = df[smiles_col].astype(str).tolist()
        self.targets = values.astype(float).tolist()

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], self.targets[idx]
