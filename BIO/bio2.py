import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1) Load the PubChem AID_1 datatable
df = pd.read_csv("AID_1_datatable.csv")

# 2) Keep the columns we need
df = df[["PUBCHEM_CID",
         "PUBCHEM_EXT_DATASOURCE_SMILES",
         "PUBCHEM_ACTIVITY_OUTCOME",
         "LogGI50_M"]]

# Drop rows without SMILES or outcome
df = df.dropna(subset=["PUBCHEM_EXT_DATASOURCE_SMILES",
                       "PUBCHEM_ACTIVITY_OUTCOME"])

# 3) Make binary label: Active = 1, Inactive = 0 (ignore others)
df = df[df["PUBCHEM_ACTIVITY_OUTCOME"].isin(["Active", "Inactive"])]
df["label"] = (df["PUBCHEM_ACTIVITY_OUTCOME"] == "Active").astype(int)

smiles_list = df["PUBCHEM_EXT_DATASOURCE_SMILES"].tolist()

# 4) Convert SMILES to Morgan fingerprints (RDKit)
def smiles_to_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

X = []
valid_idx = []

for i, smi in enumerate(smiles_list):
    fp = smiles_to_fp(smi)
    if fp is not None:
        X.append(fp)
        valid_idx.append(i)

X = np.array(X)
y = df["label"].iloc[valid_idx].values

# 5) Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6) Random Forest model
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)
clf.fit(X_train, y_train)

# 7) Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# 8) Save a CSV with predictions for all molecules
df_valid = df.iloc[valid_idx].copy()
df_valid["pred_active_prob"] = clf.predict_proba(X)[:, 1]
df_valid.to_csv("AID_1_predictions.csv", index=False)
