from rdkit import Chem
from rdkit.Chem import AllChem

# Example SMILES (replace with yours)
smiles = "CC(=O)Oc1ccccc1C(=O)O"   # Aspirin

# Step 1: Convert SMILES to molecule
mol = Chem.MolFromSmiles(smiles)

# Step 2: Add hydrogens
mol = Chem.AddHs(mol)

# Step 3: Generate 3D coordinates
AllChem.EmbedMolecule(mol, AllChem.ETKDG())

# Step 4: Energy minimization
AllChem.UFFOptimizeMolecule(mol)

# Step 5: Save as PDB (for docking)
Chem.MolToPDBFile(mol, "ligand.pdb")

print("✅ 3D structure saved as ligand.pdb")
