from typing import List

import rdkit.Chem
from rdkit import Chem
import torch
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch import nn as nn

atom_names = ["C", "N", "O", "S", "B", "Br", "Cl", "P", "I", "F"]

residue_names = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]

def get_ligand_scaffold_mask(ligand: rdkit.Chem.Mol, scaff_mols = None) -> torch.Tensor:
    if scaff_mols is None:
        murcko_scaffold = MurckoScaffold.GetScaffoldForMol(ligand)
        scaffold_substructures = ligand.GetSubstructMatches(murcko_scaffold)
        substructure_indexes = [
            i for substructure in scaffold_substructures for i in substructure
        ]
        mask = torch.zeros(len(ligand.GetAtoms()), dtype=torch.bool)
        mask[substructure_indexes] = True
        return mask
    else:
        substructure_indexes = get_scaff_idx(ligand=ligand,scaff_mols=scaff_mols)
        # substructure_indexes is linker index
        mask = torch.ones(len(ligand.GetAtoms()), dtype=torch.bool)
        mask[substructure_indexes] = False
        return mask

def get_scaff_idx(ligand: rdkit.Chem.Mol, scaff_mols: List[rdkit.Chem.Mol]) -> List[int]:
    scaff_mols = [Chem.RemoveHs(s) for s in scaff_mols]
    scaff_substructures = [ligand.GetSubstructMatches(mol) for mol in scaff_mols]
    substructure_indexes = list(set([
        i for scaff_substructure in scaff_substructures for \
            scaff_structure in scaff_substructure for i in scaff_structure
    ]))
    return substructure_indexes

def one_hot(x: List["str"], classes: List["str"]) -> torch.Tensor:
    return nn.functional.one_hot(
        torch.tensor([classes.index(s) for s in x], dtype=torch.long), len(classes)
    )


def atomic_symbols_to_one_hot(symbols: List[str]) -> torch.FloatTensor:
    return one_hot(symbols, atom_names)


def residue_names_to_one_hot(names: List[str]) -> torch.Tensor:
    return one_hot(names, residue_names)
