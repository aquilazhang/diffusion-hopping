import pandas as pd
import torch
from biopandas.pdb import PandasPdb
from biopandas.pdb.engines import amino3to1dict
from rdkit import Chem
from torch_geometric.data import HeteroData

from diffusion_hopping.data.featurization.util import (
    atom_names,
    atomic_symbols_to_one_hot,
    get_ligand_scaffold_mask,
    get_scaff_idx,
    residue_names,
    residue_names_to_one_hot,
)
from diffusion_hopping.data.protein_ligand import ProteinLigandComplex
from diffusion_hopping.data.transform import ChainSelectionTransform


class ProteinLigandSimpleFeaturization:
    def __init__(
        self, cutoff=10, c_alpha_only=False, center_complex=False, mode="chain"
    ) -> None:
        self.c_alpha_only = c_alpha_only
        self.center_complex = center_complex
        self.cutoff = cutoff
        self.mode = mode
        self.chain_selection_transform = ChainSelectionTransform(
            cutoff=cutoff, mode=mode
        )
        self.protein_features = len(residue_names) if c_alpha_only else len(atom_names)
        self.ligand_features = len(atom_names)

    @staticmethod
    def get_ligand_atom_types(ligand: Chem.Mol) -> torch.Tensor:
        return atomic_symbols_to_one_hot(
            [atom.GetSymbol() for atom in ligand.GetAtoms()]
        )

    @staticmethod
    def get_protein_atom_types(protein: pd.DataFrame) -> torch.Tensor:
        return atomic_symbols_to_one_hot(protein["element_symbol"].values)

    def get_protein_residues(self, protein: pd.DataFrame) -> torch.Tensor:
        residue_names = protein["residue_name"].map(amino3to1dict).values
        return residue_names_to_one_hot(residue_names)

    @staticmethod
    def get_ligand_positions(ligand: Chem.Mol) -> torch.Tensor:
        return torch.tensor(ligand.GetConformer().GetPositions(), dtype=torch.float)

    @staticmethod
    def get_protein_positions(protein: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(
            protein[["x_coord", "y_coord", "z_coord"]].values, dtype=torch.float
        )

    def __call__(self, complex: ProteinLigandComplex, scaff_mols = None, atom_nums=0) -> HeteroData:
        # atom nums the number of atoms need to add/extract from ligand_linker
        # remove all hydrogens from ligand
        ligand = complex.ligand.rdkit_mol()
        ligand = Chem.RemoveHs(ligand)

        if scaff_mols is not None:
            # modify the ligand (add some dummy atoms or extract some atoms)
            kept_index = torch.LongTensor(get_scaff_idx(ligand=ligand,scaff_mols=scaff_mols))
            dropped_index = torch.LongTensor([i for i in range(ligand.GetNumAtoms()) if i not in kept_index])

        # remained parts: False
        scaffold_mask = get_ligand_scaffold_mask(ligand,scaff_mols=scaff_mols)        
        x_ligand = self.get_ligand_atom_types(ligand)
        pos_ligand = self.get_ligand_positions(ligand)

        if atom_nums == 0:
            pass
        elif atom_nums > 0:
            # copy the first several atoms in linker
            x_ligand = torch.cat([x_ligand,x_ligand.index_select(dim=0,index=dropped_index[:atom_nums])],dim=0)
            pos_ligand = torch.cat([pos_ligand,pos_ligand.index_select(dim=0,index=dropped_index[:atom_nums])],dim=0)
            scaffold_mask = torch.cat([scaffold_mask,torch.zeros(atom_nums,dtype=bool)],dim=0)
        elif atom_nums < 0:
            # delete the last several atoms in linker
            x_ligand = torch.cat([
                x_ligand.index_select(dim=0,index=kept_index),
                # [:atom_nums] like [:-2]
                x_ligand.index_select(dim=0,index=dropped_index[:atom_nums])
            ],dim=0)
            pos_ligand = torch.cat([
                pos_ligand.index_select(dim=0,index=kept_index),
                pos_ligand.index_select(dim=0,index=dropped_index[:atom_nums])
            ])
            scaffold_mask = torch.cat([
                scaffold_mask.index_select(dim=0,index=kept_index),
                scaffold_mask.index_select(dim=0,index=dropped_index[:atom_nums])
            ])
        protein: PandasPdb = complex.protein.pandas_pdb()
        protein: PandasPdb = self.chain_selection_transform(protein, pos_ligand)

        if self.c_alpha_only:
            protein_df = protein.get("c-alpha")
            protein_df = protein_df[protein_df["residue_name"].isin(amino3to1dict)]
            x_protein = self.get_protein_residues(protein_df)
        else:
            # remove all hydrogens from protein
            protein_df = protein.get("hydrogen", invert=True)
            x_protein = self.get_protein_atom_types(protein_df)

        pos_protein = self.get_protein_positions(protein_df)

        # if scaffold_mask.sum() == 0:
        #    raise ValueError("No scaffold atom identified")
        # elif scaffold_mask.sum() == torch.numel(scaffold_mask):
        #    raise ValueError("Only scaffold atoms identified")

        if self.center_complex:
            dtype = pos_ligand.dtype
            mean_pos = pos_ligand.to(torch.float64).mean(dim=0)
            pos_ligand = (pos_ligand.to(torch.float64) - mean_pos).to(dtype)
            pos_protein = (pos_protein.to(torch.float64) - mean_pos).to(dtype)

        return HeteroData(
            ligand={
                "x": x_ligand,
                "pos": pos_ligand,
                "scaffold_mask": scaffold_mask,
                "ref": ligand,
                "path": complex.ligand.path,
            },
            protein={
                "x": x_protein,
                "pos": pos_protein,
                # "ref": protein_df, # maybe add protein df
                "path": complex.protein.path,
            },
            identifier=complex.identifier,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cutoff={self.cutoff}, c_alpha_only={self.c_alpha_only}, center_complex={self.center_complex}, mode={self.mode})"
