# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Library for parsing different data structures.
Code adapted from Openfold protein.py.
"""

import numpy as np
import torch
from beartype import beartype
from Bio.PDB import Chain, Model, Structure

from types import ModuleType
from typing import Any, Dict, Optional, Tuple

from src.data import protein
from src.data import (
    nucleotide_constants,
    parsing,
    protein,
    protein_constants,
    residue_constants
)

Protein = protein.Protein
MACROMOLECULE_OUTPUTS_TYPE = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]

def parse_chain_for_molecule_type_validity(
    structure: Structure.Structure,
    chain_index: int,
    chain_id: str,
    molecule_constants: ModuleType,
    nmr_okay: bool = False,
    skip_nonallowable_restypes: bool = True,
    with_gaps: bool = False,
) -> MACROMOLECULE_OUTPUTS_TYPE:
    # evaluate whether the input chain corresponds to a protein or instead a nucleic acid molecule
    X, C, S, metadata = parsing.structure_to_XCS(
        structure=structure,
        constants=molecule_constants,
        chain_id=chain_id,
        nmr_okay=nmr_okay,
        skip_nonallowable_restypes=skip_nonallowable_restypes,
        with_gaps=with_gaps,
    )
    is_valid_molecule_type = all(
        [len(X), len(C), len(S), all([len(v) for v in metadata.values()])]
    )
    metadata["chain_indices"] = torch.tensor(
        [chain_index for _ in range(len(metadata["chain_ids"]))], dtype=torch.long
    )
    metadata["is_valid_molecule_type"] = is_valid_molecule_type
    return X, C, S, metadata

def process_chain(chain: Chain, chain_id: str) -> Protein:
    """Convert a PDB chain object into a AlphaFold Protein instance.
    
    Forked from alphafold.common.protein.from_pdb_string
    
    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.
    
    Took out lines 94-97 which don't allow insertions in the PDB.
    Sabdab uses insertions for the chothia numbering so we need to allow them.
    
    Took out lines 110-112 since that would mess up CDR numbering.
    
    Args:
        chain: Instance of Biopython's chain class.
    
    Returns:
        Protein object with protein features.
    """
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []
    chain_ids = []
    for res in chain:
        res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
        restype_idx = residue_constants.restype_order.get(
            res_shortname, residue_constants.restype_num)
        pos = np.zeros((residue_constants.atom_type_num, 3))
        mask = np.zeros((residue_constants.atom_type_num,))
        res_b_factors = np.zeros((residue_constants.atom_type_num,))
        for atom in res:
            if atom.name not in residue_constants.atom_types:
                continue
            pos[residue_constants.atom_order[atom.name]] = atom.coord
            mask[residue_constants.atom_order[atom.name]] = 1.
            res_b_factors[residue_constants.atom_order[atom.name]
                          ] = atom.bfactor
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        b_factors.append(res_b_factors)
        chain_ids.append(chain_id)

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=np.array(chain_ids),
        b_factors=np.array(b_factors))

def process_chain_pdb(
    chain: Chain.Chain,
    chain_index: int,
    chain_id: str,
    nmr_okay: bool = False,
    skip_nonallowable_restypes: bool = True,
    with_gaps: bool = False,
    verbose: bool = False,
) -> Optional[MACROMOLECULE_OUTPUTS_TYPE]:
    """Convert a PDB chain object into Profluent `Macromolecule` outputs.

    Args:
        chain: Instance of Biopython's chain class.

    Returns
    -------
    X : torch.Tensor
        The coordinates of the atoms in the structure.
    C : torch.Tensor
        The chain IDs and mask for the structure.
    S : torch.Tensor
        The sequence of the structure.
    metadata : dict
        The metadata for the structure.
    """
    structure = Structure.Structure(chain.full_id[0])
    model = Model.Model(0)
    structure.add(model)
    model.add(chain)
    # evaluate whether the input chain corresponds to a protein molecule
    X, C, S, metadata = parse_chain_for_molecule_type_validity(
        structure=structure,
        chain_index=chain_index,
        chain_id=chain_id,
        molecule_constants=protein_constants,
        nmr_okay=nmr_okay,
        skip_nonallowable_restypes=skip_nonallowable_restypes,
        with_gaps=with_gaps,
    )
    if metadata["is_valid_molecule_type"]:
        # annotate the given chain as belonging to a protein molecule
        metadata["molecule_type"] = "protein"
        metadata["molecule_type_encoding"] = torch.tensor(
            [[1, 0, 0, 0] for _ in range(len(S))], dtype=torch.long
        )
        metadata["molecule_constants"] = protein_constants
        metadata["molecule_backbone_atom_name"] = "CA"
        return X, C, S, metadata
    # evaluate whether the input chain instead corresponds to a nucleic acid molecule
    X, C, S, metadata = parse_chain_for_molecule_type_validity(
        structure=structure,
        chain_index=chain_index,
        chain_id=chain_id,
        molecule_constants=nucleotide_constants,
        nmr_okay=nmr_okay,
        skip_nonallowable_restypes=skip_nonallowable_restypes,
        with_gaps=with_gaps,
    )
    if metadata["is_valid_molecule_type"]:
        # annotate the given chain as belonging to a nucleic acid molecule
        metadata["molecule_type"] = "na"
        if metadata["deoxy"].all().item():
            # note: denotes a DNA molecule type
            metadata["molecule_type_encoding"] = torch.tensor(
                [[0, 1, 0, 0] for _ in range(len(S))], dtype=torch.long
            )
        else:
            # note: denotes an RNA molecule type
            metadata["molecule_type_encoding"] = torch.tensor(
                [[0, 0, 1, 0] for _ in range(len(S))], dtype=torch.long
            )
        metadata["molecule_constants"] = nucleotide_constants
        metadata["molecule_backbone_atom_name"] = "C4'"
        return X, C, S, metadata
    else:
        if verbose:
            print (
                f"Chain {chain} with ID {chain_id} was unable to be processed as either a protein chain or a nucleic acid chain."
            )
        return None        

def macromolecule_outputs_to_dict(outputs: MACROMOLECULE_OUTPUTS_TYPE) -> Dict[str, Any]:
    outputs_dict = {
        "atom_positions": outputs[0].numpy(),
        "atom_chain_id_mask": outputs[1].numpy(),
        "aatype": outputs[2].numpy(),
        "atom_mask": outputs[3]["atom_mask"].numpy(),
        "atom_chain_indices": outputs[3]["chain_indices"].numpy(),
        "atom_deoxy": outputs[3]["deoxy"].numpy(),
        "atom_b_factors": outputs[3]["b_factors"].numpy(),
        "molecule_type_encoding": outputs[3]["molecule_type_encoding"].numpy(),
    }
    return outputs_dict        