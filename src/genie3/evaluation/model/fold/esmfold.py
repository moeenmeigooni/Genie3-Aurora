# Modified for Aurora XPU compatibility by the FRAME-IDP Aurora port; see aurora/README.md.
"""ESMFold handler for the Genie pipeline.

The upstream ESM package requires OpenFold CUDA extensions.  This handler uses
the equivalent Hugging Face ESMFold implementation instead, which is pure
PyTorch and can therefore run on CUDA or on Aurora's native ``torch.xpu``
backend.
"""

import os
import json
import torch
import logging
import numpy as np

from genie3.evaluation.model.fold.base import FoldHandler
from genie3.evaluation.utils.parse import parse_fasta


class ESMFoldHandler(FoldHandler):
    """FoldHandler implementation using the ESMFold model."""

    NAME = 'esmfold'

    def setup(self):
        """
        Load the pure-PyTorch ESMFold model and move it to the selected device.

        ``GENIE3_ESMFOLD_MODEL_DIR`` may point at a pre-downloaded model
        directory.  Aurora's activation script sets it to project storage so
        model downloads never touch the user's home directory.
        """
        if self.version not in ['unconditional', 'scaffold']:
            logging.error(f'Invalid version: {self.version}')
            exit(0)

        self.model = None
        if self.preload:
            from transformers import EsmForProteinFolding

            model_source = os.environ.get(
                "GENIE3_ESMFOLD_MODEL_DIR", "facebook/esmfold_v1"
            )
            logging.info(
                "[ESMFoldHandler] Loading Transformers ESMFold from %s on %s",
                model_source,
                self.device,
            )
            self.model = EsmForProteinFolding.from_pretrained(model_source)
            self.model = self.model.eval().to(self.device)

    def fold(self, structures_dir):
        """
        Fold all sequences in the input FASTA and save predicted structures.

        Reads sequences from structures_dir/input.fasta, runs ESMFold inference
        on each, and writes per-sequence PDB files to structures_dir.

        Args:
            structures_dir: Directory containing input.fasta and where output
                            PDB files will be written
        """

        # Sanity check
        sequences_filepath = os.path.join(structures_dir, 'input.fasta')
        if not os.path.exists(sequences_filepath):
            logging.error('Missing sequences filepath')
            exit(0)

        # Load sequences
        seq_by_name = parse_fasta(sequences_filepath)

        # Iterate
        for name in seq_by_name:

            # Set up
            outdir = os.path.join(structures_dir, name)
            os.makedirs(outdir)

            # Predict
            with torch.inference_mode():
                output = self.model.infer(seq_by_name[name])
                pdb_str = self.model.output_to_pdb(output)[0]
                pae_tensor = output.predicted_aligned_error
                if pae_tensor is None:
                    # Kept for compatibility with older ESMFold checkpoints.
                    pae_tensor = (
                        output.aligned_confidence_probs
                        * torch.arange(
                            output.aligned_confidence_probs.shape[-1],
                            device=output.aligned_confidence_probs.device,
                        )
                    ).mean(-1) * 31
                pae = pae_tensor.detach().float().cpu().numpy()[0]
                mask = output.atom37_atom_exists.detach().cpu().numpy()[0, :, 1] == 1
                pae = pae[mask,:][:,mask]
                plddt = output.plddt[0, :, 1].detach().float().cpu()
            
            # Save
            with open(os.path.join(outdir, f'{name}.pdb'), 'w') as file:
                file.write(pdb_str)
            with open(os.path.join(outdir, f'scores_{name}.json'), 'w') as file:
                out = {
                    'plddt': plddt.tolist(),
                    'pae': pae.tolist()
                }
                json.dump(out, file, indent=4)

    def _compile_filepath(self, design_filepath):
        """
        Parse name, domain, and file paths from an ESMFold design filepath.

        Args:
            design_filepath: Path to the predicted PDB file

        Returns:
            dict: 'name', 'domain', 'design_filepath', 'info_filepath'
        """
        sample_name = design_filepath.split('/')[-2]
        domain_name = sample_name.split('-')[0]
        resample_index = int(sample_name.split('-')[-1].split('_')[-1])

        elts = design_filepath.split('/')
        elts = elts[:-1] + ['scores_' + elts[-1].replace('.pdb', '.json')]
        info_filepath = '/'.join(elts)

        return {
            'name': sample_name,
            'domain': domain_name,
            'resample_id': resample_index,
            'design_filepath': design_filepath,
            'info_filepath': info_filepath
        }

    def _compile_rank(self, design_filepath):
        """
        Return rank 1 for ESMFold (single-model, no ranking).

        Args:
            design_filepath: Path to the predicted PDB file (unused)

        Returns:
            dict: {'rank': 1}
        """
        return {}
    
    def _compile_confidence(self, info_filepath):
        """
        Extract pLDDT and pAE from an ESMFold JSON file.

        Args:
            info_filepath: Path to the JSON scores file

        Returns:
            dict: 'plddt', 'pae'
        """
        with open(info_filepath) as file:
            info = json.load(file)
        return {
            'plddt': np.array(info['plddt']),
            'pae': np.array(info['pae'])
        }
    
    def _compile_ipsae(self, info_filepath, design_filepath):
        """
        Not implemented for ESMFold (no ipSAE output).

        Args:
            info_filepath: Path to the JSON scores file
            design_filepath: Path to the predicted structure file
        """
        return {}
