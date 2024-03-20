from typing import Any
import torch
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
from pytorch_lightning import LightningModule
from src.analysis import metrics 
from src.analysis import utils as au
from src.models.flow_model import FlowModel
from src.models import utils as mu
from src.data.interpolant import Interpolant 
from src.data import utils as du
from src.data import all_atom
from src.data.components.pdb import all_atom as rna_all_atom
from src.data import so3_utils
# from src.data import residue_constants # CHANGE
from src.data import nucleotide_constants
# from src.experiments import utils as eu
from src.analysis import utils as au
from pytorch_lightning.loggers.wandb import WandbLogger


class FlowModule(LightningModule):

    def __init__(self, cfg, folding_cfg=None):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant

        # Set-up vector field prediction model
        self.model = FlowModel(cfg.model)

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)

        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()
        
    def on_train_start(self):
        self._epoch_start_time = time.time()
        
    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()

    def model_step(self, noisy_batch: Any):
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask']
        is_protein_residue_mask, is_na_residue_mask = noisy_batch["is_protein_residue_mask"], noisy_batch["is_na_residue_mask"]
        
        if training_cfg.min_plddt_mask is not None:
            plddt_mask = noisy_batch['res_plddt'] > training_cfg.min_plddt_mask
            loss_mask *= plddt_mask
        
        num_batch, num_res = loss_mask.shape

        # Ground truth labels
        bb_frame_atom_idx = [2, 3, 6] # C3', C4', O4' (NOT including the C5' since it wasn't in the RNA Frame construction)
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, gt_rotmats_1.type(torch.float32))

        # gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3]
        gt_bb_atoms = rna_all_atom.to_atom27(
                            gt_trans_1, gt_rotmats_1, 
                            num_batch, num_res, 
                            is_protein_residue_mask, is_na_residue_mask
                        )[:, :, bb_frame_atom_idx]

        # Timestep used for normalization.
        t = noisy_batch['t']
        norm_scale = 1 - torch.min(t[..., None], torch.tensor(training_cfg.t_normalize_clip)) # []
        
        # Model output predictions.
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)

        # Backbone atom loss
        # pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3] # TODO: look into implementing to_atom27
        pred_bb_atoms = rna_all_atom.to_atom27(
                            pred_trans_1, pred_rotmats_1, 
                            num_batch, num_res, 
                            is_protein_residue_mask, is_na_residue_mask
                        )[:, :, bb_frame_atom_idx]
        
        gt_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        bb_atom_loss = torch.sum((gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None], dim=(-1, -2, -3)) / loss_denom

        # Translation VF loss
        trans_error = (gt_trans_1 - pred_trans_1) / norm_scale * training_cfg.trans_scale
        trans_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / norm_scale
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # NOTE: ignore aux losses, ignore this computation
        # """
        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res*3, 3]) # NOTE: figure out whether it's still num_res*3 for RNA
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res*3, 3]) # NOTE: figure out whether it's still num_res*3 for RNA
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res*3]) # NOTE: figure out whether it's still num_res*3 for RNA
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res*3]) # NOTE: figure out whether it's still num_res*3 for RNA

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum((gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask, dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)
        # """

        se3_vf_loss = trans_loss + rots_vf_loss # NOTE: main loss to analyse

        # NOTE: set aux loss weight to 0 –> ignore dist_mat_loss
        auxiliary_loss = (bb_atom_loss + dist_mat_loss) * (t[:, 0] > training_cfg.aux_loss_t_pass)
        # auxiliary_loss = (bb_atom_loss) * (t[:, 0] > training_cfg.aux_loss_t_pass)

        """
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
        """

        se3_vf_loss += auxiliary_loss
        
        if torch.isnan(se3_vf_loss).any():
            raise ValueError('NaN loss encountered')
        
        return {
            "bb_atom_loss": bb_atom_loss,
            "trans_loss": trans_loss,
            "dist_mat_loss": dist_mat_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": se3_vf_loss
        }

    # TODO: change back to validation_step(...)
    def validation_step(self, batch, batch_idx):
        '''
        RNA structural metrics
            - save samples in PDB format
            - run sample-wise analysis
            - collect batch-wise metrics
        '''
        # print ("BATCH IDX:", batch_idx)

        res_mask = batch['res_mask']
        is_na_residue_mask = batch['is_na_residue_mask'].detach().cpu().numpy()
        # print (is_na_residue_mask.shape)
        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape
        
        samples = self.interpolant.sample(
            num_batch,
            num_res,
            self.model,
        )[0][-1].numpy()

        # print (samples.shape)
        print (f"\tVALIDATING ... SAVING TO {self._sample_write_dir} | {num_batch}")

        batch_metrics = []

        for i in range(num_batch):
            final_pos = samples[i]
            saved_rna_path = au.write_complex_to_pdbs( # Save RNA atoms to PDB
                final_pos,
                os.path.join(self._sample_write_dir, f'sample_{i}_idx_{batch_idx}_len_{num_res}.pdb'),
                is_na_residue_mask=is_na_residue_mask[i],
                no_indexing=True
            )
            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_samples.append(
                    [saved_rna_path, self.global_step, wandb.Molecule(saved_rna_path)]
                )

            # TODO: Add conversion to PDB + metric calculation
            # struct_rna_metrics = metrics.calc_rna_struct_metrics(saved_rna_path)
            # c4_idx = nucleotide_constants.atom_order['C4\'']
            # rna_c4_c4_matrics = metrics.calc_rna_c4_c4_metrics(final_pos[:, c4_idx])
            # batch_metrics.append((struct_rna_metrics | rna_c4_c4_matrics))

            # NOTE: placeholder structural metric dictionary
            struct_rna_metrics = {
                'non_coil_percent': 0,
                'coil_percent': 0,
                'helix_percent': 0,
                'strand_percent': 0,
                'radius_of_gyration': 0
            }
            batch_metrics.append(struct_rna_metrics)

        batch_metrics = pd.DataFrame(batch_metrics)
        self.validation_epoch_metrics.append(batch_metrics)

    # def _validation_step(self, batch: Any, batch_idx: int):
    #     '''
    #     NOTE: Original validation function – comment out before running code
    #     '''
    #     res_mask = batch['res_mask']
    #     self.interpolant.set_device(res_mask.device)
    #     num_batch, num_res = res_mask.shape
        
    #     samples = self.interpolant.sample(
    #         num_batch,
    #         num_res,
    #         self.model,
    #     )[0][-1].numpy()

    #     batch_metrics = []
    #     for i in range(num_batch):

    #         # Write out sample to PDB file
    #         final_pos = samples[i]
    #         saved_path = au.write_prot_to_pdb(
    #             final_pos,
    #             os.path.join(
    #                 self._sample_write_dir,
    #                 f'sample_{i}_idx_{batch_idx}_len_{num_res}.pdb'),
    #             no_indexing=True
    #         )
    #         if isinstance(self.logger, WandbLogger):
    #             self.validation_epoch_samples.append(
    #                 [saved_path, self.global_step, wandb.Molecule(saved_path)]
    #             )

    #         mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
    #         ca_idx = residue_constants.atom_order['CA'] # retrieve CA atoms –> TODO: change to C4'
    #         ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, ca_idx])
    #         batch_metrics.append((mdtraj_metrics | ca_ca_metrics))

    #     batch_metrics = pd.DataFrame(batch_metrics)
    #     self.validation_epoch_metrics.append(batch_metrics)
        
    def on_validation_epoch_end(self):
        if len(self.validation_epoch_samples) > 0:
            self.logger.log_table(
                key='valid/samples',
                columns=["sample_path", "global_step", "Protein"],
                data=self.validation_epoch_samples)
            self.validation_epoch_samples.clear()

        val_epoch_metrics = pd.concat(self.validation_epoch_metrics)
        
        for metric_name, metric_val in val_epoch_metrics.mean().to_dict().items():
            self._log_scalar(
                f'valid/{metric_name}',
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics),
            )
        self.validation_epoch_metrics.clear()

    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )

    def training_step(self, batch: Any, stage: int):
        # print (batch['aatype'].shape)
        # print (stage)
        step_start_time = time.time()
        self.interpolant.set_device(batch['res_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)

        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch['trans_sc'] = model_sc['pred_trans']
        
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses['bb_atom_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        
        for k,v in total_losses.items():
            self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        # Losses to track. Stratified across t.
        t = torch.squeeze(noisy_batch['t'])
        self._log_scalar(
            "train/t",
            np.mean(du.to_numpy(t)),
            prog_bar=False, batch_size=num_batch)
        for loss_name, loss_dict in batch_losses.items():
            stratified_losses = mu.t_stratified_loss(
                t, loss_dict, loss_name=loss_name)
            for k,v in stratified_losses.items():
                self._log_scalar(
                    f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Training throughput
        self._log_scalar("train/length", batch['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
        self._log_scalar("train/batch_size", num_batch, prog_bar=False)
        
        step_time = time.time() - step_start_time
        self._log_scalar("train/eps", num_batch / step_time)
        
        train_loss = (
            total_losses[self._exp_cfg.training.loss]
            +  total_losses['auxiliary_loss']
        )
        self._log_scalar("train/loss", train_loss, batch_size=num_batch)

        return train_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )

    def predict_step(self, batch, batch_idx):
        device = f'cuda:{torch.cuda.current_device()}'
        interpolant = Interpolant(self._infer_cfg.interpolant) 
        interpolant.set_device(device)

        sample_length = batch['num_res'].item()
        diffuse_mask = torch.ones(1, sample_length)
        sample_id = batch['sample_id'].item()
        sample_dir = os.path.join(self._output_dir, f'length_{sample_length}', f'sample_{sample_id}')
        
        top_sample_csv_path = os.path.join(sample_dir, 'top_sample.csv')
        
        if os.path.exists(top_sample_csv_path):
            self._print_logger.info(f'Skipping instance {sample_id} length {sample_length}')
            return

        atom37_traj, model_traj, _ = interpolant.sample(1, sample_length, self.model)

        os.makedirs(sample_dir, exist_ok=True)
        bb_traj = du.to_numpy(torch.concat(atom37_traj, dim=0))

        sample = bb_traj[-1] # NOTE: store final state as completed sample

        sample_path = au.write_prot_to_pdb(
                        sample,
                        sample_dir,
                        no_indexing=True,
                    )
        
        # _ = eu.save_traj(
        #     bb_traj[-1],
        #     bb_traj,
        #     np.flip(du.to_numpy(torch.concat(model_traj, dim=0)), axis=0),
        #     du.to_numpy(diffuse_mask)[0],
        #     output_dir=sample_dir,
        # )
