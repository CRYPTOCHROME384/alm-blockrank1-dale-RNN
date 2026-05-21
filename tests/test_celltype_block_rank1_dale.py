import os
import sys
import unittest

import torch


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
CURRENT_RNN_DIR = os.path.join(PROJECT_ROOT, "current_rnn")
if CURRENT_RNN_DIR not in sys.path:
    sys.path.append(CURRENT_RNN_DIR)

from models import CellTypeBlockRank1DaleCurrentRNN
from losses import LossAverageTrials
from training_blockrank1_dale import _compute_epoch_loss_bundle, _normalize_loss_epoch_weights


class CellTypeBlockRank1DaleTests(unittest.TestCase):
    def test_forward_shape_and_dale_rank_constraints(self):
        type_names = ["E_broad", "PV", "SST"]
        type_signs = [1.0, -1.0, -1.0]
        neuron_type_index = torch.as_tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)

        net = CellTypeBlockRank1DaleCurrentRNN(
            N=9,
            D_in=4,
            neuron_type_index=neuron_type_index,
            type_names=type_names,
            type_signs=type_signs,
            dt=0.1,
            tau=1.0,
            substeps=1,
            nonlinearity="tanh",
            init_A=0.2,
            init_factor_scale=0.01,
        )

        u = torch.randn(2, 5, 4)
        out = net(u, return_rate=True)
        self.assertEqual(tuple(out["h"].shape), (2, 5, 9))
        self.assertEqual(tuple(out["rate"].shape), (2, 5, 9))

        J = net.materialize_J_for_debug()
        self.assertEqual(tuple(J.shape), (9, 9))
        self.assertTrue(torch.isfinite(J).all().item())

        idx_E = torch.where(neuron_type_index == 0)[0]
        idx_PV = torch.where(neuron_type_index == 1)[0]
        idx_SST = torch.where(neuron_type_index == 2)[0]

        self.assertTrue(torch.all(J[:, idx_E] >= -1e-10).item())
        self.assertTrue(torch.all(J[:, idx_PV] <= 1e-10).item())
        self.assertTrue(torch.all(J[:, idx_SST] <= 1e-10).item())

        ranks = net.numerical_block_ranks(tol=1e-6)
        for block_name, rank in ranks.items():
            self.assertEqual(rank, 1, msg=f"{block_name} should be numerical rank 1, got {rank}")

        for row in net.block_parameter_summary():
            self.assertAlmostEqual(float(row["u_norm_mean"]), 1.0, places=5)
            self.assertAlmostEqual(float(row["v_norm_mean"]), 1.0, places=5)

    def test_epoch_weighted_loss_bundle(self):
        loss_fn = LossAverageTrials()
        target = torch.zeros(2, 5, 1)
        pred = torch.zeros(2, 5, 1)
        pred[:, 0:2, :] = 1.0
        pred[:, 2:3, :] = 2.0
        pred[:, 3:5, :] = 3.0
        masks = {
            "sample": torch.tensor([True, True, False, False, False]).numpy(),
            "delay": torch.tensor([False, False, True, False, False]).numpy(),
            "response": torch.tensor([False, False, False, True, True]).numpy(),
        }
        loss_mask = torch.tensor([True, True, True, True, True], dtype=torch.bool)
        weights = _normalize_loss_epoch_weights({"sample": 1.0, "delay": 2.0, "response": 1.0})

        out = _compute_epoch_loss_bundle(
            loss_fn=loss_fn,
            target=target,
            pred=pred,
            loss_mask_t=loss_mask,
            epoch_masks=masks,
            device=pred.device,
            loss_mode="epoch_weighted_mean",
            loss_epoch_weights=weights,
        )

        self.assertAlmostEqual(float(out["overall_loss"].item()), 4.8, places=6)
        self.assertAlmostEqual(float(out["epoch_losses"]["sample"].item()), 1.0, places=6)
        self.assertAlmostEqual(float(out["epoch_losses"]["delay"].item()), 4.0, places=6)
        self.assertAlmostEqual(float(out["epoch_losses"]["response"].item()), 9.0, places=6)
        self.assertAlmostEqual(float(out["weighted_loss"].item()), 4.5, places=6)
        self.assertAlmostEqual(float(out["selected_loss"].item()), 4.5, places=6)

    def test_epoch_weighted_loss_skips_empty_masks(self):
        loss_fn = LossAverageTrials()
        target = torch.zeros(2, 3, 1)
        pred = torch.zeros(2, 3, 1)
        pred[:, 0:1, :] = 1.0
        pred[:, 1:3, :] = 2.0
        masks = {
            "sample": torch.tensor([True, False, False]).numpy(),
            "delay": torch.tensor([False, True, True]).numpy(),
            "response": torch.tensor([False, False, False]).numpy(),
        }
        weights = _normalize_loss_epoch_weights({"sample": 1.0, "delay": 2.0, "response": 1.0})

        out = _compute_epoch_loss_bundle(
            loss_fn=loss_fn,
            target=target,
            pred=pred,
            loss_mask_t=torch.tensor([True, True, True], dtype=torch.bool),
            epoch_masks=masks,
            device=pred.device,
            loss_mode="epoch_weighted_mean",
            loss_epoch_weights=weights,
        )

        self.assertTrue(torch.isnan(out["epoch_losses"]["response"]).item())
        self.assertAlmostEqual(float(out["weighted_loss"].item()), 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
