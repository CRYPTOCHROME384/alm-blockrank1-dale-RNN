import os
import sys
import unittest

import torch


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
CURRENT_RNN_DIR = os.path.join(PROJECT_ROOT, "current_rnn")
if CURRENT_RNN_DIR not in sys.path:
    sys.path.append(CURRENT_RNN_DIR)

from delay_dynamics import (
    DELAY_SHAPE_LOSS_REL_GLOBAL,
    DELAY_SHAPE_LOSS_REL_GROUP,
    build_delay_group_info,
    center_over_delay_time,
    compute_delay_shape_loss_bundle,
    compute_delay_component_errors,
    compute_delay_statistics,
    compute_relative_ac_loss,
    delay_centered_shape_loss,
    delay_derivative_loss,
    make_constant_delay_baseline,
)


class DelayDynamicsTests(unittest.TestCase):
    def test_delay_centered_shape_loss_ranks_good_shifted_flat(self):
        delay_mask = torch.tensor([False, True, True, True, False]).numpy()
        target = torch.tensor(
            [
                [[0.0], [1.0], [2.0], [1.0], [0.0]],
                [[0.0], [2.0], [3.0], [2.0], [0.0]],
            ],
            dtype=torch.float32,
        )
        pred_good = torch.tensor(
            [
                [[0.0], [1.1], [1.9], [1.0], [0.0]],
                [[0.0], [1.9], [3.1], [2.1], [0.0]],
            ],
            dtype=torch.float32,
        )
        pred_shifted = pred_good + 5.0
        pred_flat = torch.tensor(
            [
                [[0.0], [1.4], [1.4], [1.4], [0.0]],
                [[0.0], [2.4], [2.4], [2.4], [0.0]],
            ],
            dtype=torch.float32,
        )

        loss_good = float(delay_centered_shape_loss(target, pred_good, delay_mask).item())
        loss_shifted = float(delay_centered_shape_loss(target, pred_shifted, delay_mask).item())
        loss_flat = float(delay_centered_shape_loss(target, pred_flat, delay_mask).item())

        self.assertLess(loss_good, loss_flat)
        self.assertLess(loss_shifted, loss_flat)
        self.assertAlmostEqual(loss_good, loss_shifted, places=4)

    def test_centering_only_over_time_dimension(self):
        delay_mask = torch.tensor([False, True, True, True, False]).numpy()
        x = torch.tensor(
            [
                [[0.0, 10.0], [1.0, 11.0], [2.0, 13.0], [3.0, 15.0], [4.0, 17.0]],
                [[5.0, 20.0], [6.0, 19.0], [8.0, 18.0], [10.0, 17.0], [12.0, 16.0]],
            ],
            dtype=torch.float32,
        )
        out = center_over_delay_time(x, delay_mask)
        centered = out["centered"]

        self.assertEqual(tuple(centered.shape), (2, 3, 2))
        mean_over_time = centered.mean(dim=1)
        self.assertTrue(torch.allclose(mean_over_time, torch.zeros_like(mean_over_time), atol=1e-6))
        self.assertLess(float(out["max_abs_mean"].item()), 1e-6)

        # Different conditions and neurons should preserve distinct centered trajectories.
        self.assertFalse(torch.allclose(centered[0, :, 0], centered[1, :, 0]))
        self.assertFalse(torch.allclose(centered[0, :, 0], centered[0, :, 1]))

    def test_constant_delay_baseline_zeroes_delay_std_and_matches_shape_loss(self):
        delay_mask = torch.tensor([False, True, True, True, False]).numpy()
        target = torch.tensor(
            [
                [[0.0], [1.0], [2.0], [3.0], [0.0]],
                [[0.0], [4.0], [5.0], [6.0], [0.0]],
            ],
            dtype=torch.float32,
        )
        pred_const = make_constant_delay_baseline(target, delay_mask)
        stats = compute_delay_statistics(pred_const, delay_mask)
        errs = compute_delay_component_errors(target, pred_const, delay_mask)

        self.assertTrue(torch.allclose(stats["std"], torch.zeros_like(stats["std"]), atol=1e-6))
        self.assertTrue(torch.allclose(errs["shape_loss"], compute_delay_statistics(target, delay_mask)["centered_energy"], atol=1e-6))

    def test_delay_derivative_loss_zero_for_shift_only(self):
        delay_mask = torch.tensor([False, True, True, True, False]).numpy()
        target = torch.tensor(
            [
                [[0.0], [1.0], [2.0], [1.0], [0.0]],
                [[0.0], [2.0], [3.0], [2.0], [0.0]],
            ],
            dtype=torch.float32,
        )
        pred_shifted = target + 3.0
        deriv_loss = float(delay_derivative_loss(target, pred_shifted, delay_mask).item())
        self.assertAlmostEqual(deriv_loss, 0.0, places=6)

    def test_relative_ac_loss_unit_curve(self):
        delay_mask = torch.tensor([True, True, True, True, True]).numpy()
        target = torch.tensor([[[0.40], [0.45], [0.50], [0.47], [0.43]]], dtype=torch.float32)
        pred_good = torch.tensor([[[0.41], [0.46], [0.51], [0.48], [0.44]]], dtype=torch.float32)
        pred_flat = torch.tensor([[[0.45], [0.45], [0.45], [0.45], [0.45]]], dtype=torch.float32)
        pred_shifted = torch.tensor([[[1.40], [1.45], [1.50], [1.47], [1.43]]], dtype=torch.float32)

        raw_good = float(delay_centered_shape_loss(target, pred_good, delay_mask).item())
        raw_flat = float(delay_centered_shape_loss(target, pred_flat, delay_mask).item())
        raw_shifted = float(delay_centered_shape_loss(target, pred_shifted, delay_mask).item())
        rel_good, _ = compute_relative_ac_loss(target=target, pred=pred_good, delay_mask=delay_mask, scale_mode="global_ac_energy")
        rel_flat, _ = compute_relative_ac_loss(target=target, pred=pred_flat, delay_mask=delay_mask, scale_mode="global_ac_energy")
        rel_shifted, _ = compute_relative_ac_loss(target=target, pred=pred_shifted, delay_mask=delay_mask, scale_mode="global_ac_energy")

        self.assertLess(raw_good, raw_flat)
        self.assertLess(raw_shifted, raw_flat)
        self.assertLess(float(rel_good.item()), 0.1)
        self.assertLess(float(rel_shifted.item()), 1e-6)
        self.assertAlmostEqual(float(rel_flat.item()), 1.0, places=5)

    def test_relative_ac_group_loss_averages_groups(self):
        delay_mask = torch.tensor([True, True, True, True, True]).numpy()
        target = torch.tensor(
            [
                [[0.40, 0.20], [0.45, 0.25], [0.50, 0.30], [0.47, 0.27], [0.43, 0.23]],
            ],
            dtype=torch.float32,
        )
        pred_flat = torch.tensor(
            [
                [[0.45, 0.25], [0.45, 0.25], [0.45, 0.25], [0.45, 0.25], [0.45, 0.25]],
            ],
            dtype=torch.float32,
        )
        group_index = [0, 1]
        group_names = ["E", "I"]
        loss, details = compute_relative_ac_loss(
            target=target,
            pred=pred_flat,
            delay_mask=delay_mask,
            scale_mode="group_ac_energy",
            group_mode="celltype",
            group_index=group_index,
            group_names=group_names,
        )
        self.assertAlmostEqual(float(loss.item()), 1.0, places=5)
        self.assertEqual(details["n_groups_used"], 2)

    def test_delay_shape_loss_bundle_supports_relative_modes(self):
        delay_mask = torch.tensor([True, True, True, True, True]).numpy()
        target = torch.tensor([[[0.40], [0.45], [0.50], [0.47], [0.43]]], dtype=torch.float32)
        pred_flat = torch.tensor([[[0.45], [0.45], [0.45], [0.45], [0.45]]], dtype=torch.float32)
        out_global = compute_delay_shape_loss_bundle(
            target=target,
            pred=pred_flat,
            delay_mask=delay_mask,
            loss_type=DELAY_SHAPE_LOSS_REL_GLOBAL,
        )
        out_group = compute_delay_shape_loss_bundle(
            target=target,
            pred=pred_flat,
            delay_mask=delay_mask,
            loss_type=DELAY_SHAPE_LOSS_REL_GROUP,
            group_mode="all",
        )
        self.assertAlmostEqual(float(out_global["loss"].item()), 1.0, places=5)
        self.assertAlmostEqual(float(out_group["loss"].item()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
