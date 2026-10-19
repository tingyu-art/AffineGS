import math
import unittest

import torch

from methods.affinegs.gaussian_flow.indexing import (
    GAUSSIAN_VALUE_THRESHOLD,
    GaussianSpatialIndexer,
)
from methods.affinegs.gaussian_flow.model import GaussianVelocityField_Tiled
from methods.affinegs.gaussian_flow.renderer import HAS_CUDA_RENDERER


def make_model(device="cpu", jacobian_mode="full", enable_jacobian=True):
    torch.manual_seed(2026)
    model = GaussianVelocityField_Tiled(
        4, 19, 23, 8, device=device,
        enable_jacobian=enable_jacobian, jacobian_mode=jacobian_mode,
    )
    with torch.no_grad():
        model.mu.copy_(model.mu.new_tensor([
            [2.3, 3.7], [10.6, 8.2], [17.2, 20.1], [5.2, 18.4],
        ]))
        model.log_scale.copy_(model.mu.new_tensor([
            [1.3, 2.6], [2.2, 1.4], [2.8, 1.8], [1.5, 2.3],
        ]).log())
        model.rotation.copy_(model.mu.new_tensor([[0.2], [-0.3], [0.7], [1.1]]))
        model.opacity_logit.copy_(model.mu.new_tensor([[0.0], [-1.0], [1.0], [0.3]]))
    return model


def dense_reference(model, coordinates=None, epsilon=1e-8):
    """Evaluate every Gaussian at every point without using the tile indexer."""
    if coordinates is None:
        h, w = torch.meshgrid(
            torch.arange(model.H, device=model.mu.device, dtype=model.mu.dtype),
            torch.arange(model.W, device=model.mu.device, dtype=model.mu.dtype),
            indexing="ij",
        )
        coordinates = torch.stack((h, w), dim=-1)
    inverse, rotation, _ = model.compute_covariance_inv_and_R()
    delta = coordinates.unsqueeze(0) - model.mu[:, None, None, :]
    distance = torch.einsum("nhwi,nij,nhwj->nhw", delta, inverse, delta)
    response = torch.exp(-0.5 * distance)
    response = torch.where(response >= GAUSSIAN_VALUE_THRESHOLD, response, 0.0)
    weights = response * model.opacity_logit.sigmoid().view(-1, 1, 1)
    velocity = torch.matmul(rotation, model.local_velocity.unsqueeze(-1)).squeeze(-1)
    velocity = velocity[:, None, None, :]
    jacobian = model.build_jacobian_matrix(rotation)
    if jacobian is not None:
        velocity = velocity + torch.einsum("nij,nhwj->nhwi", jacobian, delta)
    numerator = (weights[..., None] * velocity).sum(dim=0).permute(2, 0, 1)
    denominator = weights.sum(dim=0).unsqueeze(0)
    return numerator / (denominator + epsilon), denominator


class GaussianTruncationTests(unittest.TestCase):
    def test_initialization_uses_full_image_and_independent_uniform_scales(self):
        torch.manual_seed(2026)
        model = GaussianVelocityField_Tiled(
            8192, 120, 320, 8, device="cpu", enable_jacobian=False,
            init_scale_h=4.0, init_scale_w=9.0,
        )
        centers = (model.mu.detach() + 16) / model.mu.new_tensor([152, 352])
        self.assertTrue(((centers >= 0) & (centers < 1)).all().item())
        self.assertGreater(model.mu[:, 1].max().item(), 0.9 * model.W)
        scales = model.log_scale.detach().exp()
        normalized = (scales - 1) / scales.new_tensor([3, 8])
        self.assertTrue(((normalized >= 0) & (normalized < 1)).all().item())
        for values in (centers, normalized):
            torch.testing.assert_close(values.mean(0), values.new_full((2,), 0.5), rtol=0, atol=0.02)
        torch.testing.assert_close(
            normalized.std(0, unbiased=False),
            normalized.new_full((2,), 1 / math.sqrt(12)), rtol=0, atol=0.01,
        )
        centered = normalized - normalized.mean(0)
        self.assertLess(abs((centered[:, 0] * centered[:, 1]).mean().item()), 0.005)

    def test_cutoff_is_applied_before_opacity_and_has_zero_tail_gradient(self):
        model = GaussianVelocityField_Tiled(1, 32, 32, 8, device="cpu", enable_jacobian=False)
        with torch.no_grad():
            model.mu.fill_(16)
            model.log_scale.fill_(math.log(3))
            model.rotation.zero_()
            model.opacity_logit.fill_(math.log(0.1 / 0.9))
        _, weights, _, _ = model()
        self.assertEqual(weights[0, 16, 28].item(), 0.0)
        self.assertGreater(weights[0, 16, 27].item(), 0.0)
        self.assertLess(weights[0, 16, 27].item(), GAUSSIAN_VALUE_THRESHOLD)
        expected = 0.1 * math.exp(-0.5 * 11**2 / (3**2 + 1e-6))
        self.assertAlmostEqual(weights[0, 16, 27].item(), expected, places=9)
        gradients = torch.autograd.grad(
            weights[0, 16, 28],
            (model.mu, model.log_scale, model.rotation, model.opacity_logit),
        )
        for gradient in gradients:
            self.assertEqual(torch.count_nonzero(gradient).item(), 0)

    def test_indexer_covers_threshold_support(self):
        model = make_model()
        model.rebuild_indexer()
        indexer = model._indexer_cache
        _, rotation, scales = model.compute_covariance_inv_and_R()
        self.assertEqual(indexer.value_threshold, 5e-4)
        self.assertAlmostEqual(indexer.support_radius, math.sqrt(-2 * math.log(5e-4)))
        for gid in range(model.num_gaussians):
            inverse, _, _ = model.compute_covariance_inv_and_R()
            for h in range(model.H):
                for w in range(model.W):
                    delta = model.mu.new_tensor([h, w]) - model.mu[gid]
                    response = torch.exp(-0.5 * delta @ inverse[gid] @ delta)
                    if response.item() < GAUSSIAN_VALUE_THRESHOLD:
                        continue
                    tile = (h // model.tile_size) * indexer.grid_w + w // model.tile_size
                    gids = indexer.cell_indices[indexer.cell_ptr[tile]:indexer.cell_ptr[tile + 1]]
                    self.assertTrue((gids == gid).any().item())
        for threshold in (0.0, -1.0, 1.0, float("nan")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                GaussianSpatialIndexer(
                    model.mu.detach(), scales.detach(), model.rotation.detach(),
                    (model.H, model.W), model.tile_size, value_threshold=threshold,
                )

    def test_native_and_higher_resolution_match_dense_reference(self):
        for enabled, mode in ((False, "full"), (True, "full"), (True, "trace_free")):
            with self.subTest(enabled=enabled, mode=mode):
                model = make_model(enable_jacobian=enabled, jacobian_mode=mode)
                expected_flow, expected_weights = dense_reference(model)
                flow, weights, _, _ = model(compute_derivatives=True)
                torch.testing.assert_close(weights, expected_weights, rtol=2e-5, atol=1e-7)
                torch.testing.assert_close(flow, expected_flow, rtol=2e-5, atol=1e-6)
                h, w = torch.meshgrid(torch.arange(38) / 2, torch.arange(46) / 2, indexing="ij")
                expected_flow, expected_weights = dense_reference(model, torch.stack((h, w), dim=-1))
                flow, weights, _, _ = model.render_at_resolution(38, 46)
                torch.testing.assert_close(weights, expected_weights, rtol=2e-5, atol=1e-7)
                torch.testing.assert_close(flow, expected_flow, rtol=2e-5, atol=1e-6)

    def test_analytic_divergence_and_vorticity_match_spatial_autograd(self):
        model = make_model()
        _, _, divergence, vorticity = model(compute_derivatives=True)
        for h, w in ((5, 5), (10, 12), (16, 18)):
            coordinates = model.mu.new_tensor([[[h, w]]], requires_grad=True)
            flow, _ = dense_reference(model, coordinates)
            grad_h = torch.autograd.grad(flow[0, 0, 0], coordinates, retain_graph=True)[0][0, 0]
            grad_w = torch.autograd.grad(flow[1, 0, 0], coordinates)[0][0, 0]
            torch.testing.assert_close(divergence[h, w], grad_h[0] + grad_w[1], rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(vorticity[h, w], grad_w[0] - grad_h[1], rtol=2e-5, atol=2e-6)

    def check_parameter_gradients(self, device):
        for enabled, mode in ((False, "full"), (True, "full"), (True, "trace_free")):
            with self.subTest(device=device, enabled=enabled, mode=mode):
                model = make_model(device=device, enable_jacobian=enabled, jacobian_mode=mode)
                reference = make_model(device=device, enable_jacobian=enabled, jacobian_mode=mode)
                epsilon = 1e-8 if device == "cuda" else 1e-31
                expected_flow, expected_weights = dense_reference(reference, epsilon=epsilon)
                flow, weights, _, _ = model()
                torch.testing.assert_close(weights, expected_weights, rtol=3e-5, atol=2e-7)
                torch.testing.assert_close(flow, expected_flow, rtol=3e-5, atol=2e-6)
                (flow.square().mean() + weights.square().mean()).backward()
                (expected_flow.square().mean() + expected_weights.square().mean()).backward()
                for (name, parameter), (_, expected_parameter) in zip(
                    model.named_parameters(), reference.named_parameters()
                ):
                    torch.testing.assert_close(
                        parameter.grad, expected_parameter.grad, rtol=4e-4, atol=4e-6,
                        msg=lambda message: f"{name}: {message}",
                    )

    def test_cpu_parameter_gradients(self):
        self.check_parameter_gradients("cpu")

    @unittest.skipUnless(torch.cuda.is_available() and HAS_CUDA_RENDERER, "CUDA renderer unavailable")
    def test_cuda_outputs_and_parameter_gradients(self):
        self.check_parameter_gradients("cuda")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
