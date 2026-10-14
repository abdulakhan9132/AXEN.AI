"""Numerical AdamW checks, CUDA AMP, and exact next-update checkpoint parity."""
import math
import random

import numpy as np
import pytest
import torch

from axen.model.foundation import AXENFoundationModel, FoundationModelConfig
from scripts.train_aexn_gpt import (create_scheduler, get_learning_rate,
    load_checkpoint, save_checkpoint, training_step)
from scripts.tune_aexn_gpt import run_experiment_trial


@pytest.fixture
def model():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    return AXENFoundationModel(FoundationModelConfig(vocab_size=32, context_length=8,
        embedding_dim=16, num_layers=1, num_heads=2, ff_dim=32, dropout=0, device="cpu"))


def batch(device="cpu"):
    x = torch.arange(8, device=device).repeat(2, 1)
    return x, (x + 1) % 8


def test_official_adamw_updates_weights_and_excludes_bias_norm(model):
    optimizer = model.configure_optimizers(amsgrad=True)
    assert isinstance(optimizer, torch.optim.AdamW)
    groups = {id(p): g["weight_decay"] for g in optimizer.param_groups for p in g["params"]}
    for name, param in model.named_parameters():
        assert groups[id(param)] == (0.01 if param.ndim >= 2 else 0)
    original = model.lm_head.weight.detach().clone()
    training_step(model, optimizer, *batch())
    assert not torch.equal(original, model.lm_head.weight)
    assert all("max_exp_avg_sq" in s for s in optimizer.state.values())


def test_moments_bias_correction_and_adaptive_update_match_reference():
    p = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
    optimizer = torch.optim.AdamW([p], lr=0.03, betas=(0.8, 0.9), eps=1e-8, weight_decay=0.1)
    expected, momentum, variance = p.detach().clone(), torch.zeros_like(p), torch.zeros_like(p)
    for step, grad in enumerate((torch.tensor([0.2, -0.4]), torch.tensor([-0.1, 0.5])), 1):
        grad = grad.to(torch.float64)
        momentum = 0.8 * momentum + 0.2 * grad
        variance = 0.9 * variance + 0.1 * grad.square()
        expected = expected * (1 - 0.03 * 0.1)
        expected -= 0.03 * (momentum / (1 - 0.8 ** step)) / ((variance / (1 - 0.9 ** step)).sqrt() + 1e-8)
        p.grad = grad.clone()
        optimizer.step()
        state = optimizer.state[p]
        torch.testing.assert_close(state["exp_avg"], momentum)
        torch.testing.assert_close(state["exp_avg_sq"], variance)
        torch.testing.assert_close(p, expected, rtol=1e-12, atol=1e-12)


def test_weight_decay_does_not_enter_moments():
    p = torch.nn.Parameter(torch.tensor([2.]))
    optimizer = torch.optim.AdamW([p], lr=0.1, weight_decay=0.2)
    p.grad = torch.zeros_like(p)
    optimizer.step()
    torch.testing.assert_close(p, torch.tensor([1.96]))
    assert optimizer.state[p]["exp_avg"].item() == 0
    assert optimizer.state[p]["exp_avg_sq"].item() == 0


def test_cross_entropy_gradients_match_analytical_derivative():
    logits = torch.tensor([[0.2, -0.4, 1.]], dtype=torch.float64, requires_grad=True)
    torch.nn.functional.cross_entropy(logits, torch.tensor([2])).backward()
    expected = logits.detach().softmax(-1)
    expected[0, 2] -= 1
    torch.testing.assert_close(logits.grad, expected)


def test_gradient_clipping_is_applied_before_update(model):
    result = training_step(model, model.configure_optimizers(), *batch(), grad_clip=0.01)
    norm = torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]).norm()
    assert result["grad_norm"] > 0.01
    assert norm <= 0.010001


def test_scheduler_warmup_cosine_and_update_order(model):
    optimizer = model.configure_optimizers(learning_rate=0.01)
    scheduler = create_scheduler(optimizer, 2, 6, 0.01, 0.001)
    rates = [training_step(model, optimizer, *batch(), scheduler=scheduler)["lr"] for _ in range(6)]
    assert rates[:3] == pytest.approx([0.005, 0.01, 0.01])
    assert rates[3] > rates[4] > rates[5]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    assert get_learning_rate(100, 2, 6, 0.01, 0.001) == 0.001


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_checkpoint_restores_exact_next_update(model, tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model.to(device)
    optimizer = model.configure_optimizers(learning_rate=0.01, amsgrad=True)
    scheduler = create_scheduler(optimizer, 2, 8, 0.01, 0.001)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda", init_scale=128)
    training_step(model, optimizer, *batch(device), scheduler=scheduler, scaler=scaler)
    path = tmp_path / "resume.pt"
    save_checkpoint(path, model, optimizer, 1, 2.5, model.config.to_dict(), scheduler, scaler,
                    {"training": {"amsgrad": True}}, best_val_loss=2.0)
    expected_random = (torch.rand(2), np.random.random(), random.random())
    expected_metrics = training_step(model, optimizer, *batch(device), scheduler=scheduler, scaler=scaler)
    expected_weights = {name: param.detach().clone() for name, param in model.named_parameters()}
    loaded = AXENFoundationModel(model.config).to(device)
    opt_loaded = loaded.configure_optimizers(learning_rate=0.01, amsgrad=True)
    sched_loaded = create_scheduler(opt_loaded, 2, 8, 0.01, 0.001)
    scale_loaded = torch.amp.GradScaler("cuda", enabled=device == "cuda", init_scale=64)
    assert load_checkpoint(path, loaded, opt_loaded, torch.device(device), sched_loaded, scale_loaded) == (1, 2.0)
    torch.testing.assert_close(torch.rand(2), expected_random[0], rtol=0, atol=0)
    assert np.random.random() == expected_random[1]
    assert random.random() == expected_random[2]
    actual = training_step(loaded, opt_loaded, *batch(device), scheduler=sched_loaded, scaler=scale_loaded)
    assert actual == expected_metrics
    assert scale_loaded.state_dict() == scaler.state_dict()
    for name, param in loaded.named_parameters():
        torch.testing.assert_close(param, expected_weights[name], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_amp_finite_and_overflow_skips_scheduler(model):
    model.cuda()
    optimizer = model.configure_optimizers()
    scheduler = create_scheduler(optimizer, 2, 8, 0.0003, 0.00003)
    scaler = torch.amp.GradScaler("cuda", init_scale=128)
    for _ in range(3):
        result = training_step(model, optimizer, *batch("cuda"), scaler=scaler, scheduler=scheduler)
        assert math.isfinite(result["loss"]) and result["update_applied"]
        assert all(torch.isfinite(p).all() for p in model.parameters())
    original = model.lm_head.weight.detach().clone()
    epoch = scheduler.last_epoch
    hook = model.lm_head.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    try:
        result = training_step(model, optimizer, *batch("cuda"), scaler=scaler, scheduler=scheduler)
    finally:
        hook.remove()
    assert not result["update_applied"]
    assert scheduler.last_epoch == epoch
    torch.testing.assert_close(model.lm_head.weight, original, rtol=0, atol=0)


def test_small_dataset_training_loss_decreases(model):
    optimizer = model.configure_optimizers(learning_rate=0.02, weight_decay=0)
    losses = [training_step(model, optimizer, *batch())["loss"] for _ in range(25)]
    assert losses[-1] < losses[0] * 0.5


def test_legacy_checkpoint_still_loads(model, tmp_path):
    optimizer = model.configure_optimizers()
    path = tmp_path / "legacy.pt"
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "step": 3, "val_loss": 1.5}, path)
    scheduler = create_scheduler(optimizer, 2, 10, 0.0003, 0.00003)
    assert load_checkpoint(path, model, optimizer, scheduler=scheduler) == (3, 1.5)
    assert scheduler.last_epoch == 3
    assert optimizer.param_groups[0]["lr"] == pytest.approx(get_learning_rate(3, 2, 10, 0.0003, 0.00003))
