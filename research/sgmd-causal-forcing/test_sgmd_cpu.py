"""
CPU-only smoke test for the SGMD (Score Gradient Matching Distillation) losses
in model/sgmd.py, using tiny random tensors and mock score networks.

Run from the repo root (no GPU required):
    python tests/test_sgmd_cpu.py

What is verified:
1. _get_sigma is consistent with FlowMatchScheduler.add_noise (same lookup).
2. The generator loss (L_Fisher + lambda * L_NR) and the critic loss
   (lambda * L_RC) both compute finite values.
3. Generator update: nonzero grads on generator params (the gradient flows
   through the fake score network's INPUT Jacobian -- under DMD's no_grad
   surrogate this path would not exist), and ZERO grads on the real-score
   path (its params are left requires_grad=True on purpose: the no_grad +
   detached-input blocking must stop gradients even for trainable params).
4. Fake-score update: zero grads on the generator (rollout under no_grad).
5. The trainer's two-backward ordering discards the fake-score param grads
   deposited by the generator backward (critic zero_grad before critic
   backward), so the psi update contains only residual-contraction grads.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

if not torch.cuda.is_available():
    # wan/modules/t5.py and demo_utils/memory.py call torch.cuda.current_device()
    # at import time; return a plain index so the import chain works on CPU.
    torch.cuda.current_device = lambda: 0

from model.sgmd import SGMD
from utils.scheduler import FlowMatchScheduler

BATCH_SIZE, NUM_FRAME, CHANNELS, HEIGHT, WIDTH = 2, 3, 4, 6, 6
SHAPE = [BATCH_SIZE, NUM_FRAME, CHANNELS, HEIGHT, WIDTH]


class MockScoreNetwork(nn.Module):
    """Tiny stand-in for WanDiffusionWrapper: returns (flow_pred, x0_pred)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(CHANNELS, CHANNELS)

    def forward(self, noisy_image_or_video, conditional_dict, timestep):
        x = noisy_image_or_video.permute(0, 1, 3, 4, 2)  # [B, F, H, W, C]
        x0_pred = self.proj(x).permute(0, 1, 4, 2, 3)
        return None, x0_pred


class MockGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(CHANNELS, CHANNELS)

    def forward(self, noise):
        x = noise.permute(0, 1, 3, 4, 2)
        return self.proj(x).permute(0, 1, 4, 2, 3)


def build_model():
    """Build an SGMD instance without loading any Wan checkpoints."""
    model = SGMD.__new__(SGMD)
    nn.Module.__init__(model)

    model.device = torch.device("cpu")
    model.generator = MockGenerator()
    model.fake_score = MockScoreNetwork()
    # Deliberately trainable: the test must prove the no_grad + detached-input
    # blocking, not merely a requires_grad=False flag (production freezes it).
    model.real_score = MockScoreNetwork()

    # Same construction as WanDiffusionWrapper.__init__ (timestep_shift 5.0).
    model.scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    model.scheduler.set_timesteps(1000, training=True)

    # Hyperparameters as in SGMD.__init__ with the 1-step framewise config.
    model.num_frame_per_block = 1
    model.same_step_across_blocks = True
    model.num_training_frames = NUM_FRAME
    model.independent_first_frame = False
    model.num_train_timestep = 1000
    model.min_step = 20
    model.max_step = 980
    model.real_guidance_scale = 3.0
    model.fake_guidance_scale = 0.0
    model.timestep_shift = 5.0
    model.ts_schedule = False
    model.ts_schedule_max = False
    model.min_score_timestep = 0
    model.sgmd_lambda = 0.1

    # Bypass the backward-simulation rollout: run the mock generator on fixed
    # noise. critic_loss wraps this call in torch.no_grad ("theta detached").
    def fake_run_generator(image_or_video_shape, conditional_dict, initial_latent=None):
        noise = torch.randn(image_or_video_shape)
        pred_image = model.generator(noise)
        return pred_image, None, None, None

    model._run_generator = fake_run_generator
    return model


def zero_all_grads(model):
    for module in (model.generator, model.fake_score, model.real_score):
        for p in module.parameters():
            p.grad = None


def conditional_dicts():
    return {"prompt_embeds": None}, {"prompt_embeds": None}


def test_sigma_consistent_with_add_noise():
    model = build_model()
    timestep = torch.tensor([[100.0, 500.0, 900.0]]).repeat(BATCH_SIZE, 1)
    sigma = model._get_sigma(timestep)
    assert sigma.shape == (BATCH_SIZE, NUM_FRAME, 1, 1, 1)

    # add_noise(ones, zeros, t) = (1 - sigma) * 1 + sigma * 0 = 1 - sigma
    ones = torch.ones(BATCH_SIZE * NUM_FRAME, CHANNELS, HEIGHT, WIDTH)
    xt = model.scheduler.add_noise(
        ones, torch.zeros_like(ones), timestep.flatten(0, 1))
    expected = (1 - sigma).expand(-1, -1, CHANNELS, HEIGHT, WIDTH).flatten(0, 1)
    assert torch.allclose(xt, expected, atol=1e-6), \
        "_get_sigma disagrees with the sigma used by scheduler.add_noise"
    print("PASS: _get_sigma matches scheduler.add_noise")


def test_generator_update():
    torch.manual_seed(0)
    model = build_model()
    cond, uncond = conditional_dicts()
    zero_all_grads(model)

    loss, log_dict = model.generator_loss(
        image_or_video_shape=SHAPE,
        conditional_dict=cond,
        unconditional_dict=uncond,
        clean_latent=None
    )

    # Finite loss and finite log components (trainer logging contract).
    assert torch.isfinite(loss).item(), f"generator loss not finite: {loss}"
    assert "dmdtrain_gradient_norm" in log_dict  # read by trainer/distillation.py
    for key in ("dmdtrain_gradient_norm", "sgmd_fisher_loss", "sgmd_nr_loss"):
        assert torch.isfinite(log_dict[key]).all(), f"{key} not finite"

    loss.backward()

    # Nonzero grads on every generator param: gradient reached theta through
    # the fake network's input Jacobian (DMD's no_grad surrogate has no such
    # path -- with these mocks it would leave the generator grads at zero).
    for name, p in model.generator.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, \
            f"generator param {name} has no grad"

    # ZERO grads on the real-score path, even though its params are trainable:
    # torch.no_grad + detached input block the teacher entirely.
    for name, p in model.real_score.named_parameters():
        assert p.grad is None, f"real_score param {name} received grad {p.grad}"

    # The fake score DOES accumulate (to-be-discarded) param grads here, which
    # is exactly the shared-graph behavior of the paper's Appendix B pseudocode;
    # test_two_backward_ordering verifies the trainer discards them.
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.fake_score.parameters()), \
        "expected the generator backward to traverse the fake score network"
    print(f"PASS: generator update (loss={loss.item():.4e}, "
          f"fisher={log_dict['sgmd_fisher_loss'].item():.4e}, "
          f"nr={log_dict['sgmd_nr_loss'].item():.4e})")


def test_critic_update():
    torch.manual_seed(1)
    model = build_model()
    cond, uncond = conditional_dicts()
    zero_all_grads(model)

    loss, log_dict = model.critic_loss(
        image_or_video_shape=SHAPE,
        conditional_dict=cond,
        unconditional_dict=uncond,
        clean_latent=None
    )
    assert torch.isfinite(loss).item(), f"critic loss not finite: {loss}"
    assert "critic_timestep" in log_dict

    loss.backward()

    # Nonzero grads on the fake score (it is being trained)...
    for name, p in model.fake_score.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, \
            f"fake_score param {name} has no grad"
    # ...zero grads on the generator (rollout under no_grad, "theta detached")
    # and none on the real score (not involved in L_RC).
    for name, p in model.generator.named_parameters():
        assert p.grad is None, f"generator param {name} received grad {p.grad}"
    for name, p in model.real_score.named_parameters():
        assert p.grad is None, f"real_score param {name} received grad {p.grad}"

    # The lambda knob multiplies L_RC (eq. 22).
    torch.manual_seed(2)
    loss_low, _ = model.critic_loss(
        image_or_video_shape=SHAPE, conditional_dict=cond,
        unconditional_dict=uncond, clean_latent=None)
    model.sgmd_lambda = 0.2
    torch.manual_seed(2)
    loss_high, _ = model.critic_loss(
        image_or_video_shape=SHAPE, conditional_dict=cond,
        unconditional_dict=uncond, clean_latent=None)
    assert torch.allclose(loss_high, 2 * loss_low), \
        "critic loss does not scale linearly with sgmd_lambda"
    print(f"PASS: critic update (loss={loss.item():.4e})")


def test_two_backward_ordering():
    """Simulate one trainer iteration (trainer/distillation.py train loop with
    dfake_gen_update_ratio=1) and verify Algorithm 1's update-level detach
    semantics: theta steps on grad(L_Fisher + lambda*L_NR), psi steps on
    grad(lambda*L_RC) only."""
    torch.manual_seed(3)
    model = build_model()
    cond, uncond = conditional_dicts()
    generator_optimizer = torch.optim.SGD(model.generator.parameters(), lr=0.1)
    critic_optimizer = torch.optim.SGD(model.fake_score.parameters(), lr=0.1)

    # --- Generator update (backward #1) ---
    generator_optimizer.zero_grad(set_to_none=True)
    gen_loss, _ = model.generator_loss(
        image_or_video_shape=SHAPE, conditional_dict=cond,
        unconditional_dict=uncond, clean_latent=None)
    gen_loss.backward()
    theta_before = [p.detach().clone() for p in model.generator.parameters()]
    generator_optimizer.step()
    assert any(not torch.equal(p.detach(), b) for p, b in
               zip(model.generator.parameters(), theta_before)), \
        "generator params did not move on the generator update"

    # --- Critic update (backward #2) ---
    # The trainer zeroes critic grads BEFORE the critic backward, discarding
    # the psi grads leaked by backward #1 ("update generator with psi detached").
    critic_optimizer.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in model.fake_score.parameters()), \
        "leaked fake-score grads were not discarded before the critic update"

    theta_grads_after_gen = [None if p.grad is None else p.grad.detach().clone()
                             for p in model.generator.parameters()]
    critic_loss, _ = model.critic_loss(
        image_or_video_shape=SHAPE, conditional_dict=cond,
        unconditional_dict=uncond, clean_latent=None)
    critic_loss.backward()

    # The critic backward adds nothing to the generator ("theta detached").
    for p, g in zip(model.generator.parameters(), theta_grads_after_gen):
        if g is None:
            assert p.grad is None
        else:
            assert torch.equal(p.grad, g), \
                "critic backward modified generator grads"

    psi_before = [p.detach().clone() for p in model.fake_score.parameters()]
    critic_optimizer.step()
    assert any(not torch.equal(p.detach(), b) for p, b in
               zip(model.fake_score.parameters(), psi_before)), \
        "fake-score params did not move on the critic update"
    print("PASS: two-backward ordering (psi detached in gen update, "
          "theta detached in critic update)")


if __name__ == "__main__":
    test_sigma_consistent_with_add_noise()
    test_generator_update()
    test_critic_update()
    test_two_backward_ordering()
    print("ALL SGMD CPU SMOKE TESTS PASSED")
