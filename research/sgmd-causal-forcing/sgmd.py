from pipeline import SelfForcingTrainingPipeline
from typing import Optional, Tuple
import torch

from model.base import SelfForcingModel


class SGMD(SelfForcingModel):
    def __init__(self, args, device):
        """
        Initialize the SGMD (Score Gradient Matching Distillation) module
        (https://arxiv.org/abs/2605.30116).
        This class is self-contained and compute generator and fake score losses
        in the forward pass.

        Unlike DMD's reverse-KL surrogate (where the score difference is computed
        under torch.no_grad and injected through a detached MSE), SGMD plays a
        Fisher-divergence score-gradient game with two separate backwards:
        - Generator update: L_Fisher + sgmd_lambda * L_NR (eq. 21), where the
          gradient reaches the generator THROUGH the fake score network's input
          Jacobian. The fake score's parameters are "detached" at the update
          level: the generator optimizer holds only generator parameters, and
          any parameter gradients this backward deposits on the fake score are
          zeroed by the trainer (critic_optimizer.zero_grad) before the critic
          backward runs. The real (teacher) score path is doubly blocked
          (torch.no_grad + detached input).
        - Fake score update: sgmd_lambda * L_RC (eq. 22, "residual contraction")
          on a fresh generator rollout under torch.no_grad ("theta detached").
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: SelfForcingTrainingPipeline = None

        # Step 2: Initialize all sgmd hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)
        # Weight of the negative-residual / residual-contraction pair
        # (lambda in eq. 21-22; paper default/best 0.1, >=0.5 fails to converge).
        self.sgmd_lambda = getattr(args, "sgmd_lambda", 0.1)
        self.sgmd_fisher_normalization = getattr(args, "sgmd_fisher_normalization", "none")
        if self.sgmd_fisher_normalization not in ("none", "batch_mean"):
            raise ValueError(f"unknown sgmd_fisher_normalization: {self.sgmd_fisher_normalization!r}")

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

    def _sample_score_timestep(
        self,
        batch_size: int,
        num_frame: int,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0
    ) -> torch.Tensor:
        """
        Sample the score-evaluation timestep with the same schedule as DMD:
        uniform in [min, max), shift-warped, clamped to [min_step, max_step].
        Output: a tensor with shape [B, F] (float after the shift warp).
        """
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            batch_size,
            num_frame,
            self.num_frame_per_block,
            uniform_timestep=True
        )

        if self.timestep_shift > 1:
            timestep = self.timestep_shift * \
                (timestep / 1000) / \
                (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
        timestep = timestep.clamp(self.min_step, self.max_step)
        return timestep

    def _get_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Look up the flow-matching sigma_t for each (already shift-warped) timestep,
        using the same nearest-timestep indexing as FlowMatchScheduler.add_noise,
        so that c(t) is consistent with the actually-applied noising:
        x_t = (1 - sigma_t) * x0 + sigma_t * noise, i.e. alpha_t = 1 - sigma_t.
        Input:
            - timestep: a tensor with shape [B, F].
        Output:
            - sigma: a tensor with shape [B, F, 1, 1, 1] for broadcasting.
        """
        batch_size, num_frame = timestep.shape
        timesteps_table = self.scheduler.timesteps.to(timestep.device)
        sigmas_table = self.scheduler.sigmas.to(timestep.device)
        timestep_id = torch.argmin(
            (timesteps_table.unsqueeze(0) - timestep.flatten(0, 1).unsqueeze(1)).abs(), dim=1)
        sigma = sigmas_table[timestep_id].unflatten(0, (batch_size, num_frame))
        return sigma.reshape(batch_size, num_frame, 1, 1, 1)

    def compute_score_gradient_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the SGMD generator loss L_Fisher + sgmd_lambda * L_NR
        (eq. 21 in https://arxiv.org/abs/2605.30116):
        - L_Fisher = 0.5 * c(t) * ||x_fake - x_real||^2 with c(t) = alpha_t^2 / sigma_t^4
          (the exact score-space Fisher divergence rewritten in x0 space; x_real is
          the CFG'd frozen teacher prediction, fully detached).
        - L_NR = -0.5 * ||sg[x0] - x_fake||^2 ("negative residual"): with the direct
          x0 path stop-gradded, descending this term pulls x0 toward the fake
          prediction through the fake network's input Jacobian.
        The fake score forward is NOT wrapped in torch.no_grad: the input Jacobian
        d x_fake / d x_t * d x_t / d x0 is the only path carrying gradient to the
        generator. Parameter gradients deposited on the fake score here are
        discarded by the trainer's optimizer ordering (see class docstring).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - sgmd_loss: a scalar tensor representing the SGMD generator loss.
            - sgmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        # Step 1: Randomly sample timestep based on the given schedule and corresponding noise.
        # Unlike DMD, the noising is NOT under torch.no_grad: the path
        # x0 -> x_t -> fake score -> loss must keep the autograd graph.
        timestep = self._sample_score_timestep(
            batch_size, num_frame, denoised_timestep_from, denoised_timestep_to)

        noise = torch.randn_like(image_or_video)
        noisy_latent = self.scheduler.add_noise(
            image_or_video.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))

        # Step 2: Compute the fake score's x0 prediction, keeping the graph
        # through its input (this is the theta-gradient carrier).
        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_latent,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        if self.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_latent,
                conditional_dict=unconditional_dict,
                timestep=timestep
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        # Step 3: Compute the real score with cfg (https://arxiv.org/abs/2207.12598).
        # The teacher path is doubly blocked: torch.no_grad AND a detached input
        # ("teacher stop-gradient Fisher", eq. 13).
        with torch.no_grad():
            detached_noisy_latent = noisy_latent.detach()
            _, pred_real_image_cond = self.real_score(
                noisy_image_or_video=detached_noisy_latent,
                conditional_dict=conditional_dict,
                timestep=timestep
            )

            _, pred_real_image_uncond = self.real_score(
                noisy_image_or_video=detached_noisy_latent,
                conditional_dict=unconditional_dict,
                timestep=timestep
            )

            pred_real_image = pred_real_image_cond + (
                pred_real_image_cond - pred_real_image_uncond
            ) * self.real_guidance_scale

        # Step 4: Compute the two loss terms (in double precision, like DMD).
        sigma_t = self._get_sigma(timestep).double()
        alpha_t = 1 - sigma_t
        fisher_weight = alpha_t ** 2 / sigma_t ** 4

        delta = pred_fake_image.double() - pred_real_image.double()
        residual = original_latent.detach().double() - pred_fake_image.double()

        # Optional Fisher-weight normalization. The exact metric c(t)=alpha^2/sigma^4
        # spans ~1e10 over the sampled sigma range, so low-sigma draws dominate the
        # gradient scale across steps (implicated in the high-frequency dissolution
        # seen in the lambda=0.1 raw-weight pilot). "batch_mean" rescales the fisher
        # term by the mean magnitude of its own pointwise gradient (detached scalar),
        # bounding cross-step scale. NOTE: with batch_size=1 and one sigma per sample
        # this scalar is effectively per-sample; and normalization rebalances the
        # relative weight of the lambda*L_NR term (raw-mode lambda calibration does
        # not transfer). Default "none" reproduces the paper/pilot exactly.
        norm_factor = None
        if getattr(self, "sgmd_fisher_normalization", "none") == "batch_mean":
            weighted_mag = fisher_weight * delta.abs()
            if gradient_mask is not None:
                norm_factor = weighted_mag[gradient_mask].mean().detach().clamp_min(1e-8)
            else:
                norm_factor = weighted_mag.mean().detach().clamp_min(1e-8)
            fisher_weight = fisher_weight / norm_factor

        if gradient_mask is not None:
            # Useless if we set always 21 latent frames
            fisher_loss = 0.5 * (fisher_weight * delta ** 2)[gradient_mask].mean()
            nr_loss = -0.5 * (residual ** 2)[gradient_mask].mean()
        else:
            fisher_loss = 0.5 * (fisher_weight * delta ** 2).mean()
            nr_loss = -0.5 * (residual ** 2).mean()

        sgmd_loss = fisher_loss + self.sgmd_lambda * nr_loss

        sgmd_log_dict = {
            # Pointwise gradient of the loss w.r.t. the fake prediction
            # (diagnostic analog of DMD's normalized KL grad; the key name is
            # kept so the trainer's logging works unchanged).
            "dmdtrain_gradient_norm": torch.mean(
                torch.abs(fisher_weight * delta + self.sgmd_lambda * residual)).detach(),
            "sgmd_fisher_loss": fisher_loss.detach(),
            "sgmd_nr_loss": nr_loss.detach(),
            "timestep": timestep.detach()
        }
        if norm_factor is not None:
            sgmd_log_dict["sgmd_fisher_norm_factor"] = norm_factor

        return sgmd_loss, sgmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the SGMD generator loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Unroll generator to obtain fake videos
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent
        )

        # Step 2: Compute the SGMD loss
        sgmd_loss, sgmd_log_dict = self.compute_score_gradient_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to
        )

        return sgmd_loss, sgmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the fake score with the SGMD
        residual-contraction loss sgmd_lambda * L_RC, where
        L_RC = 0.5 * ||sg[x0] - x_fake||^2 (eq. 22 in https://arxiv.org/abs/2605.30116):
        the fake score chases/tracks the generator's x0 in x0 space (no
        flow-matching denoising target, unlike DMD's critic). The generator
        rollout runs under torch.no_grad ("theta detached"); only the fake score
        forward builds a graph. The real score is not involved.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the critic loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # Step 1: Run generator on backward simulated noisy input
        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent
            )

        # Step 2: Compute the fake prediction on a freshly noised sample
        critic_timestep = self._sample_score_timestep(
            image_or_video_shape[0], image_or_video_shape[1],
            denoised_timestep_from, denoised_timestep_to)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )

        # Step 3: Residual contraction loss (x0-space MSE toward the frozen
        # generator sample; in double precision, like the generator loss)
        residual = generated_image.double() - pred_fake_image.double()
        rc_loss = self.sgmd_lambda * 0.5 * (residual ** 2).mean()

        # Step 4: Debugging Log
        critic_log_dict = {
            "critic_timestep": critic_timestep.detach()
        }

        return rc_loss, critic_log_dict
