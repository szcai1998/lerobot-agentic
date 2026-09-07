import shutil
import tempfile
from pathlib import Path

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def test_act_policy_architecture_and_parameter_delta():
    """
    Verifies that stock LeRobot 0.6.1 ACT architecture invariants hold:
    - ACT-B has no environment_state feature and exactly 51,573,639 parameters.
    - ACT-G has 13-DoF environment_state feature and exactly 51,581,319 parameters.
    - Parameter delta is exactly 7,680 (7,168 linear proj + 512 1D pos embedding).
    - 4 encoder layers, 1 decoder layer, 4 VAE encoder layers, shared ResNet-18 backbone.
    """
    cfg_b = ACTConfig(
        input_features={
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
        chunk_size=50,
        n_action_steps=10,
        n_decoder_layers=1,
        n_encoder_layers=4,
        n_vae_encoder_layers=4,
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        latent_dim=32,
        temporal_ensemble_coeff=None,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        device="cpu",
    )
    policy_b = ACTPolicy(cfg_b)

    cfg_g = ACTConfig(
        input_features={
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(13,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
        chunk_size=50,
        n_action_steps=10,
        n_decoder_layers=1,
        n_encoder_layers=4,
        n_vae_encoder_layers=4,
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        latent_dim=32,
        temporal_ensemble_coeff=None,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        device="cpu",
    )
    policy_g = ACTPolicy(cfg_g)

    # Configuration invariants
    assert cfg_b.env_state_feature is None
    assert cfg_g.env_state_feature is not None
    assert cfg_g.env_state_feature.shape == (13,)
    assert cfg_b.chunk_size == 50 and cfg_g.chunk_size == 50
    assert cfg_b.n_action_steps == 10 and cfg_g.n_action_steps == 10
    assert cfg_b.n_decoder_layers == 1 and cfg_g.n_decoder_layers == 1
    assert cfg_b.temporal_ensemble_coeff is None and cfg_g.temporal_ensemble_coeff is None

    # Shared ResNet backbone check
    assert hasattr(policy_b.model, "backbone")
    assert hasattr(policy_g.model, "backbone")

    # Exact parameter counts
    params_b = sum(p.numel() for p in policy_b.parameters())
    params_g = sum(p.numel() for p in policy_g.parameters())
    assert params_b == 51_573_639, f"Expected 51,573,639 for ACT-B, got {params_b}"
    assert params_g == 51_581_319, f"Expected 51,581,319 for ACT-G, got {params_g}"
    assert params_g - params_b == 7_680, f"Expected 7,680 delta, got {params_g - params_b}"

    # Verify optimizer parameter grouping separates backbone parameters
    optim_groups = policy_g.get_optim_params()
    assert len(optim_groups) == 2
    backbone_group = next(g for g in optim_groups if g.get("lr") is not None)
    transformer_group = next(g for g in optim_groups if g.get("lr") is None)
    backbone_params = sum(p.numel() for p in backbone_group["params"])
    transformer_params = sum(p.numel() for p in transformer_group["params"])
    assert backbone_params == 11_166_912
    assert transformer_params == 40_414_407


def test_act_queue_semantics_and_reset():
    """
    Verifies that receding-horizon queue populates n_action_steps=10 actions,
    depletes over 10 consecutive select_action() calls, triggers re-inference
    on step 11, and flushes immediately on policy.reset().
    """
    cfg = ACTConfig(
        input_features={
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
        chunk_size=50,
        n_action_steps=10,
        n_decoder_layers=1,
        pretrained_backbone_weights=None,
        device="cpu",
    )
    policy = ACTPolicy(cfg)
    policy.eval()
    policy.reset()

    assert len(policy._action_queue) == 0

    batch = {
        "observation.images.top": torch.randn(1, 3, 480, 640),
        "observation.images.wrist": torch.randn(1, 3, 480, 640),
        "observation.state": torch.randn(1, 7),
    }

    # Step 1: Triggers inference (predicts 50 actions, queues n_action_steps=10, pops 1)
    act1 = policy.select_action(batch)
    assert act1.shape == (1, 7)
    assert len(policy._action_queue) == 9

    # Steps 2..10: Consumes remaining 9 queued actions without re-inference
    for _ in range(9):
        policy.select_action(batch)
    assert len(policy._action_queue) == 0

    # Step 11: Queue exhausted -> triggers new inference, queues 10, pops 1
    act11 = policy.select_action(batch)
    assert act11.shape == (1, 7)
    assert len(policy._action_queue) == 9

    # Reset flushes queue completely
    policy.reset()
    assert len(policy._action_queue) == 0


def test_amp_training_step_and_grad_accum():
    """
    Verifies forward and backward training steps with gradient accumulation,
    AMP autocast, GradScaler, and unscaled gradient clipping.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ACTConfig(
        input_features={
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(13,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
        chunk_size=50,
        n_action_steps=10,
        n_decoder_layers=1,
        pretrained_backbone_weights=None,
        device=str(device),
    )
    policy = ACTPolicy(cfg).to(device)
    policy.train()

    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"), init_scale=1024.0)

    bs = 2
    grad_accum_steps = 2
    optimizer.zero_grad()

    for _micro_step in range(grad_accum_steps):
        batch = {
            "observation.images.top": torch.rand(bs, 3, 480, 640, device=device),
            "observation.images.wrist": torch.rand(bs, 3, 480, 640, device=device),
            "observation.state": torch.randn(bs, 7, device=device) * 0.1,
            "observation.environment_state": torch.randn(bs, 13, device=device) * 0.1,
            "action": torch.randn(bs, 50, 7, device=device) * 0.1,
            "action_is_pad": torch.zeros(bs, 50, dtype=torch.bool, device=device),
        }
        with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
            loss, loss_dict = policy(batch)
            loss = loss / grad_accum_steps

        assert torch.isfinite(loss)
        assert "l1_loss" in loss_dict
        assert "kld_loss" in loss_dict
        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
    assert torch.isfinite(grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()


def test_checkpoint_save_and_reload():
    """
    Verifies that a policy can be saved to disk with official save_pretrained
    and reloaded cleanly with matching weights and config.
    """
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        cfg = ACTConfig(
            input_features={
                "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            },
            output_features={
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            },
            chunk_size=50,
            n_action_steps=10,
            n_decoder_layers=1,
            pretrained_backbone_weights=None,
            device="cpu",
        )
        policy = ACTPolicy(cfg)
        save_path = tmp_dir / "act_b_test"
        policy.save_pretrained(save_path)

        reloaded = ACTPolicy.from_pretrained(save_path)
        assert reloaded.config.chunk_size == 50
        assert reloaded.config.n_decoder_layers == 1
        assert sum(p.numel() for p in reloaded.parameters()) == 51_573_639
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
