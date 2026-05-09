# Diffusion Policy on Push-T

Implementation of **Diffusion Policy** trained on the **Push-T** robotic manipulation task using the **ManiSkill** simulator.

---

## What are we trying to do?

The goal is to teach a robot arm to **push a T-shaped block to a target position** on a table — purely by learning from expert demonstrations, without hand-crafted reward shaping or manual programming of motion.

This is called **Imitation Learning** (or Learning from Demonstrations). The robot watches recorded expert trajectories and learns a policy that can reproduce the same kind of behaviour in new situations.

The specific challenge with robotic manipulation is that the action space is **continuous and multimodal** — at any given moment, there may be several equally valid ways to push the block toward the goal. A policy that just averages over all valid actions will produce a bad action that achieves neither. Diffusion Policy is designed specifically to handle this.

---

## How are we solving it?

### The core idea: treat action generation as denoising

Instead of directly predicting the next action, we frame it as a **denoising problem** borrowed from image generation (DDPM / Stable Diffusion):

1. During **training**, we take a correct action sequence from a demonstration, add random Gaussian noise to it, and ask a neural network to predict what noise was added. This is repeated for many different noise levels.
2. During **inference**, we start from **pure random noise** and run the network repeatedly to gradually remove the noise, step by step, until a clean, executable action sequence emerges.

Because the denoising process can converge to different valid actions from different random starting points, the policy naturally captures **multiple modes** of behaviour — exactly what averaging-based policies cannot do.

### Observation conditioning

The denoiser does not work blindly. At every denoising step it is shown:

- A short **history of camera images** (the last 2 frames) so it knows the current state of the scene.
- The current **diffusion timestep** so it knows how noisy its input is and how aggressively to denoise.

Both pieces of information are injected into the network through **FiLM conditioning** (Feature-wise Linear Modulation), which scales and shifts internal feature maps at every layer based on the conditioning signal.

### Predict a horizon, execute a slice

Rather than predicting one action at a time, the policy predicts a **window of 16 future actions** at once. Of those 16, only the first 8 are actually executed before the policy is called again. This receding-horizon approach gives the policy enough lookahead to plan smooth motions while staying reactive to changes in the scene.

---

## Model Architecture

```
Observation (2 frames of 96×96 RGB)
        │
        ▼
┌─────────────────────┐
│  ObservationEncoder │  ResNet-18 → 256-dim vector
│  (GAP + 2-layer MLP)│
└─────────┬───────────┘
          │ obs_cond (256-d)
          │
          │         Diffusion timestep t
          │                │
          │         ┌──────▼──────────┐
          │         │ SinusoidalPosEmb│  → 256-d time embedding
          │         └──────┬──────────┘
          │                │
          └────────────────┤
                     concat (512-d)
                           │
                           ▼
              ┌────────────────────────┐
              │   ConditionalUNet1D    │  operates on action sequence (T=16)
              │                        │
              │  [Encoder]             │
              │   ResBlock → downsample│  FiLM applied at every block
              │   ResBlock → downsample│  using the 512-d conditioning vector
              │   ResBlock → downsample│
              │                        │
              │  [Bottleneck]          │
              │   ResBlock             │
              │                        │
              │  [Decoder]             │
              │   upsample + skip conn │
              │   ResBlock             │
              │   upsample + skip conn │
              │   ResBlock             │
              └──────────┬─────────────┘
                         │
                         ▼
              Predicted noise  ε̂  (16 × action_dim)
```

**Training loss:** MSE between the true noise `ε` and the predicted noise `ε̂`.

**Inference:** Start from `x_T ~ N(0,I)`, run 20 DDPM denoising steps to recover `x_0` (the clean action sequence), execute the first 8 actions.

### Visual Encoder

Following the original Diffusion Policy paper (Section 4.3 and Appendix C.1), we use a **ResNet-18** (pretrained on ImageNet) as the visual encoder:

- ResNet-18 backbone with **ELU activations** (replacing standard ReLU as per Appendix C.1)
- The final global average pooling layer and FC layer are removed to obtain spatial feature maps
- Global average pooling produces a 512-d feature vector
- A **2-layer MLP projection** (Linear → ELU → Linear) maps this to the 256-d observation conditioning vector

This provides significantly better feature depth and receptive field compared to a simple 4-layer CNN, enabling more accurate tracking of the T-block's orientation and spatial relationships.

### Key numbers (defaults)

| Hyperparameter | Value | What it controls |
|---|---|---|
| `obs_horizon` | 2 | Frames of history used as input |
| `pred_horizon` | 16 | Future action steps predicted per call |
| `action_horizon` | 8 | Steps actually executed before next call |
| `obs_cond_dim` | 256 | Size of the observation embedding |
| `num_diffusion_steps` | 100 | DDPM training noise levels |
| `num_inference_steps` | 20 | Denoising steps at runtime |
| `image_size` | 96 × 96 | Input resolution |

---

## Folder Structure

```
Continual-RL/
│
├── run.ipynb                  ← THE ONLY FILE YOU RUN IN COLAB
│                                 Thin runner cells; no logic lives here.
│
├── setup/
│   └── install.py             ← Vulkan GPU rendering setup + pip installs.
│                                 Packages are cached to Google Drive so
│                                 subsequent restarts take ~10 s, not 3 min.
│
├── envs/
│   ├── make_env.py            ← Factory functions: make_pusht_env(),
│   │                             make_eval_env(), generic make_env().
│   └── wrappers.py            ← DriveRecordEpisode (auto-saves videos to Drive),
│                                 FrameStackWrapper (stacks obs history for policy).
│
├── data/
│   ├── demo_loader.py         ← Download expert demos, convert them to the
│   │                             right observation/action format, replay episodes.
│   └── dataset.py             ← PushTDemoDataset: sliding-window dataset over
│                                 trajectories. Normalizer: fits min-max stats
│                                 once and saves them to Drive.
│
├── policy/
│   ├── networks.py            ← All neural network building blocks:
│   │                             - SinusoidalPosEmb  (timestep → vector)
│   │                             - FiLM              (conditioning injection)
│   │                             - ResBlock1D        (1-D residual block)
│   │                             - ConditionalUNet1D (noise predictor)
│   │                             - ResNet18Encoder   (ResNet-18 visual encoder)
│   │                             - ObservationEncoder (ResNet-18 or MLP)
│   └── diffusion_policy.py    ← DiffusionPolicy: ties encoder + UNet + DDPM
│                                 scheduler together. Exposes compute_loss()
│                                 for training and predict_action() for eval.
│                                 save() / load() write to Drive.
│
├── training/
│   ├── config.py              ← Dataclass configs (EnvConfig, PolicyConfig,
│   │                             TrainConfig) with JSON save/load so every
│   │                             experiment is reproducible from its config file.
│   └── trainer.py             ← Full training loop: builds dataset, policy,
│                                 optimizer; saves checkpoints to Drive every N
│                                 epochs; resumes from any checkpoint; runs eval
│                                 rollouts and writes TensorBoard logs to Drive.
│
└── utils/
    └── visualization.py       ← Helpers for Jupyter/Colab:
                                  show_rgb, show_camera_view, show_pointcloud,
                                  show_eval_grid, display_video, plot_training_curve.
```

---

## Google Colab workflow

The entire project is designed so that **all persistent state lives on Google Drive** and the Colab VM is treated as a stateless compute node.

```
Session restart
      │
      ▼
Cell 0 – Mount Drive + add project to sys.path      (5 s)
Cell 1 – full_setup()  →  Vulkan + pip (cached)     (~10 s after first run)
Cell 2 – Sanity check env                           (optional)
      │
      ▼  (only needed once ever)
Cell 3 – download_demos()
Cell 4 – convert_demos()
      │
      ▼  (every experiment)
Cell 5 – Build Config, edit hyperparameters, save to Drive
Cell 6 – Trainer(cfg).train()   ← checkpoints auto-saved to Drive
      │
      ▼  (after training or after restart)
Cell 8 – Trainer(cfg).evaluate()  ←  loads latest checkpoint from Drive
Cell 9 – display_video(...)
```

If Colab disconnects mid-training, set `resume_from` in Cell 5 to the path of the last saved checkpoint and re-run Cell 6. No progress is lost.

---

## Dependencies

pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


| Package | Purpose |
|---|---|
| `mani_skill` | GPU-accelerated robotics simulator |
| `diffusers` | DDPM scheduler |
| `einops` | Tensor reshaping inside UNet |
| `gym-pusht` | Push-T task registration |
| `zarr` | Efficient trajectory storage |
| `tensorboard` | Training curve logging |
| `torchvision` | ResNet-18 pretrained weights |
