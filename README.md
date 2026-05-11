# Continual RL on ManiSkill Push-T

A four-part assignment that takes a robot manipulation policy from **imitation learning** through **failure analysis**, **LLM-driven reward engineering**, and **residual RL fine-tuning** — all on the ManiSkill Push-T task.

The robot must push a T-shaped block onto a goal T silhouette on a table. Success requires ≥ 90% overlap in both position and orientation.

---

## Repository Architecture

```
Continual-RL/
│
├── run.ipynb                          # Exploratory notebook: env setup, data inspection,
│                                      # policy rollouts, failure visualization
│
├── official_diffusion_policy/         # T-I  ── Imitation Learning baseline
│   ├── train.py                       #   State-based diffusion policy training
│   ├── train_rgbd.py                  #   Visual (RGB-D) diffusion policy training
│   ├── baselines.sh                   #   Tuned training commands for all tasks
│   └── diffusion_policy/
│       ├── conditional_unet1d.py      #   1-D U-Net noise predictor (FiLM conditioned)
│       ├── evaluate.py                #   Evaluation loop (success rate, videos)
│       ├── make_env.py                #   CPU / GPU env factory + wrappers
│       ├── plain_conv.py              #   CNN encoder for RGB-D observations
│       └── utils.py                   #   HDF5 demo loading, dataset class
│
├── llm_reward_gen/                    # T-III ── LLM Reward & Curriculum Generation
│   ├── video_analysis.py              #   Frame extraction from failure mp4s, failure
│   │                                  #   mode heuristics from state trajectories
│   ├── prompts.py                     #   Exact prompt templates (reward + episode config)
│   ├── pipeline.py                    #   Anthropic API client: video→frames→LLM→code
│   │                                  #   with iterative refinement on validation errors
│   ├── custom_envs.py                 #   PushTLLMRewardEnv — ManiSkill v3 subclass;
│   │                                  #   hot-loads generated reward & episode functions;
│   │                                  #   registers as "PushT-LLMReward-v1"
│   ├── validation.py                  #   7-check validator (shape, range, gradient signal,
│   │                                  #   success consistency, correlation, NaN/Inf)
│   │                                  #   Uses MockEnv — no simulator required
│   ├── run_generate.py                #   CLI: video → reward code + episode config
│   ├── run_validate.py                #   CLI: validate generated code files
│   ├── failure_configs/
│   │   ├── rotation_failure.json      #   Pre-filled config for rotation failure mode
│   │   └── overshoot.json             #   Pre-filled config for overshoot failure mode
│   ├── generated/                     #   Auto-populated by run_generate.py
│   │   └── (reward_*.py, episode_config_*.py)
│   └── docs/
│       ├── literature_review.md       #   T-III lit review (Eureka, Text2Reward, …)
│       ├── failure_modes.md           #   Failure mode documentation template
│       └── prompts_log.md             #   Auto-appended log of every LLM call
│
├── ppo/                               # T-IV ── RL Fine-tuning
│   ├── ppo.py                         #   Vanilla PPO on ManiSkill v3 (baseline)
│   ├── ppo_llm_reward.py              #   PPO wired to PushT-LLMReward-v1:
│   │                                  #   failure-biased training + targeted eval +
│   │                                  #   nominal eval (catastrophic forgetting check)
│   ├── ppo_fast.py                    #   Compiled PPO with CUDA graphs (LeanRL)
│   ├── ppo_rgb.py                     #   Visual PPO baseline
│   └── baselines.sh / examples.sh     #   Tuned example commands
│
└── policy_decorator/                  # T-IV (alt) ── Residual Policy (SAC-based)
    ├── offline/                       #   Train base policy (DP or BET) from demos
    ├── online/                        #   Online residual fine-tuning with SAC
    │   ├── pi_dec_diffusion_maniskill2.py   # Diffusion base + SAC residual (state)
    │   ├── pi_dec_diffusion_maniskill2_rgbd.py  # Diffusion base + SAC residual (visual)
    │   └── pi_dec_bet_maniskill2.py         # BET base + SAC residual
    ├── nets/                          #   Network architectures (UNet, BET, CNN)
    ├── envs/maniskill_fixed.py        #   Custom env subclasses (ManiSkill v2 pattern)
    └── utils/                         #   HDF5 loading, samplers, profiling
```

### Data Flow

```
Demonstrations (HDF5)
        │
        ▼
[T-I] official_diffusion_policy/train.py
        │  Diffusion Policy (imitation learning)
        │  Output: checkpoints/best.pt
        │
        ▼
[T-II] Evaluate + record failure videos
        │  Identify ≥2 failure modes with quantitative bounds
        │  Output: eval_videos/*.mp4, failure characterization
        │
        ▼
[T-III] llm_reward_gen/run_generate.py
        │  LLM reads failure video → generates reward fn + episode sampler
        │  Output: generated/reward_*.py, generated/episode_config_*.py
        │
        ▼
[T-IV] ppo/ppo_llm_reward.py   (or policy_decorator/online/)
        │  Residual/fresh policy fine-tuned with LLM reward
        │  Evaluated on: (a) targeted failure episodes, (b) full distribution
        │  Output: runs/<name>/final_ckpt.pt + TensorBoard logs
```

---

## Environment Setup

**Prerequisites:** NVIDIA GPU with CUDA, Linux (ManiSkill GPU sim requires Vulkan).

```bash
# Clone and enter repo
git clone <repo_url> && cd Continual-RL

# Create the Python virtual environment (Python 3.10–3.11 recommended)
python -m venv rl_env
source rl_env/bin/activate

# Install ManiSkill v3 and all dependencies
pip install mani-skill
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers accelerate tyro tensorboard wandb opencv-python h5py
pip install anthropic          # for T-III LLM pipeline

# Install the diffusion policy package (needed for T-I imports)
cd official_diffusion_policy && pip install -e . && cd ..
```

---

## T-I: Train Diffusion Policy

### 1. Download demonstration data

```bash
# Download Push-T demonstrations
python -m mani_skill.utils.download_demo "PushT-v1"
```

### 2. Preprocess (replay) trajectories into the right observation format

```bash
DEMO_PATH=~/.maniskill/demos

# State-based observations (fastest, used for most experiments)
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ${DEMO_PATH}/PushT-v1/rl/trajectory.h5 \
  --use-first-env-state \
  -c pd_joint_delta_pos \
  -o state \
  --save-traj \
  --num-envs 10 \
  -b physx_cuda

# RGB-D observations (for visual policy)
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ${DEMO_PATH}/PushT-v1/rl/trajectory.h5 \
  --use-first-env-state \
  -c pd_joint_delta_pos \
  -o rgb \
  --save-traj \
  --num-envs 10 \
  -b physx_cuda
```

### 3. Train (state-based)

```bash
cd official_diffusion_policy

# Single seed
python train.py \
  --env-id PushT-v1 \
  --demo-path ~/.maniskill/demos/PushT-v1/rl/trajectory.state.pd_joint_delta_pos.physx_cuda.h5 \
  --control-mode "pd_joint_delta_pos" \
  --sim-backend "physx_cuda" \
  --num-demos 100 \
  --max_episode_steps 150 \
  --num_eval_envs 100 \
  --total_iters 50000 \
  --act_horizon 1 \
  --exp-name diffusion_policy-PushT-v1-state-seed1

# Run across 5 seeds to get baseline success rate (250 rollouts each)
for seed in 1 2 3 4 5; do
  python train.py \
    --env-id PushT-v1 \
    --demo-path ~/.maniskill/demos/PushT-v1/rl/trajectory.state.pd_joint_delta_pos.physx_cuda.h5 \
    --control-mode "pd_joint_delta_pos" \
    --sim-backend "physx_cuda" \
    --num-demos 100 \
    --max_episode_steps 150 \
    --num_eval_envs 100 \
    --num_eval_episodes 250 \
    --total_iters 50000 \
    --act_horizon 1 \
    --seed ${seed} \
    --exp-name diffusion_policy-PushT-v1-state-seed${seed}
done
```

### 4. Train (visual RGB-D)

```bash
python train_rgbd.py \
  --env-id PushT-v1 \
  --demo-path ~/.maniskill/demos/PushT-v1/rl/trajectory.rgb.pd_joint_delta_pos.physx_cuda.h5 \
  --control-mode "pd_joint_delta_pos" \
  --sim-backend "physx_cuda" \
  --num-demos 100 \
  --obs-mode "rgb" \
  --max_episode_steps 150 \
  --total_iters 100000 \
  --exp-name diffusion_policy-PushT-v1-rgb-seed1

cd ..
```

**Key hyperparameters:**

| Parameter | Default | Notes |
|-----------|---------|-------|
| `obs_horizon` | 2 | History of observations fed to policy |
| `act_horizon` | 1 | Actions executed before re-planning |
| `pred_horizon` | 16 | Total denoised action window length |
| `unet_dims` | [64, 128, 256] | ~4.5 M parameters |
| `lr` | 1e-4 | AdamW learning rate |
| `batch_size` | 1024 | Samples per gradient step |
| `total_iters` | 50 000 | Gradient steps |

**Output:** `runs/<exp_name>/best.pt` — checkpoint with best eval success rate.

---

## T-II: Failure Mode Identification

### 1. Run evaluation and record failure videos

```bash
cd official_diffusion_policy

python -c "
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
import torch, gymnasium as gym

# Load your trained checkpoint
# Then run 250 episodes per seed, recording videos
# Filter episodes where success=False
"

# Or use the built-in eval in train.py with --capture-video
# Videos are written to runs/<exp_name>/eval_videos/
```

### 2. Analyse failure trajectories

Open `run.ipynb` (Section: *Failure Analysis*) to:
- Plot T-block trajectories over 250 rollouts
- Compute per-rollout rotation error, final XY distance to goal
- Cluster episodes by outcome (rotation failure / overshoot / stuck / partial)
- Identify initial-state ranges that predict each failure

### 3. Record per-failure-mode success rates

For each identified failure mode, configure initial-state bounds in `run.ipynb` and run 250 evaluation rollouts under that restricted initialization — this is your **targeted-evaluation baseline**.

```bash
cd ..
```

---

## T-III: LLM Reward and Episode Generation

### Prerequisites

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # Set your API key
```

### 1. Generate reward function and episode sampler from a failure video

```bash
# Using a pre-filled failure config (edit quantitative_chars and video path first):
python llm_reward_gen/run_generate.py \
  --config llm_reward_gen/failure_configs/rotation_failure.json \
  --video-path runs/<exp_name>/eval_videos/failure_001.mp4

# Or specify everything inline:
python llm_reward_gen/run_generate.py \
  --video-path runs/<exp_name>/eval_videos/failure_001.mp4 \
  --failure-mode rotation_failure \
  --failure-description "T-block reaches goal XY position but is 60-120 degrees off target rotation. Policy oscillates end-effector without correcting orientation." \
  --quantitative-chars \
      "T-block final XY dist < 0.04m but intersection < 90%" \
      "Rotation error > 45 deg at episode end" \
      "Occurs when initial Z-rotation in [1.5, 3.5] rad" \
  --baseline-success-rate 62 \
  --model claude-sonnet-4-20250514

# Repeat for the second failure mode:
python llm_reward_gen/run_generate.py \
  --config llm_reward_gen/failure_configs/overshoot.json \
  --video-path runs/<exp_name>/eval_videos/overshoot_001.mp4
```

Generated files appear in `llm_reward_gen/generated/`:
- `reward_rotation_failure_v001.py` — dense reward function
- `episode_config_rotation_failure_v001.py` — failure-biased episode sampler

### 2. Validate the generated code

```bash
# Validation without simulator (fast, uses MockEnv)
python llm_reward_gen/run_validate.py \
  --reward-code llm_reward_gen/generated/reward_rotation_failure_v001.py \
  --episode-config llm_reward_gen/generated/episode_config_rotation_failure_v001.py

# Validation with full simulator (slow, most accurate — run on GPU machine)
python llm_reward_gen/run_validate.py \
  --reward-code llm_reward_gen/generated/reward_rotation_failure_v001.py \
  --episode-config llm_reward_gen/generated/episode_config_rotation_failure_v001.py \
  --use-sim
```

Checks performed:
1. Syntax / import — code runs without errors
2. Output shape — returns tensor `(B,)`
3. Range — values in `[-1, 10]`
4. Success consistency — reward = 3.0 at success states
5. Gradient signal — reward increases as T-block approaches goal
6. Correlation — Pearson r > 0.3 with proximity-to-goal over 500 random states
7. No reward hacking — no random state achieves reward > reward-at-goal

### 3. Inspect and document

- View the generated code in `llm_reward_gen/generated/`
- Check `llm_reward_gen/docs/prompts_log.md` for every LLM call, retry, and validation result
- Fill in `llm_reward_gen/docs/failure_modes.md` with your T-II characterization data

**How reward integration works:**

```python
# custom_envs.py registers "PushT-LLMReward-v1" as a ManiSkill v3 environment
# that subclasses PushTEnv and overrides:
#   compute_dense_reward()  → calls generated reward_*.py
#   _initialize_episode()   → 70% failure-biased + 30% uniform episodes

import gymnasium as gym
import llm_reward_gen.custom_envs  # triggers @register_env

env = gym.make(
    "PushT-LLMReward-v1",
    num_envs=512,
    obs_mode="state",
    reward_mode="dense",          # ← activates LLM reward
    control_mode="pd_joint_delta_pos",
    sim_backend="physx_cuda",
    reward_code_path="llm_reward_gen/generated/reward_rotation_failure_v001.py",
    episode_config_code_path="llm_reward_gen/generated/episode_config_rotation_failure_v001.py",
    failure_bias_ratio=0.7,       # 70% failure-biased episodes
)
```

---

## T-IV: Residual Policy Fine-tuning

### Option A — PPO with LLM reward (recommended for Push-T)

Fine-tune a **fresh PPO policy** on `PushT-LLMReward-v1` using the dense LLM reward and failure-biased episode distribution.

```bash
# Fine-tune on rotation failure mode
python ppo/ppo_llm_reward.py \
  --reward-code-path llm_reward_gen/generated/reward_rotation_failure_v001.py \
  --episode-config-path llm_reward_gen/generated/episode_config_rotation_failure_v001.py \
  --failure-mode rotation_failure \
  --num-envs 512 \
  --total-timesteps 5000000 \
  --gamma 0.8 \
  --gae-lambda 0.9 \
  --num-steps 50 \
  --update-epochs 4 \
  --eval-nominal \              # also eval on nominal distribution (forgetting check)
  --capture-video \
  --seed 1

# Fine-tune on overshoot failure mode
python ppo/ppo_llm_reward.py \
  --reward-code-path llm_reward_gen/generated/reward_overshoot_v001.py \
  --episode-config-path llm_reward_gen/generated/episode_config_overshoot_v001.py \
  --failure-mode overshoot \
  --num-envs 512 \
  --total-timesteps 5000000 \
  --eval-nominal \
  --seed 1
```

**Outputs per run:**
- `runs/<name>/ckpt_*.pt` — periodic checkpoints
- `runs/<name>/final_ckpt.pt` — final model
- `runs/<name>/eval_videos/` — evaluation rollout videos
- TensorBoard: `tensorboard --logdir runs/`

### Option B — Policy Decorator (SAC residual on ManiSkill v2 tasks)

> **Note:** `policy_decorator/` targets ManiSkill v2 (`mani_skill2`). PushT-v1 is a ManiSkill v3 environment. Use Option A for Push-T. Option B works as-is for the ManiSkill v2 tasks (PegInsertionSide, TurnFaucet, PushChair).

```bash
# Step 1: Train base policy offline
cd policy_decorator
python offline/diffusion_policy_unet_maniskill2.py \
  --env-id PegInsertionSide-v2 \
  --demo-path data/PegInsertionSide/trajectory.h5 \
  --total-iters 1000000

# Step 2: Online residual fine-tuning with SAC
python online/pi_dec_diffusion_maniskill2.py \
  --env-id PegInsertionSide-v2 \
  --base-policy-ckpt checkpoints/diffusion_PegInsertionSide/best.pt \
  --res-scale 0.1 \
  --prog-explore 30000 \
  --total-timesteps 2000000
cd ..
```

### Evaluation

```bash
# Evaluate on targeted failure episodes (100% failure-biased init)
python ppo/ppo_llm_reward.py \
  --evaluate \
  --checkpoint runs/<run_name>/final_ckpt.pt \
  --reward-code-path llm_reward_gen/generated/reward_rotation_failure_v001.py \
  --episode-config-path llm_reward_gen/generated/episode_config_rotation_failure_v001.py \
  --failure-mode rotation_failure \
  --eval-failure-bias-ratio 1.0 \
  --num-eval-envs 50 \
  --num-eval-steps 250 \
  --capture-video

# Evaluate on nominal full distribution (check catastrophic forgetting)
python ppo/ppo.py \
  --env-id PushT-v1 \
  --evaluate \
  --checkpoint runs/<run_name>/final_ckpt.pt \
  --num-eval-envs 50 \
  --num-eval-steps 250 \
  --capture-video
```

**Expected results:**
- **Targeted failure episodes:** improvement over base diffusion policy success rate
- **Nominal distribution:** ≤ 5% degradation vs. T-I baseline (near-zero forgetting)

### Run all 5 seeds for final reporting

```bash
for seed in 1 2 3 4 5; do
  python ppo/ppo_llm_reward.py \
    --reward-code-path llm_reward_gen/generated/reward_rotation_failure_v001.py \
    --episode-config-path llm_reward_gen/generated/episode_config_rotation_failure_v001.py \
    --failure-mode rotation_failure \
    --num-envs 512 \
    --total-timesteps 5000000 \
    --eval-nominal \
    --seed ${seed} \
    --exp-name ppo_llm_rotation_seed${seed}
done
```

---

## Key State Variables in PushT-v1

The ManiSkill Push-T environment (`PushT-v1`) exposes these state variables, which the LLM reward functions use directly:

| Variable | Shape | Description |
|----------|-------|-------------|
| `self.tee.pose.p` | `(B, 3)` | T-block XYZ position |
| `self.tee.pose.q` | `(B, 4)` | T-block quaternion `[w, x, y, z]` |
| `self.goal_tee.pose.p` | `(B, 3)` | Goal T position — fixed at `[-0.156, -0.1, 0.001]` |
| `self.goal_z_rot` | scalar | Goal Z-rotation = `(5/3)π ≈ 5.236` rad |
| `self.agent.tcp.pose.p` | `(B, 3)` | End-effector (stick tip) position |
| `self.quat_to_z_euler(q)` | `(B,)` | Z euler angle from quaternion batch |
| `info["success"]` | `(B,)` bool | True when intersection ≥ 90% |

Episode spawn box (relative to goal): x ∈ `[-0.1, +0.1]`, y ∈ `[-0.1, +0.2]`, z = `0.021`, rotation uniform `[0, 2π]`.

---

## ManiSkill v2 vs v3 Note

| Component | ManiSkill Version | Notes |
|-----------|------------------|-------|
| `official_diffusion_policy/` | v3 (`mani_skill`) | Push-T is v3-only |
| `ppo/ppo.py`, `ppo_llm_reward.py` | v3 (`mani_skill`) | GPU-parallel with `ManiSkillVectorEnv` |
| `llm_reward_gen/custom_envs.py` | v3 (`mani_skill`) | Subclasses `PushTEnv` from v3 |
| `policy_decorator/` | v2 (`mani_skill2`) | Works for PegInsertion, TurnFaucet, PushChair |

To use Policy Decorator residual training with Push-T, adapt `policy_decorator/online/pi_dec_diffusion_maniskill2.py` to use v3 imports (env creation pattern from `ppo/ppo.py`).

---

## Repository-Wide Quick Reference

```bash
# Setup
source rl_env/bin/activate
export ANTHROPIC_API_KEY="sk-ant-..."

# T-I: Train
cd official_diffusion_policy
python train.py --env-id PushT-v1 --demo-path <path> --total_iters 50000
cd ..

# T-III: Generate LLM reward
python llm_reward_gen/run_generate.py --config llm_reward_gen/failure_configs/rotation_failure.json --video-path <failure_video.mp4>

# T-III: Validate
python llm_reward_gen/run_validate.py --reward-code llm_reward_gen/generated/reward_rotation_failure_v001.py --episode-config llm_reward_gen/generated/episode_config_rotation_failure_v001.py

# T-IV: Fine-tune
python ppo/ppo_llm_reward.py --reward-code-path llm_reward_gen/generated/reward_rotation_failure_v001.py --episode-config-path llm_reward_gen/generated/episode_config_rotation_failure_v001.py --failure-mode rotation_failure --eval-nominal

# Monitor training
tensorboard --logdir runs/

# Replay a trajectory with ray-tracing for nice videos
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path runs/<name>/eval_videos/trajectory.h5 \
  --use-env-states --shader="rt-fast" --save-video --allow-failure -o none
```

---

## Citations

**Diffusion Policy**
```bibtex
@inproceedings{Chi2023DiffusionPolicy,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Chi, Cheng and Feng, Siyuan and Du, Yilun and Xu, Zhenjia and Cousineau, Eric and Burchfiel, Benjamin and Song, Shuran},
  booktitle={RSS},
  year={2023}
}
```

**Policy Decorator**
```bibtex
@article{PolicyDecorator2024,
  title={Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model},
  author={...},
  year={2024}
}
```

**Eureka (LLM Reward Design)**
```bibtex
@article{Ma2023Eureka,
  title={Eureka: Human-Level Reward Design via Coding Large Language Models},
  author={Ma, Yecheng Jason and Liang, William and Wang, Guanzhi and Huang, De-An and Bastani, Osbert and Jayaraman, Dinesh and Zhu, Yuke and Fan, Linxi and Anandkumar, Anima},
  journal={arXiv:2310.12931},
  year={2023}
}
```

**ManiSkill**
```bibtex
@article{ManiSkill3,
  title={ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI},
  author={Stone Tao and Fanbo Xiang and Arth Shukla and Yuzhe Qin and Xander Hinrichsen and Xiaodi Yuan and Chen Bao and Xinsong Lin and Yulin Liu and Tse-kai Chan and Yuan Gao and Xuanlin Li and Tongzhou Mu and Nan Jiang and Tonghe Fang and Derek Lim and Rui Chen and Hao Su},
  journal={arXiv preprint arXiv:2410.00425},
  year={2024}
}
```
