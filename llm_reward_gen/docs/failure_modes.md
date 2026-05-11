# Failure Modes Documentation

## Overview

This document records the failure modes identified in Part 2 (T-II), the LLM-generated
rewards and episode configs designed to address them, and the LLM failure modes
encountered during generation.

---

## Failure Mode 1: [TO BE FILLED FROM T-II ANALYSIS]

### Qualitative Description
<!-- Example: "Robot pushes T-block near goal position but fails to achieve correct
orientation; block oscillates between 60° and 120° off target rotation" -->

### Quantitative Characterization
<!-- Example:
- T-block reaches within 0.03m of goal XY position
- Final rotation error (|cos(θ_block - θ_goal) - 1|) > 0.3
- Occurs with high probability when initial T-block Z-rotation ∈ [1.5, 3.5] rad
- Success rate in this configuration: __% (over 250 rollouts, 5 seeds)
-->

### Video Files
<!-- List mp4 files illustrating this failure mode -->

### Generated Reward Function
- **File**: `generated/reward_[failure_mode]_v001.py`
- **Why it helps**: [Explanation of what reward terms were added]
- **Key reward terms**:
  - [Term 1]: [Description]
  - [Term 2]: [Description]

### Generated Episode Config
- **File**: `generated/episode_config_[failure_mode]_v001.py`
- **Bias**: [Description of how initial conditions are biased]

---

## Failure Mode 2: [TO BE FILLED FROM T-II ANALYSIS]

### Qualitative Description
<!-- Example: "Robot pushes T-block past the goal and oscillates" -->

### Quantitative Characterization
<!-- Example:
- T-block crosses goal region (dist < 0.03m) then overshoots (dist > 0.05m)
- Occurs when initial T-block position is in range x ∈ [-0.25, -0.1], y ∈ [0.0, 0.15]
- Success rate in this configuration: __% (over 250 rollouts, 5 seeds)
-->

### Video Files
<!-- List mp4 files illustrating this failure mode -->

### Generated Reward Function
- **File**: `generated/reward_[failure_mode]_v001.py`
- **Why it helps**: [Explanation]

### Generated Episode Config
- **File**: `generated/episode_config_[failure_mode]_v001.py`

---

## LLM Failure Modes Encountered

### 1. Reward Hacking
**Description**: The LLM generated a reward term that could be maximized without
actually solving the task (e.g., reward maximized by keeping TCP at origin).
**Detection**: Low correlation between generated reward and ground-truth intersection
metric; reward validation check failed.
**Fix**: Re-prompted with explicit anti-hacking instructions; added requirement that
reward should not be maximizable without T-block moving toward goal.

### 2. API Hallucination
**Description**: LLM called methods that don't exist on PushTEnv (e.g., 
`self.tee.get_velocity()`, `self.compute_2d_projection()`).
**Detection**: Runtime ImportError / AttributeError on first validation.
**Fix**: Refined prompt to include explicit list of ALL available API methods.

### 3. Non-smooth Reward
**Description**: Generated reward used hard `if/else` on continuous state variables,
creating zero-gradient regions that stalled RL training.
**Detection**: Static code analysis flagged conditional branches on floats.
**Fix**: Re-prompted with explicit requirement: "no hard if/else on continuous
variables; use smooth approximations like torch.tanh or torch.sigmoid."

### 4. Scale Mismatch
**Description**: Reward components were orders of magnitude apart (e.g., one term
produced values in [0, 0.001] while another in [0, 100]).
**Detection**: Range check in validation; reward max >> 10.
**Fix**: Required each component to be normalized to [0, 1] range explicitly.

### 5. Wrong Quaternion Convention
**Description**: Episode config sampler used (x, y, z, w) quaternion ordering instead
of ManiSkill's (w, x, y, z) convention.
**Detection**: Quaternion norm check passed but orientations were clearly wrong in
sim visualization.
**Fix**: Explicitly stated in prompt: "quaternion format is [w, x, y, z]".

---

## Manual Effort Summary

| Issue | Attempts to Fix | Resolution |
|-------|----------------|------------|
| [Issue 1] | [N] | [How resolved] |
| [Issue 2] | [N] | [How resolved] |

---

## Notes for Submission

Fill in the actual failure mode data from T-II evaluation runs before submission.
The LLM failure modes section should be updated with real examples from the
generation runs, referencing specific entries in `prompts_log.md`.
