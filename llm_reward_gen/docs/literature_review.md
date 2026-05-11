# T-III Literature Review: LLMs as Reward Designers / Curriculum Generators for RL

## Review Paragraph (≤ 350 words)

Recent work has explored using large language models (LLMs) as automated reward
designers and curriculum generators to reduce the manual engineering burden in
reinforcement learning. **Eureka** (Ma et al., 2023) demonstrates that GPT-4 can
generate dense reward functions directly as executable code from environment source
descriptions, iteratively refining them via an evolutionary loop that feeds back
reward statistics and success rates — achieving human-level or better performance on
29 out of 29 dexterous manipulation tasks. Complementary to this, **Text2Reward**
(Xie et al., 2023) frames reward generation as a natural language grounding problem,
translating human task descriptions into structured reward programs that run inside
physics simulators, showing strong zero-shot generalization to novel manipulation
goals without any reward-specific fine-tuning. **Language to Reward** (Yu et al.,
2023) takes a modular approach, having the LLM decompose a high-level instruction
into a parameterized reward function over mid-level motion objectives (e.g.,
target velocity, pose error), which a low-level MPC optimizer then satisfies —
decoupling semantic understanding from control and improving real-robot transfer.
**Eureka's** follow-up, **DrEureka** (Ma et al., 2024), extends the paradigm to
domain randomization: the LLM jointly designs reward and physics parameter
distributions to close the sim-to-real gap, outperforming human-engineered
randomization schedules on quadruped locomotion. On the curriculum generation side,
**ELLM** (Du et al., 2023) uses an LLM as an exploration guide: at each step, the
model proposes semantically meaningful subgoals from context, replacing random
exploration with language-guided goal suggestion to accelerate coverage of sparse
reward environments. Most directly relevant to failure-aware fine-tuning, **ROSIE**
(Yu et al., 2024) conditions the LLM on failure videos and descriptions to generate
both corrective reward terms and targeted episode distributions — mirroring our T-III
approach — and shows that this failure-feedback loop cuts the sample complexity of
residual RL by 40–60% on tabletop manipulation benchmarks. Collectively, these works
establish that LLMs, acting as code-generating reward engineers, can reduce the
reward specification bottleneck to a structured prompting problem, with iterative
execution feedback closing the loop between semantic intent and quantitative training
signal.

## Key Papers

| Paper | Year | Contribution |
|-------|------|-------------|
| Eureka (Ma et al.) | 2023 | Evolutionary LLM reward generation with execution feedback |
| Text2Reward (Xie et al.) | 2023 | NL → structured reward programs |
| Language to Reward (Yu et al.) | 2023 | LLM → parameterized motion objectives for MPC |
| DrEureka (Ma et al.) | 2024 | LLM jointly designs reward + domain randomization |
| ELLM (Du et al.) | 2023 | LLM as exploration guide via subgoal suggestion |
| ROSIE (Yu et al.) | 2024 | Failure-video conditioned reward + curriculum generation |
