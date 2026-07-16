---
title: "Recursive Harness Self-Improvement: Reorganizing Multi-Agent Systems"
mathjax: true
layout: post
categories: media
comments: true
header-includes:
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{amsthm}
---

<div align="center">
  <video
    src="../assets/RHI_combined_full.mp4"
    poster="../assets/RHI_combined_full_poster.png"
    autoplay loop muted playsinline controls
    width="860"
    style="max-width: 100%; border-radius: 14px; box-shadow: 0 6px 24px rgba(0,0,0,0.12);">
    Your browser does not support embedded video.
    <a href="../assets/RHI_combined_full.mp4">Open the video directly.</a>
  </video>
  <br>
  <em>Recursive Harness Self-Improvement (RHI) reorganizes multi-agent system</em>
</div>

## Will models eat the harness?

A growing view in AI is that harnesses are temporary. As models become more capable, the argument goes, they will internalize the reasoning patterns, tool-use strategies, and coordination rules that we currently build around them. [Noam Brown described the ideal harness as "no harness"](https://www.latent.space/p/noam-brown): external scaffolding is a crutch that increasingly general models should eventually move beyond. [Logan Kilpatrick made a closely related prediction](https://sequoiacap.com/podcast/google-deepminds-logan-kilpatrick-why-the-model-eats-the-harness/): the model "eats" the scaffolding, turning today's external orchestration into part of tomorrow's native model system.

We agree that models will absorb many capabilities now implemented in harnesses. But this does not make the harness irrelevant. It changes its role.

Before a capability can be internalized, a harness is often the system that elicits it, tests it, and records it. Harnesses organize agent workflows that produce solution attempts, critiques, tool calls, intermediate failures, and successful trajectories. These execution traces can then become post-training data for future foundation models.

This creates a recursive model–harness co-evolution loop:

> **Better model → stronger harness → better execution traces → post-training → better model**

A stronger model enables more capable harnesses. Better harnesses, in turn, can produce more useful traces for training the next model. We call this **harness-in-the-loop learning**: optimizing a harness not only for immediate agent performance, but also for the quality of the traces available for future training. The specific scaffolding may move into model weights or provider systems, but harness design remains an active part of the data flywheel: it determines what behavior is elicited and what data is collected for the next round of learning.

The key bottleneck is therefore not only the number of traces, but their quality. A weak harness can generate duplicated work, irrelevant context, missing evidence, or poorly coordinated attempts. A strong harness can produce traces with clearer task signal, better division of labor, and more useful feedback.

Our paper studies the **first half of this loop**. We keep the foundation model fixed and ask how to improve a user-constructed, task-specific harness so that it produces better executions. We do not yet train a future model on those traces; closing that second half of the loop is future work. User-constructed harnesses are a practical optimization surface because they can be specialized without retraining the model or continually redesigning a provider-built system for every task.

To address the first half, we introduce **Recursive Harness Self-Improvement (RHI)**. RHI represents the harness as editable text and adapts it through pairwise feedback from its own previous attempts. The method is designed to be lightweight enough for users to apply to individual tasks and effective within only a few update iterations.

Across 30 synthetic machine-learning research tasks, one or two RHI updates allowed `high`-reasoning agents to outperform the stronger same-family test-time-scaling settings we evaluated. With Claude Opus 4.8, the resulting agent execution also cost up to **60% less** than the `ultracode` baseline.

<!-- The central lesson is simple:

> **Harnesses are not only inference-time scaffolds. They are data-generating systems that shape both current agent performance and the training signal available to future models.** -->

## What exactly is a harness?

Much of the recent work on automatic harness optimization searches in **code space**. [Meta-Harness](https://arxiv.org/abs/2603.28052) evolves executable harness code from earlier candidates and their execution traces. [AutoHarness](https://arxiv.org/abs/2603.03329) synthesizes and refines code harnesses from environment feedback. [Self-Harness](https://arxiv.org/abs/2606.09498) proposes harness edits and accepts them through regression testing. Related workflow-search methods likewise represent agents as programs, graphs, or executable pipelines.

RHI changes the optimization variable. Instead of treating the harness as an executable program, we treat it as a **prompt-level object**: a structured piece of text appended to the task prompt. The optimizer does not rewrite provider code or search over executable workflows. It rewrites the textual specification that tells the coding agent how to organize its work.

> **In RHI, harness optimization is text optimization.**

This choice matters because users often cannot access the implementation of a provider-built agent. They can, however, control the prompt. Representing the harness as text therefore makes task-specific optimization possible even when the model and its underlying agent system are black boxes. It also gives the LLM optimizer a natural interface: read the current harness, interpret the comparison history, and propose the next textual revision.

Under this prompt-level formulation, we define the harness as the agent loop and decompose it into four components:

- **Roles** define what each agent is responsible for.
- **Instructions** define how each agent should work.
- **Contracts** define what information an agent must return.
- **Hops** define when agents run and how control moves through the workflow.

Roles and instructions specify the agent design. Contracts and hops specify the workflow. Together, they turn multi-agent coordination into textual information that can be inspected, compared, and improved.

<div align="center" style="margin: 28px 0;">
  <img src="../assets/rhi-harness-components.png" alt="The RHI harness consists of roles, instructions, contracts, and workflow hops" width="900" style="max-width: 100%;">
  <br>
  <em>RHI represents agent design and workflow as an editable textual harness.</em>
</div>

This decomposition makes harness optimization an information-routing problem. For example, a generic contract may ask an experimental agent to "return its findings." A task-specific contract can instead require the exact metrics, assumptions, failure cases, and artifact paths needed by the final report. The latter sends less irrelevant context and more decision-relevant evidence.

Hops provide the complementary control structure. They can require evaluation before reporting, trigger a second experiment when an acceptance criterion fails, or call a reviewer only after implementation is complete. Together, contracts and hops determine what information moves through the system and when.

For concrete examples of how RHI updates a harness, see [Appendix B: Examples of RHI Harness]().

## Recursive Harness Self-Improvement

Searching over a large population of harnesses is expensive: every candidate requires a complete agent execution, and comparing all candidate pairs grows quadratically with the population size.

This cost is not merely theoretical. [*Rethinking the Evaluation of Harness Evolution for Agents*](https://arxiv.org/pdf/2607.12227) compares automatic harness evolution with simpler test-time-scaling methods under matched feedback and inference budgets. On Terminal-Bench 2.1, harness evolution did not consistently outperform parallel sampling or sequential refinement, and its improvements transferred poorly to held-out tasks.

The practical lesson is not that harness search has no value. It is that search must earn its cost. For a fixed budget, every token spent discovering a harness is a token that could have been spent directly improving the task output. **A practical harness optimizer must therefore be lightweight and produce useful updates within only a few iterations.**

RHI is designed around this constraint. It replaces broad population search with a local question:

> Did the current harness produce a better result than the previous harness?

For a task $x$, let $H^{(i)}$ be the harness at iteration $i$ and $y^{(i)}$ the repository produced with that harness. Each RHI iteration performs four steps:

1. Run the fixed coding agent with $H^{(i)}$ to produce $y^{(i)}$.
2. Ask an LLM evaluator to compare $y^{(i)}$ with $y^{(i-1)}$.
3. Add the preference feedback to a self-comparison history.
4. Ask a separate LLM optimizer to revise the harness using its current version and the accumulated history.

<div align="center" style="margin: 28px 0;">
  <img src="../assets/rhi-algorithm.png" alt="The Recursive Harness Self-Improvement loop" width="950" style="max-width: 100%;">
  <br>
  <em>RHI compares consecutive outputs, saves the feedback, and uses the accumulated history to revise the harness.</em>
</div>

Because the previous output is cached, each update requires only one new agent execution and one new comparison per task, plus the harness-optimizer call. The accumulated history acts as a semantic form of momentum: later updates can preserve revisions associated with wins and reconsider those associated with losses.

The evaluator uses the evaluation rubric to produce feedback, but the harness optimizer does not see that rubric directly. It must infer useful revisions from the comparison history.

This is noisy local ascent, not gradient descent. A comparison with one predecessor is not a global estimate of harness quality, and improvement is not guaranteed to be monotonic. But under a standard latent-utility model of pairwise preference, beating the previous harness is aligned with moving upward in the same quality ordering.

The word *recursive* is intentionally bounded. RHI updates a persistent system component using evidence generated by its earlier versions. It does **not** change model weights, rewrite the evaluator, or improve the harness optimizer itself.

## Experimental setup

We constructed 30 synthetic, open-ended ML research tasks: ten each in quantitative finance, robotics, and pharmaceutical machine learning. Each task asks the agent to build a complete research repository, including a report, reproducible code, metrics, visualizations, and an artifact index.

These tasks have no single correct scalar answer. We therefore compare repositories with LLM judges using six criteria: deliverable coverage, empirical rigor, reproducibility, presentation, engineering quality, and task alignment. Results are averaged across multiple evaluator configurations and three random seeds.

We test three progressively stronger base models—Claude Sonnet 4.6, Opus 4.7, and Opus 4.8. In each case, RHI adapts a `high`-reasoning agent and compares it with stronger same-family settings such as `xhigh`, `max`, and `ultracode`. This isolates the question we care about: **can a better harness outperform additional reasoning inside the default harness?**

## Results: a few harness updates beat the evaluated scaling baselines

<div align="center" style="margin: 28px 0;">
  <img src="../assets/rhi-results.png" alt="RHI performance and normalized execution cost across Sonnet 4.6, Opus 4.7, and Opus 4.8" width="1000" style="max-width: 100%;">
  <br>
  <em>Few RHI iterations improve pairwise wins beyond the evaluated test-time-scaling baselines. The overlaid line shows normalized execution cost.</em>
</div>

| Base model | RHI result | Execution-cost comparison |
|:--|:--|:--|
| Sonnet 4.6 | After two updates, `high` + RHI beat `max`, winning 20 of 30 comparisons. | 7% lower than `max` |
| Opus 4.7 | After one update, `high` + RHI surpassed both `xhigh` and `max`. | 18% lower than `max` |
| Opus 4.8 | After two updates, `high` + RHI surpassed `xhigh`, `max`, and `ultracode`. | 23% lower than `max`; 60% lower than `ultracode` |

The Opus 4.8 result is especially notable because `ultracode` already uses a provider-built dynamic multi-agent workflow. On this benchmark, a task-specific user-side harness outperformed that general-purpose system harness.

These cost numbers describe the **final execution with the evolved harness**, not the total cost of discovering it. RHI still pays for each intermediate agent run, evaluation, and optimizer call. Adaptation is therefore most attractive when the harness will be reused, when a few iterations fit within the task budget, or when the intermediate executions are themselves useful.

The gains also do not appear to come primarily from longer generations. For Sonnet 4.6 and Opus 4.8, output-token use stayed nearly flat across RHI iterations while performance improved. For Opus 4.7, tokens and performance rose together, so that experiment is inconclusive. Across all three models, however, the evolved harnesses used fewer cache reads and writes than the strongest comparison settings, consistent with more efficient context management.

RHI complements stronger models rather than replacing them. Improving the Sonnet 4.6 harness did not consistently close the gap to the stronger Opus 4.7 agents. Model scaling and harness optimization remain separate axes of progress.

## What changes inside an improved harness?

Our analysis points to the interfaces between agents.

The first RHI update produced the largest semantic change to the full harness: consecutive cosine similarity was 0.82 for the first transition, then 0.97, 0.98, and 0.99. Later updates behaved more like refinement than redesign.

At the component level, **contracts showed the clearest task-dependent clustering** across embedding models and projection methods. They also stabilized earlier than roles, instructions, and hops. This suggests that RHI first learns what information should cross the interface between agents.

We then asked whether the update trajectory was consistent with a more general objective. Let $X$ be the task and $Z_c$ the representation of harness component $c$. The following information-theoretic objective summarizes the observed pattern:

$$
J
=
\underbrace{\sum_{c\in\mathcal{C}_{\mathrm{target}}} I(Z_c;X)}_{\text{task information in targeted components}}
-
\beta\,
\underbrace{\mathrm{TC}(\{Z_c\}_{c\in\mathcal{C}}\mid X)}_{\text{redundancy after conditioning on the task}},
\qquad \beta>0.
$$

The first term favors task information in the components targeted by the optimizer prompt—contracts and hops in our implementation. The second discourages different harness components from repeating the same information once their shared task is taken into account.

Across two embedding models and both raw and permutation-debiased estimates, task information increased monotonically in contracts and hops from iterations 1 to 4. It decreased in roles and stayed roughly flat or declined in instructions. Over the same iterations, task-conditional redundancy decreased monotonically in every configuration.

The interpretation is not that every part of a good harness should become more task-specific. It is more selective:

> **Put task information in the components that route computation, and give each component a distinct function.**

This objective is a hypothesis, not a loss that RHI explicitly optimizes. The measurements use text embeddings as external proxies and are correlational; they do not reveal the optimizer model's internal mechanism.

## Scope and outlook

The results are promising, but their scope matters. The benchmark contains 30 synthetic ML research tasks, and quality is measured through pairwise LLM judgment. Broader conclusions require real user workloads, other agent environments, and additional model providers. The efficiency analysis also separates the cost of an evolved execution from the total adaptation cost.

RHI provides one concrete step toward the larger model–harness co-evolution loop. The next step is to test whether the resulting execution traces improve future models through post-training—and whether those models, in turn, enable the next generation of harnesses.

The usual scaling question is: *How much more reasoning should the model do?* RHI asks a prior question: *What structure should organize that reasoning?* Our results suggest that, for agent systems, improving that structure can be a powerful complement to scaling.

---

<div align="center" style="margin: 28px 0;">
  <strong>Recursive Harness Self-Improvement</strong><br>
  Hyunin Lee<sup>1,2</sup>, Jinglue Xu<sup>1</sup>, Jeffrey Seely<sup>1</sup>, Donghyun Lee<sup>2</sup>, Matei Zaharia<sup>2</sup>, and Yujin Tang<sup>1</sup><br>
  <em><sup>1</sup>Sakana AI &nbsp;·&nbsp; <sup>2</sup>UC Berkeley</em>
</div>

**Paper and code:** links will be added when the public release is available.

For questions or potential collaborations, please [reach out](mailto:hyunin@berkeley.edu).
