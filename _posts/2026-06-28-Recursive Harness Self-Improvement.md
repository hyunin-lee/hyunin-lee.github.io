---
title: "Recursive Harness Self-Improvement: Teaching Agents to Improve Their Own Coordination"
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
  <em>Recursive Harness Self-Improvement (RHI) adapts the coordination layer around a fixed model.</em>
</div>

Most conversations about making AI agents better begin with two levers: use a stronger model, or spend more compute at inference time. There is a third lever that is easier to overlook: improve the system that organizes the model's work.

That system is the **harness**. It determines which agents are available, what each one is asked to do, what information they exchange, and when control passes between them. More reasoning inside a poorly organized workflow can simply produce a more expensive version of the same mistake.

Our paper introduces **Recursive Harness Self-Improvement (RHI)**, a lightweight method for adapting this coordination layer to a task. RHI keeps the foundation model fixed, represents the harness as editable text, and revises it using pairwise feedback from its own previous attempts.

Across 30 synthetic machine-learning research tasks, one or two RHI updates were enough for `high`-reasoning agents to outperform the stronger same-family test-time-scaling baselines we evaluated. On Claude Opus 4.8, the evolved harness also reduced the cost of the evaluated run by as much as **60%** relative to the `ultracode` baseline.

The more interesting result, however, is not merely that the harness improved. It is *how* it improved: the largest changes appeared in the interfaces that route information between agents.

## The harness is part of the intelligence of an agent system

A foundation model does not become a useful coding or research agent on its own. Around it sits a system of prompts, tools, memory, routing, verification, and control flow. Providers supply part of this system; users add another layer through project instructions, skills, subagents, and task-specific workflows.

In this paper, we use a precise, deliberately narrow definition: the harness is the **agent loop**, represented by four kinds of text.

- **Roles** define each agent's responsibility.
- **Instructions** specify how an agent should perform that responsibility.
- **Contracts** define what information an agent must return to the orchestrator.
- **Hops** define the interaction sequence: which agent runs, when it runs, and what happens next.

Roles and instructions describe the agents. Contracts and hops describe the workflow connecting them.

<div align="center" style="margin: 28px 0;">
  <img src="../assets/rhi-harness-components.png" alt="The RHI harness is decomposed into roles, instructions, contracts, and workflow hops" width="900" style="max-width: 100%;">
  <br>
  <em>RHI represents agent design and workflow as a textual, editable harness.</em>
</div>

This decomposition matters because coordination is an information-routing problem. Consider a subagent that runs experiments. A weak contract might say, "return your findings." A task-specific contract can instead require the exact metrics, assumptions, failure cases, artifact paths, and evidence that the orchestrator will need to write the final report. The second contract passes less irrelevant context and more decision-relevant information.

Similarly, a hop can require a reviewer to run only after implementation and evaluation are complete, or trigger another experiment only when a stated acceptance criterion fails. A harness therefore controls not only *how much* computation is used, but *where the computation goes and what survives between stages*.

## Why optimize the harness locally?

In principle, we would like to search over many candidate harnesses, run every one, compare their outputs, and choose the best. For open-ended coding tasks, that becomes expensive very quickly. Each candidate requires a complete agent execution, and an all-pairs population comparison grows quadratically with the number of candidates.

RHI replaces this broad search with a local question:

> Did the current harness produce a better result than the immediately previous harness?

At each iteration, RHI needs only one new agent execution and one new pairwise comparison per task, plus the LLM call that writes the next harness. The previous output is cached. This makes the execution-and-comparison portion of each update constant in the population size, although the agent execution itself can still be substantial.

The trade-off is important. A comparison against one predecessor is a noisy local signal, not an unbiased estimate of global harness quality. RHI partly addresses this by saving all previous preference feedback in a **self-comparison history**. The optimizer uses that history as a kind of semantic momentum: later revisions can preserve changes that helped and reconsider changes associated with losses.

Under a standard latent-utility model of pairwise preferences, beating the previous harness is aligned with moving upward in the same underlying quality ordering. In practice, the evaluator is noisy and the textual search space is discrete, so this is best understood as **noisy local ascent**, not gradient descent and not a guarantee of monotonic improvement.

## The RHI loop

For a task $x$, let $H^{(i)}$ be the harness at iteration $i$, and let $y^{(i)}$ be the repository produced under that harness. One RHI iteration has four steps:

1. Run the fixed coding agent on the task with $H^{(i)}$, producing $y^{(i)}$.
2. Ask an LLM evaluator to compare $y^{(i)}$ with $y^{(i-1)}$.
3. Add the evaluator's preference feedback to the accumulated history.
4. Ask a separate LLM harness optimizer to revise $H^{(i)}$ using the current harness and that history.

The evaluation rubric is used to produce feedback, but it is not given directly to the harness optimizer. The optimizer must infer useful revisions from the history of comparisons.

<div align="center" style="margin: 28px 0;">
  <img src="../assets/rhi-algorithm.png" alt="The Recursive Harness Self-Improvement loop" width="950" style="max-width: 100%;">
  <br>
  <em>Each iteration compares the current output with the previous output, stores the preference, and uses the accumulated history to revise the harness.</em>
</div>

This is the sense in which RHI is recursive: a persistent system component is revised using evidence generated by its own earlier versions. The claim is intentionally bounded. RHI does **not** update model weights, rewrite the evaluator, or improve the harness optimizer itself. It recursively improves the task-specific harness around fixed models.

Treating the harness as a prompt-level object is also what makes the method practical for black-box agents. RHI does not need access to provider code or model internals; the evolved harness is appended to the original task prompt as a structured specification.

## How we evaluated it

We constructed 30 synthetic, open-ended machine-learning research tasks: ten each in quantitative finance, robotics, and pharmaceutical machine learning. The tasks were generated from domain-relevant job descriptions and ask the agent to produce a complete research repository, not a short answer. Standard deliverables include:

- a research report;
- reproducible code and documented entry points;
- metrics and experiment summaries;
- visualizations; and
- an index of generated artifacts.

These tasks do not have a single scalar answer. A repository may be stronger along several dimensions at once: task alignment, empirical rigor, reproducibility, presentation, deliverable coverage, and engineering quality. We therefore used pairwise LLM-as-a-judge evaluation over those criteria.

The experiments cover three progressively stronger base models: Claude Sonnet 4.6, Claude Opus 4.7, and Claude Opus 4.8. RHI starts from a `high`-reasoning agent and is compared with stronger same-family settings such as `xhigh`, `max`, and `ultracode`. Each comparison is repeated with three seeds. We also vary the evaluator model and reasoning setting: two evaluator configurations for the Sonnet 4.6 and Opus 4.8 experiments, and six for Opus 4.7.

This design asks a focused question: for a fixed model family, can improving the harness beat spending more inference compute inside the default harness?

## What we found

<div align="center" style="margin: 28px 0;">
  <img src="../assets/rhi-results.png" alt="RHI performance and normalized inference cost across Sonnet 4.6, Opus 4.7, and Opus 4.8" width="1000" style="max-width: 100%;">
  <br>
  <em>Few RHI iterations raise pairwise wins beyond the evaluated test-time-scaling baselines. The overlaid line reports the normalized cost of each RHI run; the dashed line marks the comparator's cost.</em>
</div>

### 1. A few harness updates lifted the observed test-time-scaling ceiling

For Sonnet 4.6, `high` with the harness after two RHI iterations won 20 of 30 comparisons against `max`. Later harness versions continued to outperform that baseline.

For Opus 4.7, the initial harness underperformed both `xhigh` and `max`. After a single RHI update, the `high`-reasoning agent surpassed both in the aggregated pairwise evaluation.

For Opus 4.8, two iterations were enough for `high` plus RHI to outperform all three evaluated baselines: `xhigh`, `max`, and `ultracode`. This last comparison is notable because `ultracode` already includes a provider-built dynamic multi-agent workflow. On this benchmark, a user-side harness specialized to the task outperformed that general-purpose system harness.

The conclusion is not that any prompt can beat any provider scaffold. It is narrower and more useful: **the harness can be a binding constraint on test-time scaling, and task-specific adaptation can move that constraint.**

### 2. Longer outputs do not explain the full gain

On Sonnet 4.6 and Opus 4.8, output-token use remained roughly flat across RHI iterations while performance continued to improve. That separates the observed gains from a simple "the agent wrote more" explanation.

For Opus 4.7, output tokens increased alongside performance, and the experiment contains only one update. The evidence there is inconclusive. The paper therefore makes the careful claim that longer generations are **not the primary explanation across the study**, not that RHI never uses additional tokens.

### 3. The evolved harnesses used context more efficiently

The strongest efficiency changes appeared in cache reads and writes:

- On Sonnet 4.6, the two-iteration RHI run cost 7% less than `max`, with 33% lower cache read/write usage.
- On Opus 4.7, the one-iteration RHI run cost 18% less than `max`, with 37% lower cache read/write usage.
- On Opus 4.8, the two-iteration RHI run cost 23% less than `max` and 60% less than `ultracode`. Cache read/write usage fell by 32% and 64%, respectively.

These quantities are normalized using the coding agent's internal cost accounting. They measure the evaluated execution under a given harness, not the cumulative cost of discovering that harness. RHI still spends one full agent run, one evaluation, and one harness-optimizer call on every update. The adaptation cost is easiest to justify when an evolved harness will be reused, when several iterations are affordable, or when the improved executions themselves are valuable as traces.

### 4. Harness improvement complements stronger models

RHI improved the weaker Sonnet 4.6 agent, but it did not consistently close the gap to the stronger Opus 4.7 baselines. Harness optimization is therefore not a substitute for train-time scaling. Better models and better harnesses are separate, complementary axes.

## Why does it work? The evidence points to information flow

The harness analysis begins with a simple observation: the first RHI update made the largest semantic move. The cosine similarity between consecutive full harnesses was 0.82 from the initial harness to the first revision, then 0.97, 0.98, and 0.99 for later transitions. RHI appears to make a large task-specific correction first and then refine it.

At the component level, **contracts showed the clearest task-dependent clustering** across embedding models and projection methods. Hops and instructions also specialized, but less distinctly; high-level roles changed more gradually. Contracts also stabilized earlier than the other components.

This fits the cost results. A contract acts like a learned sparsity pattern over inter-agent communication: it selects which evidence moves downstream instead of forwarding an entire interaction history. Hops decide when that evidence is produced and consumed. Better contracts and hops can therefore improve both solution quality and context efficiency.

The paper's analyses are diagnostic, not mechanistic. The coding agent is black-box, so text embeddings serve as an external proxy for semantic change. They show that the harness trajectory is systematic and task-dependent; they do not reveal the model's internal causal computation.

## A hypothesis about the objective behind RHI

RHI never evaluates an explicit loss for harness quality. The optimizer sees a textual harness and preference history, then proposes another textual harness. We nevertheless found that the trajectories are consistent with a compact information-theoretic objective.

Let $X$ denote the task and $Z_c$ the representation of harness component $c$. The hypothesis has two terms:

$$
J
=
\underbrace{\sum_{c\in\mathcal{C}_{\mathrm{ext}}} I(Z_c;X)}_{\text{task information in targeted components}}
-
\beta\,
\underbrace{\mathrm{TC}(\{Z_c\}_{c\in\mathcal{C}}\mid X)}_{\text{redundancy after conditioning on the task}},
\qquad \beta>0.
$$

The first term asks the components emphasized by the optimizer prompt—contracts and hops in our implementation—to carry more information about the task. The second asks the harness components to avoid duplicating one another once their shared task is accounted for.

Both terms are necessary. Maximizing task information alone has a trivial failure mode: copy the task description into every role, instruction, contract, and hop. The harness would be task-specific but badly organized. Reducing task-conditional redundancy encourages a division of labor: roles define expertise, instructions define behavior, contracts define what information moves, and hops define control flow.

Across two embedding models, with both raw and permutation-debiased estimates, task information increased monotonically in contracts and hops from iteration 1 to iteration 4. It decreased for roles and stayed roughly flat or declined for instructions. Over the same iterations, task-conditional total correlation decreased monotonically in every configuration.

The cleanest summary is:

> A better harness does not tell every component more. It puts task information in the components that route computation, while making the components less redundant with one another.

This objective is a **hypothesis**, not a loss that RHI explicitly optimizes and not proof of the optimizer model's latent objective. The measurements are embedding-based and correlational. Their value is that they turn a qualitative design intuition into a testable proposal for future harness optimizers.

## What the paper does—and does not—establish

The results support a promising case for task-specific harness adaptation, but their scope matters.

- The benchmark contains 30 synthetic ML research tasks. Broader claims require evaluation on real user workloads, other agent environments, and other model providers.
- Quality is measured by LLM pairwise judgment. Multiple judges, seeds, standardized deliverables, and a common rubric improve robustness, but they do not remove evaluator bias.
- The efficiency numbers describe individual executions of evolved harnesses. A deployment-level calculation must also account for the earlier executions and evaluations used during adaptation.
- The component analysis is correlational. It identifies a consistent pattern in the textual harnesses, not a causal mechanism inside the model.
- The study keeps model weights fixed. It does not test whether RHI-generated traces improve future models through post-training.

That final point defines the title's larger ambition. Agent harnesses are not only inference-time scaffolds; they are also data-generating systems. Their attempts, critiques, tool calls, failures, and successful solutions can become training data for future models. Better models can then support better harnesses, which can produce better traces again.

RHI studies the **first half** of that loop: how a fixed model can improve a user-constructed harness and thereby produce better task executions. Closing the loop—training future models on those traces and measuring the next cycle of improvement—remains future work.

## The takeaway

The usual scaling question is: *How much more reasoning should the model do?* RHI asks a prior question: *What structure should organize that reasoning?*

On the benchmark studied here, a small number of local, text-level harness updates improved performance beyond stronger test-time-scaling settings while reducing the cost of the final executions. The evidence points not to longer reasoning, but to better placement of information—especially through task-specific contracts and workflow hops.

As agent systems grow more capable, harnesses may not disappear. Their role may become more consequential: they decide how model capability is converted into action and what traces enter the next generation of learning.

---

<div align="center" style="margin: 28px 0;">
  <strong>Recursive Harness Self-Improvement</strong><br>
  Hyunin Lee<sup>1,2</sup>, Jinglue Xu<sup>1</sup>, Jeffrey Seely<sup>1</sup>, Donghyun Lee<sup>2</sup>, Matei Zaharia<sup>2</sup>, and Yujin Tang<sup>1</sup><br>
  <em><sup>1</sup>Sakana AI &nbsp;·&nbsp; <sup>2</sup>UC Berkeley</em>
</div>

**Paper and code:** links will be added here when the public release is available.

For questions or potential collaborations, please [reach out](mailto:hyunin@berkeley.edu).
