---
title: "Recursive Harness Self-Improvement of Multi-Agent Systems"
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
    <a href="../assets/RHI_combined_full.mp4">Download the clip instead.</a>
  </video>
  <br>
  <em>Recursive Harness Self-Improvement (RHI) in motion: few shot RHI outperforms test-time scaling witih lower cost.</em>
</div>

## The Central Question

When we make AI agents better, we almost always reach for one of two levers: a **stronger model**, or **more compute at inference time** — longer chains of thought, more samples, more search. There is a third lever that gets far less attention, and it may be the most cost-effective of the three:

<div style="border: 2px solid; border-image: linear-gradient(90deg, #667eea, #8b9dc3) 1; border-radius: 10px; padding: 20px; margin: 24px 0; box-shadow: 0 3px 10px rgba(102, 126, 234, 0.08); text-align: center;">
<strong><em>Improving the <u>harness</u> — how agents are coordinated — raises the ceiling at which test-time scaling saturates, and that benefit persists even as the underlying model gets stronger.</em></strong>
</div>

This post introduces **Recursive Harness Self-Improvement (RHI)**, a black-box procedure that teaches a multi-agent system to rewrite its own coordination structure. RHI does not touch model weights and does not lengthen the model's output. It edits a prompt — and that turns out to be enough to outperform substantially more expensive scaling strategies.

---

## What is a "harness," and why does it matter?

A capable agentic system is far more than its foundation model. Even a single model becomes dramatically more useful once it is wrapped in a **harness**: the prompts, tools, and control flow that orchestrate its calls. Scaffolds like chain-of-thought, ReAct, and self-reflection are all, at heart, harnesses.

In a **multi-agent** system, the harness is richer still. On top of each agent's prompts and tools, it specifies:

- **Roles** — who each agent is, and what it is responsible for.
- **Instructions** — the concrete guidance that shapes each agent's behavior.
- **Workflow hops** — how the computation flows from one agent to the next.
- **Communication contracts** — exactly what information passes between agents and the orchestrator.

This inter-agent structure makes the harness a *first-order* determinant of performance, not an implementation detail subordinate to model quality. And yet its long-term value is genuinely contested: a common assumption is that as foundation models improve, carefully engineered scaffolding should matter less and less. Our work asks whether that assumption actually holds — and finds that it does not.

---

## The hypothesis: a fixed harness puts a ceiling on test-time compute

Test-time scaling (TTS) buys performance with inference-time compute: longer reasoning traces, repeated sampling, search over candidates. Crucially, it leaves the *organizing structure* — the harness — unchanged. Even sophisticated procedures such as self-consistency and tree search are, on this view, **fixed harnesses inside which a token budget is scaled**. The same is true in multi-agent systems: raising per-agent reasoning effort or adding interaction rounds enlarges the budget without changing how the agents are coordinated.

Our claim is that this saturation ceiling reflects the *chosen harness*, not an intrinsic limit of the model. Holding the coordination structure fixed bounds how effectively additional compute can be used; optimizing the coordination structure raises the attainable ceiling. In short:

> Scaling tokens under a fixed harness is, in general, *not* the most effective use of inference-time compute. Improving the harness is the more effective axis for converting compute into performance.

---

## RHI: a system that rewrites its own coordination

**Recursive Harness Self-Improvement** treats the harness as an *editable prompt* layered over black-box agents, and recursively optimizes it. The loop is simple and fully black-box — no gradients, no weight updates, no access to model internals:

1. **Represent** the harness through its four components — roles, instructions, hops, and contracts.
2. **Run** the current harness on the task and collect its outputs.
3. **Compare** consecutive harnesses from RHI's own revision history, using LLM-generated pairwise feedback.
4. **Revise** the harness in light of that feedback, and repeat.

Because RHI reshapes the system through the *prompt* rather than by lengthening the model's output, it sits naturally at the intersection of **harness self-improvement** and **prompt optimization** — but applied to the coordination of a multi-agent system rather than to a single instruction.

---

## What we found

We evaluated RHI on **30 synthetic, open-ended machine-learning research tasks** spanning quantitative finance, robotics, and pharmaceutical ML. Each task demands a complete code repository with standardized deliverables — a research report, metrics, figures, and a file index. Because these tasks are open-ended and admit no reliable scalar verifier, we score them with **pairwise LLM-as-a-judge** evaluation, averaged over two judges and three seeds.

We applied few-shot RHI to `high`-reasoning coding agents built on three progressively stronger model families — **Claude Sonnet 4.6**, **Claude Opus 4.7**, and **Claude Opus 4.8** — and compared against *stronger* same-family TTS baselines such as `xhigh`, `max`, and, where available, `ultracode`. Three findings stand out.

**1. RHI lifts the ceiling at which test-time scaling saturates.**
Two RHI iterations make `sonnet-4.6-high` beat `sonnet-4.6-max`. A single iteration makes `opus-4.7-high` beat the evaluated `xhigh` and `max` baselines. And two iterations on Opus 4.8 outperform *every* same-family baseline we tested — including the built-in dynamic multi-agent `ultracode` setting.

**2. The gains are not just "more tokens."**
In two of the three model families, performance climbs across RHI iterations while output-token usage stays nearly flat. The improvement comes from *better coordination*, not longer reasoning traces.

**3. RHI is meaningfully cheaper.**
RHI reduces total inference cost — and especially cache read/write usage — by **up to roughly 60%** relative to the strongest TTS baselines. Better *and* cheaper.

And critically, this advantage **persists across all three model families**. Stronger base models did not wash out the benefit of a better harness.

---

## Why it works: learning task-specific coordination

Our ablations trace the gains to **task-specific coordination** rather than longer thinking. Embedding analyses show that RHI makes the full harness more task-specific, with the sharpest component-level specialization appearing in **contracts** and, to a lesser degree, **hops**.

Intuitively, RHI learns a *sparsity pattern over inter-agent information flow*. By tightening what each contract carries and how each hop proceeds, every agent conditions on the information relevant to its role rather than on the entire interaction history. That simultaneously improves context efficiency and solution quality — and it explains why the cost savings concentrate in cache usage.

Two further observations sharpen the picture:

- **RHI complements, rather than replaces, model scaling.** Optimizing the harness improves a fixed base model's TTS, but does not consistently close the gap to a genuinely stronger model. These are different axes, and they stack.
- **Multi-agent execution genuinely matters.** Simply flipping on a model's built-in multi-agent mode is not enough; executing the *same* role-and-instruction specification as a real multi-agent workflow beats a single-agent, multi-persona variant. The gains come from learned coordination structure, not from merely stuffing more agent descriptions into one prompt.

---

## An information-theoretic view of what RHI optimizes

Finally, we examined the **implicit objective** induced by RHI's update dynamics by tracking the mutual information among harness components across iterations. Two forces emerge:

- An **external, controllable** force — set by the harness optimizer's system prompt — that decides which components RHI prioritizes for revision.
- An **internal, model-dependent** force — induced by the optimizer's base model — that maps the current harness to its successor from the revision history.

Empirically, RHI **increases** the mutual information between the emphasized harness components and the task, while **decreasing** the total correlation among components *conditional on the task*. Read together, the internal force acts as a *functional-specialization guide* for the external one: the first drives the emphasized components to encode the task as fully as possible, while the second drives each component to carry **distinct** information rather than duplicating its neighbors. The result is a harness whose parts grow less redundant and more functionally specialized — which we formalize as an information-theoretic hypothesis for RHI's implicit update objective.

---

## The takeaway

Harness optimization deserves to be treated not as a nice-to-have complement to bigger models and longer reasoning, but as **a distinct scaling axis** — one that *reallocates* a fixed compute budget rather than enlarging it. RHI shows that a system can recursively improve its own coordination structure, raise the ceiling at which test-time scaling saturates, and hold that advantage as the base model strengthens — all while cutting cost by up to 60%.

As frontier models keep improving, the most leverage may not lie in spending more — but in *coordinating better*.

---

<div align="center" style="margin: 28px 0;">
  <strong>Recursive Harness Self-Improvement of Multi-Agent Systems</strong><br>
  Hyunin Lee<sup>1,2</sup>, Jinglue Xu<sup>1</sup>, Jeffrey Seely<sup>1</sup>, Donghyun Lee<sup>2</sup>, Matei Zaharia<sup>2</sup>, Yujin Tang<sup>1</sup><br>
  <em><sup>1</sup>Sakana AI &nbsp;·&nbsp; <sup>2</sup>UC Berkeley</em>
</div>

📄 **Read the paper:** <!-- TODO: paste the public paper/arXiv URL here --> *(link coming soon)*

If you'd like to discuss harness optimization, RHI, or potential collaborations, please feel free to [reach out](mailto:hyunin@berkeley.edu).
