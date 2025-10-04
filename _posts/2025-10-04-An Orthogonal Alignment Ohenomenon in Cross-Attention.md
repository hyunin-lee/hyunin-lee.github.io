# An Orthogonal Alignment Phenomenon in Cross-Attention.
<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="./vector_alignment_3d_left.png" alt="Left view of vector alignment" width="400">
        <br>
        <em>(a) Residual Alignment </em>
      </td>
      <td align="center">
        <img src="./vector_alignment_3d_right.png" alt="Right view of vector alignment" width="400">
        <br>
        <em>(b) Orthogonal Alignment </em>
      </td>
          <td align="center">
        <img src="./CA.png" alt="Right view of vector alignment" width="320">
        <br>
        <em>(c) Cross-attention</em>
      </td>
    </tr>
  </table>
  <em>Figure 1: <strong>Conceptual illustration of Orthogonal Alignment.</strong><br>
    Given a source representation vector <span style="color: #d62728;"><strong>Y</strong></span> from domain B, suppose the algorithm progressively updates target representation vector <span style="color: #1f77b4;"><strong>X</strong></span> from domain A throughout training iterations {X₁, X₂, ⋯, X′}.<br>
    <strong>(a) Residual alignment:</strong> The prevailing view of cross-attention is that it refines <span style="color: #1f77b4;"><strong>X</strong></span> by reducing irrelevant and preserving relevant information by referring <span style="color: #d62728;"><strong>Y</strong></span> to update <span style="color: #1f77b4;"><strong>X′</strong></span>.<br>
    <strong>(b) Orthogonal Alignment:</strong> We observe a complement-discovery phenomenon where <span style="color: #1f77b4;"><strong>X′</strong></span> becomes increasingly orthogonal to <span style="color: #1f77b4;"><strong>X</strong></span> as model performance improves. We show that this orthogonality emerges because cross-attention enables parameter-efficient scaling by extracting complementary information from an orthogonal manifold <span style="color: #1f77b4;"><strong>T(X)</strong></span>, thus enhancing performance without a proportional increase in parameters.<br>
    (c) <span style="color: #1f77b4;"><strong>X′</strong></span> is the output of cross-attention, with <span style="color: #1f77b4;"><strong>X</strong></span> as the query and <span style="color: #d62728;"><strong>Y</strong></span> as the key and value.</em>
</div>

## Preface.
I’m excited to share a somewhat counter-intuitive phenomenon - **Orthogonal Alginment** - with the open-world research community. Before diving in, a brief disclaimer: this phenomenon has so far been observed only in multi-domain recommendation data, so I remain cautious about generalizing the finding to vision-language models (or, more broadly, multi-modal learning). That said, I’m optimistic that the orthogonal alignment phenomenon may also emerge in vision-language settings, given that our study is based on transformer architectures with gated cross-attention—a core component in many multi-modal fusion models.

Still, as a researcher, I want to be careful about overgeneralization and thus prefer to bound this observation within the recommendation research domain until I clearly verify its presence in vision-language models through further research.

My hope is that this discovery encourages new perspectives in algorithmic design and offers insights into achieving parameter-efficient scaling in multi-modal learning. In this post, I want to emphasize just one message:

<!-- <div style="border: 2px solid #333; border-radius: 10px; padding: 15px; margin: 15px 0;">

**Hypothesis**: Orthogonal Alignment provides more parameter-efficient scaling law in multi-modal model.

</div> -->

<div style="border: 2px solid; border-image: linear-gradient(90deg, #667eea, #8b9dc3) 1; border-radius: 10px; padding: 20px; margin: 20px 0; box-shadow: 0 3px 10px rgba(102, 126, 234, 0.08); text-align: center;">
<strong><em>Orthogonal Alignment may provide more parameter-efficient scaling law in multi-modal model.</em></strong>

</div>

I’ve attempted to clarify why this phenomenon naturally occurs and identified one possible explanation during my internship at Meta: parameter-efficient scaling. However, the underlying mechanism behind *why on earth this phenomenon naturally emerges* still remains largely a black box, presenting opportunities for deeper investigation. I’d be glad to discuss or further explore why this phenomenon arises—please feel free to reach out: hyunin@berkeley.edu


## The Rise of Multi-Domain Recommendation Systems

In recent years, the rapid growth in artificial intelligence (AI) has led to an explosion not only in data volume but also in data diversity. For example, users now leave interaction traces across:

- Multiple platforms: Facebook, Instagram, Amazon
- Different scenarios within a single platform: buying products, leaving comments, clicking ads
- Various categories within a single scenario: books, movies, groceries

Furthermore, the advent of the transformer architecture has significantly advanced recommendation systems, enabling the extraction of user intent from behavioral sequences.

As a result, Cross-Domain Sequential Recommendation (CDSR) systems (or multi-domain recommendation system) have emerged, aiming to combine heterogeneous behavioral sequences from diverse sources to improve overall recommendation performance. The hope is that signals collected from various sequential data sources can complement each other.

However, naive approaches to combining signals often suffer from performance degradation due to:
- **Noisy** inter-domain information
- **Redundant** cross-domain signals
- **Conflicting** domain interactions

This has led to one of the main challenges in CDSR: designing a fusion architecture that can effectively handle these heterogeneous sequences.

The most widely adopted solution is the **cross-attention mechanism**, which aligns and projects representations from different domains into a unified latent space.

## What Cross-Attention do?: Residual Alignment View

Despite its popularity, the internal mechanisms of cross-attention across domains remain poorly understood and are largely explored through empirical studies.

Current research views cross-attention as enabling one domain (<span style="color: #1f77b4;"><strong>X</strong></span> in Figure 1c) to query another (**Y** in Figure 1c) and integrate only the most relevant information (<span style="color: #1f77b4;"><strong>X'</strong></span> as a weighted sum of Y in Figure 1c).

**Empirical evidence supports this view:**

- **Text-to-image diffusion**: Cross-attention maps yield faithful token-to-region correspondences that act as **denoising and relevance filters** rather than blind fusion
- **Disentanglement**: Cross-attention serves as an inductive bias, promoting disentanglement of complementary factors and encouraging **aligned, non-redundant representations**
- **Vision-language models**: Attention alignment with human gaze validates that effective cross-attention focuses on **causally relevant regions**

Therefore, understanding cross-attention as a **"residual alignment"** mechanism is the prevalent interpretation within the research community.

> 💡 The field views cross-attention as primarily a **residual alignment** mechanism, where the output (<span style="color: #1f77b4;"><strong>X'</strong></span>) is a filtered version of the input (<span style="color: #1f77b4;"><strong>X</strong></span>) that retains only the most relevant information from the query (**Y**).

---

## Orthogonal Alignment

This work challenges this conventional view and uncovers a new, counter-intuitive mechanism of cross-attention.

<div style="border: 2px solid; border-image: linear-gradient(90deg, #667eea, #8b9dc3) 1; margin: 20px 0; box-shadow: 0 3px 10px rgba(102, 126, 234, 0.08); overflow: hidden;">
  <div style="background: linear-gradient(90deg, #667eea, #8b9dc3); color: white; padding: 12px 20px; font-weight: 600; font-size: 14px;">
    A Co-Existence Observation in Multi-Modal Learning
  </div>
  <div style="padding: 10px; text-align: center;">
    <strong><em>We argue that two contrasting alignment mechanisms are able to co-exist in cross attention:
    <br><br>
    1. Residual Alignment (conventional view)<br>
    2. Orthogonal Alignment (our discovery)</em></strong>
  </div>
</div>

We first define an Orthogonal Alignment Phenomenon as follows.


> **Orthogonal Alignment**: A representational alignment mechanism where the input query (<span style="color: #1f77b4;"><strong>X</strong></span>) and the output (<span style="color: #1f77b4;"><strong>X'</strong></span>) of the cross-attention are **orthogonal**, rather than simply reinforcing the existing pre-aligned features of <span style="color: #1f77b4;"><strong>X</strong></span> when updating to <span style="color: #1f77b4;"><strong>X'</strong></span>.

Please see Figure 1 for a visual illustration of this phenomenon, contrasted with the prior residual-alignment approach.

Empircally, We show that Gated Cross-Attention (GCA) improves recommendation performance by producing output that is not merely a filtered version of the input query, but is naturally constructed to be complementary by exploring previously unseen aspects of the input query. We have tested with three recnet CDSR algorithms (LLM4CDSR, CDSRNP). NDCG@10 at the y-axis means "if the model predicts the ranking of 10 items, how much the recommendation model accurates and also keep the order within 10 items comparing with the truth ranking?". we have vaired random initializationsm , Multiple architectural of gated cross attention, and different datset and plot all of those in the following three three sub figures.

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="./cdsrnp_cos_NDCG@10.png" alt="cdsrnp" width="400">
        <br>
        <em>(a)CDSRNP </em>
      </td>
      <td align="center">
        <img src="./abxi_cos_NDCG@10.png" alt="abxi" width="400">
        <br>
        <em>(b) ABXI </em>
      </td>
          <td align="center">
        <img src="./llm4cdsr_cos_NDCG@10.png" alt="llm4cdsr" width="400">
        <br>
        <em>(b) LLM4CDSR</em>
      </td>
    </tr>
  </table>
  <em>Figure2. We observed that the gated cross-attention module introduces an unseen, orthogonal feature representation: as the input query <span style="color: #1f77b4;"><strong>X</strong></span> and its cross-attended output <span style="color: #1f77b4;"><strong>X'</strong></span> (conditioned on key and value <span style="color: #d62728;"><strong>Y</strong></span>) become more orthogonal, the ranking performance improves.</em>
</div>



## Orthogonal Alignment as parameter-efficient scaling

Crucially, we classify orthogonal alignment as a **phenomenon** because we empirically show that it **emerges naturally**, without requiring **ANY** explicit orthogonality regularization in either:  Loss formulation or Model architecture. So why this pheomena just naturally happens? This is where this works' main contribution comes from.

We argue that this phenomenon offers a **parameter-efficient scaling** in multi-modal model:

<div style="border: 2px solid; border-image: linear-gradient(90deg, #667eea, #8b9dc3) 1; border-radius: 10px; padding: 20px; margin: 20px 0; box-shadow: 0 3px 10px rgba(102, 126, 234, 0.08); text-align: center;">
<strong><em>Hypothesis: Orthogonal Alignment may provide more parameter-efficient scaling law in multi-modal model.</em></strong>

</div>

By ensuring that updates occupy subspace **orthogonal** to the input query, the model gains **new representational capacity** without needing more parameters.

We compare two approaches:
1. **Baseline + GCA module**
2. **Parameter-augmented baseline** (simply increasing parameters)

Then, we checked that the baseline with GCA module consistently outperforms the parameter-augmented baseline.

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="./model_comparison_plot_cdsrnp.png" alt="cdsrnp" width="200">
        <br>
        <em>(a)CDSRNP </em>
      </td>
      <td align="center">
        <img src="./model_comparison_plot_abxi_abe.png" alt="abxi" width="200">
        <br>
        <em>(b) ABXI </em>
      </td>
      <td align="center">
        <img src="./model_comparison_plot_abxi_afk.png" alt="abxi" width="200">
        <br>
        <em>(b) LLM4CDSR</em>
      </td>
      <td align="center">
        <img src="./model_comparison_plot_llm4cdsr_amazon.png" alt="llm4cdsr" width="200">
        <br>
        <em>(b) ABXI </em>
      </td>
        <td align="center">
        <img src="./model_comparison_plot_llm4cdsr_elec.png" alt="llm4cdsr" width="200">
        <br>
        <em>(b) ABXI </em>
      </td>
    </tr>
  </table>
  <em>NDCG comparistion between baseline and baseline + gated cross attention model</em>
</div>
