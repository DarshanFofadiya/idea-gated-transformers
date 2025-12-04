# Idea-Gated Transformers: Enforcing Semantic Coherence via Differentiable Vocabulary Pruning

**Author:** Darshan Fofadiya  
**Email:** fofadiyadarshan@gmail.com  
**Date:** December 2, 2025

Read the [arxiv](https://arxiv.org/abs/2512.03343) paper here

## Abstract

Autoregressive Language Models (LLMs) trained on Next-Token Prediction (NTP) often suffer from "Topic Drift" where the generation wanders away from the initial prompt due to a reliance on local associations rather than global planning [Holtzman et al., 2020]. While scaling model size mitigates this [Brown et al., 2020], the fundamental myopia of the NTP objective remains.

In this work, we introduce the **Idea-Gated Transformer**, a novel architecture that separates semantic planning from syntactic generation. We introduce an auxiliary "Idea Head" trained to predict the bag-of-words distribution for a future context window, creating a latent "Concept Vector" that actively gates the main vocabulary during generation. We propose a differentiable gating mechanism that suppresses semantically irrelevant tokens, effectively pruning the search space in real-time.

Experiments on WikiText-103 demonstrate that while the Idea-Gated model achieves comparable validation perplexity to a standard GPT-2 baseline, it exhibits significantly superior Domain Retention. Qualitative and quantitative analysis reveals that the gating mechanism successfully locks generation into specific semantic clusters (e.g., Finance, Science) and resists associative drift, offering a parameter-efficient path toward more controllable language modeling.

## Architecture

The Idea-Gated Transformer modifies the standard Decoder-only Transformer architecture [Radford et al., 2019]. It introduces a twin-head system on a shared backbone:

1.  **Token Head:** Predicts the immediate next token $x_{t+1}$ via standard cross-entropy loss on gated logits.
2.  **Idea Head:** A lightweight 2-layer MLP that predicts the presence of unique tokens in a future window $W=\{x_{t+1},...,x_{t+K}\}$ $(K=20)$ using BCE loss on multi-hot targets.
3.  **Soft Gating Mechanism:** Modulates Token Head logits with the Idea Head's sigmoid probabilities:

$$Gate = \alpha \cdot log(\sigma(z_{idea}) + \epsilon)$$
$$Gate_{clamped} = max(Gate, \beta)$$
$$z_{final} = z_{token} + Gate_{clamped}$$

Where $\alpha=0.5$ and $\beta=-2.0$ for tunable fluency vs. coherence. This decouples "System 2" planning (semantic bag-of-words) from "System 1" generation (syntactic tokens), inspired by dual-process theory [Kahneman, 2011].

##  Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/dfofadiya/idea-gated-transformers.git](https://github.com/dfofadiya/idea-gated-transformers.git)
    cd idea-gated-transformers
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

##  Usage

### 1. Train the Models
We provide scripts to train both the Baseline (Standard GPT-2) and the Idea-Gated model on the WikiText-103 dataset.

```bash
# Train Idea-Gated Model
python train.py -model_type gated -output_dir weights/gated

# Train Baseline Model
python train.py -model_type baseline -output_dir weights/baseline
```
### 2. Evaluation (Topic Retention)
To reproduce the "Stickiness" results from the paper, run the evaluation script. This tests the model's ability to stay on topic across Finance, War, and Science domains.

```bash
python evaluate.py \
  --baseline_path weights/baseline/model.pt \
  --gated_path weights/gated/model.pt
```

## Results
Our experiments on WikiText-103 show that the Idea-Gated model significantly outperforms the baseline in specialized domains by resisting topic drift.

| Domain    | Baseline Stickiness | Idea-Gated Stickiness | Improvement |
| :-------- | :------------------ | :-------------------- | :---------- |
| Chemistry | 8.2%                | 10.3%                 | +25.6%      |
| Hardware  | 0.8%                | 1.2%                  | +50.0%      |
| Medicine  | 3.9%                | 4.8%                  | +23.0%      |
| Finance   | 5.2%                | 4.5%                  | -13.0%      |

Stickiness is defined as the ratio of domain-specific terms generated per 100 tokens.

## Citation
If you use this code or architecture in your research, please cite:

```bibtex
@article{fofadiya2025ideagated,
  title={Idea-Gated Transformers: Enforcing Semantic Coherence via Differentiable Vocabulary Pruning},
  author={Fofadiya, Darshan},
  journal={arXiv preprint},
  year={2025}
}
