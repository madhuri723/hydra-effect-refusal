# Decapitating the Hydra: Upstream Control of LLM Refusal

Mechanistic interpretability research on refusal mechanisms in Gemma-2-2b-it, using sparse autoencoders to identify and manipulate the features governing refusal and jailbreak behavior.

## Findings

**Refusal is governed upstream, not downstream.** While downstream layers (14–16) host the refusal features that produce a model's "I cannot" response, refusal is actually controlled by upstream harm sensors in layers 9–13. Muting 40 of these upstream features collapses the model's refusal behavior without touching the downstream refusal circuitry at all.

**Jailbreaks overpower refusal rather than suppress it.** In successful DAN-style jailbreaks, harm features aren't suppressed — they activate at even higher intensity than on direct harmful requests. Jailbreaks instead succeed by activating a separate set of compliance features that outweigh the refusal signal. On a small trial set of 15 HarmBench requests, a compliance/refusal energy ratio at or above 1.05 predicted jailbreak success with 100% accuracy. Separately, ablating a bank of 40 compliance features blocked 56% of tested DAN-style jailbreaks across 150 prompts (100 HarmBench, 50 JailbreakBench), with zero false positives on benign prompts. Neither result has been tested against other jailbreak styles or at larger scale yet.

![Jailbreak](docs/assets/images/image2.png)

## Project Highlights
- SAE-based mechanistic analysis of refusal and compliance mechanisms in Gemma-2-2b-it
- Causal confirmation via activation steering — muting or injecting specific features shifts model behavior in the predicted direction
- Full writeups: [Decapitating the Hydra](https://madhuri723.github.io/hydra-effect-refusal/2026/02/19/hydra-deep-dive.html) and [Jailbreaks Overpower Refusal: Compliance Features](https://madhuri723.github.io/hydra-effect-refusal/2026/03/17/compliance-features.html)

## Getting Started

Requires a GPU. Install dependencies in this order:
```bash
pip install transformer_lens --no-deps
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The full experiment set lives in `experiments/`, meant to be run in order:
- `00_setup.ipynb` — model and dataset setup
- `01_downstream_experiments.ipynb` through `08a_compliance_threshold.ipynb` — the complete experiment sequence

Cached activations and intermediate results are stored on a private Hugging Face dataset for reproducing this specific research — not publicly downloadable. Running the experiments from scratch on the provided code and prompts will regenerate the same results.

---
*Inspired by the work of Prakash et al. and Yeo et al.*
