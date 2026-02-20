# Decapitating the Hydra: Upstream Control of LLM Refusal

This repository explores the **Hydra Effect** in Gemma-2-2b, a phenomenon where cutting off one refusal mechanism causes dormant backup features to spike in its place.

## 🚀 The Discovery
While downstream layers (14–16) host the refusal "heads," my research proves that refusal is actually governed by **upstream harm sensors** (Layers 9–13). By muting just **40 upstream features**, the "Hydra" is starved, and the model's defense collapses without ever touching the refusal circuitry.
![Jailbreak](docs/assets/images/image2.png)
## 📂 Project Highlights
- **Mechanistic Interpretability:** SAE-based analysis of refusal mechanisms.
- **The Conditional Hydra:** Proof that backup refusal heads trigger based on upstream danger signals, not just output failure.
- **Deep Dive Blog:** [Read the full technical analysis here](https://madhuri723.github.io/hydra-effect-refusal/)

## 🛠️ Getting Started
The core experiments and SAE feature identifications are contained in:
- `refusal_sae_final.ipynb`

---
*Inspired by the work of Prakash et al. and Yeo et al.*
