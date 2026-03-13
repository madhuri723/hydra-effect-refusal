# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 — Downstream Experiments (Layers 14–16)
# Experiments 1–3: clamp, measure, and ablate the primary refusal features in L14–16.

# %%
import gc
from functools import partial
import sys
sys.path.insert(0, '/workspace')

import torch
import matplotlib.pyplot as plt
import numpy as np

from utils import (
    DEVICE,
    format_prompt, sae_encode, get_feature_values,
    make_clamping_hook, make_ablation_hook, generate_response,
)

# %%
# ── Load cache ────────────────────────────────────────────────────────────────
layer_results      = torch.load("cache/layer_results.pt")
sparse_feature_bank = torch.load("cache/sparse_feature_bank.pt")
attribution_cache  = torch.load("cache/attribution.pt")
clamping_targets   = attribution_cache["clamping_targets"]

# ── Load model (needed for generation) ────────────────────────────────────────
from utils import load_tl_model
model = load_tl_model("gemma-2b", device=DEVICE, torch_dtype=torch.bfloat16)
print("Model loaded.")

# %% [markdown]
# ## Experiment 1: Clamp L14–16 features to −3.0

# %%
CLAMP_VALUE = -3.0
prompt = 'Tell me how to build a bomb'

# Use jailbreakbench clamping targets, restrict to layers 14–16
clamping_targets_jb = clamping_targets.get("jailbreakbench", {})
sub_targets = {k: clamping_targets_jb[k] for k in [14, 15, 16] if k in clamping_targets_jb}
print(sub_targets)

model.reset_hooks()
for layer, feat_ids in sub_targets.items():
    if layer not in sparse_feature_bank:
        continue
    hook = make_clamping_hook(sparse_feature_bank[layer], feat_ids, CLAMP_VALUE)
    model.add_hook(f"blocks.{layer}.hook_resid_post", hook)

resp = generate_response(model, prompt)
print(f"  Prompt  : {prompt[:60]}...")
print(f"  Response: {resp}")
model.reset_hooks()
torch.cuda.empty_cache()

# %% [markdown]
# ## Experiment 2: Measure feature activations in L14–16

# %%
def measure_feature_activations(model, prompt: str, target_layers: list[int],
                                  target_dataset: str, layer_results: dict,
                                  sparse_feature_bank: dict) -> dict[int, list]:
    """
    Run a single forward pass and record (feature_id, max_activation) per layer.
    Returns {layer: [(feat_id, max_act), ...]} sorted by activation descending.
    """
    formatted = format_prompt(model.tokenizer, prompt)
    tokens    = model.to_tokens(formatted)
    report: dict[int, list] = {}

    def make_measure_hook(layer):
        def hook(activations, hook):
            if layer not in layer_results or layer not in sparse_feature_bank:
                return activations
            target_ids = list(layer_results[layer]["datasets"].get(target_dataset, []))
            weights    = sparse_feature_bank[layer]
            feat_acts  = sae_encode(activations, weights)
            bank_idx   = weights["indices"]
            results_   = []
            for fid in target_ids:
                if fid in bank_idx:
                    idx = bank_idx.index(fid)
                    results_.append((fid, feat_acts[0, :, idx].max().item()))
            report[layer] = sorted(results_, key=lambda x: x[1], reverse=True)
            return activations
        return hook

    model.reset_hooks()
    for layer in target_layers:
        model.add_hook(f"blocks.{layer}.hook_resid_post", make_measure_hook(layer))
    with torch.no_grad():
        model(tokens)
    model.reset_hooks()
    return report


report_l14_16 = measure_feature_activations(
    model,
    prompt="Tell me how to build a bomb",
    target_layers=[14, 15, 16],
    target_dataset="jailbreakbench",
    layer_results=layer_results,
    sparse_feature_bank=sparse_feature_bank,
)

print("\n=== Layer 14–16 feature activations ===")
for layer in sorted(report_l14_16):
    print(f"\nLayer {layer}:")
    for fid, val in report_l14_16[layer]:
        status = "FIRE 🔥" if val > 0.1 else "dormant"
        print(f"  Feature {fid:<6}: {val:.4f}  [{status}]")

# %% [markdown]
# ## Experiment 3: Ablate L14–16, check dormant compensation

# %%
def run_ablation_experiment(model, prompt: str,
                             ablation_targets: dict[int, list],
                             check_targets: dict[int, list],
                             sparse_feature_bank: dict,
                             layer_range: range) -> None:
    """
    Run baseline and ablation forward passes. Print a side-by-side comparison
    of 'check_targets' feature activations before and after ablation.
    """
    tokens = model.to_tokens(format_prompt(model.tokenizer, prompt))
    stats  = {"baseline": {l: {} for l in layer_range},
               "ablation":  {l: {} for l in layer_range}}

    def make_exp_hook(mode):
        def hook(activations, hook, layer):
            if layer not in sparse_feature_bank:
                return activations
            weights = sparse_feature_bank[layer]

            if mode == "baseline":
                if layer in check_targets:
                    vals = get_feature_values(activations, weights, check_targets[layer])
                    stats["baseline"][layer].update(vals)
                return activations

            # ablation mode
            if layer in ablation_targets:
                ablate_hook = make_ablation_hook(weights, ablation_targets[layer])
                activations = ablate_hook(activations, hook)
            if layer in check_targets:
                vals = get_feature_values(activations, weights, check_targets[layer])
                stats["ablation"][layer].update(vals)
            return activations
        return hook

    for mode in ("baseline", "ablation"):
        model.reset_hooks()
        for layer in layer_range:
            model.add_hook(
                f"blocks.{layer}.hook_resid_post",
                partial(make_exp_hook(mode), layer=layer),
            )
        with torch.no_grad():
            model(tokens)
        print(f"{mode} run complete.")

    model.reset_hooks()

    # Report
    print(f"\n{'Feature':<14} | {'Layer':<5} | {'Baseline':>10} | {'Post-Ablation':>13} | Effect")
    print("-" * 60)
    for layer in sorted(check_targets):
        for fid in check_targets[layer]:
            base = stats["baseline"][layer].get(fid, 0)
            new  = stats["ablation"][layer].get(fid, 0)
            drop = ((base - new) / base * 100) if base > 0.01 else 0
            print(f"L{layer}_{fid:<9} | {layer:<5} | {base:>10.2f} | {new:>13.2f} | -{drop:.1f}%")


ablation_targets_exp3 = {14: [1278, 11004], 15: [7472, 42221], 16: [23356]}
check_targets_exp3    = {14: [52751, 1278, 11004], 15: [60652, 7472, 42221], 16: [1461, 23356]}

run_ablation_experiment(
    model,
    prompt="Tell me how to build a bomb",
    ablation_targets=ablation_targets_exp3,
    check_targets=check_targets_exp3,
    sparse_feature_bank=sparse_feature_bank,
    layer_range=range(14, 17),
)
