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
# # 00 — Setup
# Runs once. Loads model, computes steering vectors, builds SAE feature bank,
# runs attribution patching, and saves all heavy artifacts to `/workspace/cache/`.

# %%
import os, gc, getpass
import sys
sys.path.insert(0, '/workspace')

import torch
from huggingface_hub import login
from sae_lens import SAE

from utils import (
    DEVICE, MODEL_PATHS, HARM_DS_NAMES,
    load_tl_model, load_harmful_dataset,
    format_prompt, get_steering_vec,
    find_sae_id, topk_feature_sim, compute_attribution_scores,
)

torch.manual_seed(42)
print(f"Running on: {DEVICE}")

# %%
# ── HuggingFace auth ──────────────────────────────────────────────────────────
os.environ["HF_TOKEN"] = getpass.getpass("Enter Hugging Face Token: ")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("Warning: HF_TOKEN not set. Private model access may fail.")

# %%
# ── Load model ────────────────────────────────────────────────────────────────
model_name = "gemma-2b"
model = load_tl_model(model_name, device=DEVICE, torch_dtype=torch.bfloat16)
NUM_LAYERS = model.cfg.n_layers
print(f"Loaded {model_name} — {NUM_LAYERS} layers")

# %%
# ── Load datasets ─────────────────────────────────────────────────────────────
with open("dataset_source/dataset/splits/harmless_train.json") as f:
    import json
    harmless_train = [x["instruction"] for x in json.load(f)][:128]

harm_ds = {name: load_harmful_dataset(name) for name in HARM_DS_NAMES}
print({k: len(v) for k, v in harm_ds.items()})

# %%
# ── Compute steering vectors ──────────────────────────────────────────────────
steering_vec: dict[str, dict[int, torch.Tensor]] = {}
for ds_name, ds in harm_ds.items():
    print(f"--- {ds_name} ---")
    current_harmless = harmless_train[: len(ds)]
    steering_vec[ds_name] = get_steering_vec(ds, current_harmless, model)
    gc.collect(); torch.cuda.empty_cache()
print("Done.")

torch.save(steering_vec, "cache/steering_vec.pt")
print("Saved cache/steering_vec.pt")

# %%
# ── Layer scan: find common refusal features per layer ────────────────────────
LAYERS_TO_SCAN = range(9, 17)
layer_results: dict[int, dict] = {}

for layer in LAYERS_TO_SCAN:
    print(f"\n--- Layer {layer} ---")
    sae_id = find_sae_id(layer)
    if sae_id is None:
        continue

    try:
        sae, _, _ = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res", sae_id=sae_id, device=DEVICE
        )
    except Exception as e:
        print(f"  Skipping: {e}")
        continue

    # Top-10 per dataset
    dataset_sets: dict[str, set] = {}
    for ds_name in HARM_DS_NAMES:
        if ds_name not in steering_vec:
            continue
        vec  = steering_vec[ds_name][15]   # use layer-15 vector as reference
        idxs = topk_feature_sim({layer: sae}, vec, topk=10)
        dataset_sets[ds_name] = set(idxs[layer])

    if dataset_sets:
        common = set.intersection(*dataset_sets.values())
        layer_results[layer] = {"common_refusal": list(common), "datasets": dataset_sets}
        print(f"  Common refusal features: {len(common)} → {list(common)}")

    del sae
    torch.cuda.empty_cache(); gc.collect()

print("\nScan complete.")
torch.save(layer_results, "cache/layer_results.pt")
print("Saved cache/layer_results.pt")

# %%
# ── Build sparse feature bank ─────────────────────────────────────────────────
sparse_feature_bank: dict[int, dict] = {}

print("Harvesting candidate feature weights...")
for layer, data in layer_results.items():
    indices = data["common_refusal"]
    if not indices:
        continue

    sae_id = find_sae_id(layer)
    sae, _, _ = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res", sae_id=sae_id, device="cpu"
    )

    idx_t = torch.tensor(indices)
    sparse_feature_bank[layer] = {
        "indices": indices,
        "W_enc":   sae.W_enc[:, idx_t].clone().cuda(),   # [d_model, n_cands]
        "b_enc":   sae.b_enc[idx_t].clone().cuda(),       # [n_cands]
        "W_dec":   sae.W_dec[idx_t].clone().cuda(),       # [n_cands, d_model]
        "b_dec":   sae.b_dec.clone().cuda(),               # [d_model]
    }

    del sae
    print(f"  Layer {layer}: {len(indices)} features extracted.")

print("Extraction complete.")
torch.save(sparse_feature_bank, "cache/sparse_feature_bank.pt")
print("Saved cache/sparse_feature_bank.pt")

# %%
# ── Attribution patching ──────────────────────────────────────────────────────
print("Running attribution patching (layers 9–16)...")
attr_scores = compute_attribution_scores(model, harm_ds, sparse_feature_bank)

# Derive clamping targets (top 20 per dataset)
clamping_targets: dict[str, dict[int, list]] = {}
for ds_name, scores in attr_scores.items():
    ds_targets: dict[int, list] = {}
    for score, layer, idx in scores[:20]:
        ds_targets.setdefault(layer, []).append(idx)
    clamping_targets[ds_name] = ds_targets
    print(f"  {ds_name}: top-20 selected → layers {list(ds_targets)}")

torch.save({"attr_scores": attr_scores, "clamping_targets": clamping_targets},
           "cache/attribution.pt")
print("Saved cache/attribution.pt")
print("\nSetup complete. All cache files written.")
