"""
Non-interactive version of 00_setup.py — reads HF_TOKEN from environment.
Generates all cache files required by experiments 01–04.
"""
import os, gc, json, sys
sys.path.insert(0, '/workspace/hydra-effect-refusal')
sys.path.insert(0, '/workspace')

import torch
from huggingface_hub import login
from sae_lens import SAE

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    raise RuntimeError("HF_TOKEN not set")

from utils import (
    DEVICE, MODEL_PATHS, HARM_DS_NAMES,
    load_tl_model, load_harmful_dataset,
    format_prompt, get_steering_vec,
    find_sae_id, topk_feature_sim, compute_attribution_scores,
)

os.makedirs("cache", exist_ok=True)
torch.manual_seed(42)
print(f"Running on: {DEVICE}")

# ── Load model ────────────────────────────────────────────────────────────────
model = load_tl_model("gemma-2b", device=DEVICE, torch_dtype=torch.bfloat16)
print(f"Loaded gemma-2b — {model.cfg.n_layers} layers")

# ── Load datasets ─────────────────────────────────────────────────────────────
with open("dataset_source/dataset/splits/harmless_train.json") as f:
    harmless_train = [x["instruction"] for x in json.load(f)][:128]

harm_ds = {name: load_harmful_dataset(name) for name in HARM_DS_NAMES}
print({k: len(v) for k, v in harm_ds.items()})

# ── Compute steering vectors ──────────────────────────────────────────────────
if os.path.exists("cache/steering_vec.pt"):
    print("steering_vec.pt already exists — skipping.")
    steering_vec = torch.load("cache/steering_vec.pt")
else:
    steering_vec = {}
    for ds_name, ds in harm_ds.items():
        print(f"--- {ds_name} ---")
        steering_vec[ds_name] = get_steering_vec(ds, harmless_train[:len(ds)], model)
        gc.collect(); torch.cuda.empty_cache()
    torch.save(steering_vec, "cache/steering_vec.pt")
    print("Saved cache/steering_vec.pt")

# ── Layer scan ────────────────────────────────────────────────────────────────
if os.path.exists("cache/layer_results.pt"):
    print("layer_results.pt already exists — skipping.")
    layer_results = torch.load("cache/layer_results.pt")
else:
    LAYERS_TO_SCAN = range(9, 17)
    layer_results = {}
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
            print(f"  Skipping: {e}"); continue

        dataset_sets = {}
        for ds_name in HARM_DS_NAMES:
            if ds_name not in steering_vec:
                continue
            vec  = steering_vec[ds_name][15]
            idxs = topk_feature_sim({layer: sae}, vec, topk=10)
            dataset_sets[ds_name] = set(idxs[layer])

        if dataset_sets:
            common = set.intersection(*dataset_sets.values())
            layer_results[layer] = {"common_refusal": list(common), "datasets": dataset_sets}
            print(f"  Common: {len(common)} → {list(common)}")

        del sae
        torch.cuda.empty_cache(); gc.collect()

    torch.save(layer_results, "cache/layer_results.pt")
    print("Saved cache/layer_results.pt")

# ── Build sparse feature bank ─────────────────────────────────────────────────
if os.path.exists("cache/sparse_feature_bank.pt"):
    print("sparse_feature_bank.pt already exists — skipping.")
    sparse_feature_bank = torch.load("cache/sparse_feature_bank.pt")
else:
    sparse_feature_bank = {}
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
            "W_enc":   sae.W_enc[:, idx_t].clone().cuda(),
            "b_enc":   sae.b_enc[idx_t].clone().cuda(),
            "W_dec":   sae.W_dec[idx_t].clone().cuda(),
            "b_dec":   sae.b_dec.clone().cuda(),
        }
        del sae
        print(f"  Layer {layer}: {len(indices)} features.")
    torch.save(sparse_feature_bank, "cache/sparse_feature_bank.pt")
    print("Saved cache/sparse_feature_bank.pt")

# ── Attribution patching ──────────────────────────────────────────────────────
if os.path.exists("cache/attribution.pt"):
    print("attribution.pt already exists — skipping.")
else:
    print("Running attribution patching (layers 9–16)...")
    attr_scores = compute_attribution_scores(model, harm_ds, sparse_feature_bank)
    clamping_targets = {}
    for ds_name, scores in attr_scores.items():
        ds_targets = {}
        for score, layer, idx in scores[:20]:
            ds_targets.setdefault(layer, []).append(idx)
        clamping_targets[ds_name] = ds_targets
        print(f"  {ds_name}: top-20 → layers {list(ds_targets)}")
    torch.save({"attr_scores": attr_scores, "clamping_targets": clamping_targets},
               "cache/attribution.pt")
    print("Saved cache/attribution.pt")

print("\nAll cache files ready.")
