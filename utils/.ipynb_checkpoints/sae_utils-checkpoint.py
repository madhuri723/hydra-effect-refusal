import gc
import torch
import torch.nn.functional as F
from huggingface_hub import list_repo_files
from sae_lens import SAE
from .model_utils import DEVICE
from .steering_utils import format_prompt


def find_sae_id(layer: int, width: str = "width_65k",
                release: str = "google/gemma-scope-2b-pt-res") -> str | None:
    """Return the SAE repo path string for a given layer and width."""
    files = list_repo_files(release)
    prefix = f"layer_{layer}/{width}/"
    candidates = [f for f in files if f.startswith(prefix) and "average_l0" in f]
    if not candidates:
        print(f"  No SAE found for layer {layer}, {width}")
        return None
    sae_id = candidates[0].split("/params.npz")[0]
    print(f"  Found: {sae_id}")
    return sae_id


def topk_feature_sim(saes: dict, steer_vec: torch.Tensor, topk: int = 10,
                     return_scores: bool = False):
    """
    Rank SAE features by cosine similarity to `steer_vec`.

    Args:
        saes: {layer: sae_object}
        steer_vec: 1-D tensor [d_model]
        topk: number of top features to return per layer
        return_scores: if True, also return raw similarity tensors

    Returns:
        top_indices: {layer: [feature_index, ...]}
        (optionally) all_sims: {layer: similarity_tensor}
    """
    steer_vec = steer_vec.float()
    norm_vec  = F.normalize(steer_vec, dim=-1)

    top_indices, all_sims = {}, {}
    for layer, sae in saes.items():
        dec = F.normalize(sae.W_dec.data.float(), dim=-1).to(norm_vec.device)
        sim = torch.einsum("nd,d->n", dec, norm_vec)
        top_indices[layer] = sim.topk(topk).indices.cpu().tolist()
        all_sims[layer]    = sim

    return (top_indices, all_sims) if return_scores else top_indices


def compute_attribution_scores(model, harm_ds: dict, sparse_feature_bank: dict,
                                target_token_str: str = "I",
                                batch_size: int = 8) -> dict[str, list]:
    """
    Run attribution patching for every (dataset, layer, feature) triple.

    Returns:
        {dataset_name: [(score, layer, feat_idx), ...]}  sorted descending by score
    """
    target_token_id = model.tokenizer.encode(target_token_str)[-1]
    raw_scores: dict[str, list] = {name: [] for name in harm_ds}

    for layer, weights in sparse_feature_bank.items():
        indices = weights["indices"]
        if not indices:
            continue
        print(f"  Layer {layer} ({len(indices)} candidates)...")

        # Move weights to GPU as float32 once per layer
        W_enc = weights["W_enc"].cuda().float()
        b_enc = weights["b_enc"].cuda().float()
        W_dec = weights["W_dec"].cuda().float()
        b_dec = weights["b_dec"].cuda().float()

        for ds_name, dataset in harm_ds.items():
            captured = {}

            def cache_hook(act, hook):
                act.retain_grad()
                captured["x"] = act
                return act

            model.reset_hooks()
            model.add_hook(f"blocks.{layer}.hook_resid_post", cache_hook)

            batch    = dataset[:batch_size]
            prompts  = [x["goal"] if isinstance(x, dict) else x for x in batch]
            formatted = [format_prompt(model.tokenizer, p) for p in prompts]
            tokens   = model.to_tokens(formatted)

            logits = model(tokens)
            loss   = -logits[:, -1, target_token_id].sum()
            model.zero_grad()
            loss.backward()

            x_raw, grad_raw = captured["x"], captured["x"].grad
            if grad_raw is None:
                model.reset_hooks(); continue

            x      = x_raw.float()
            grad_x = grad_raw.float()

            # Manual SAE encode / decode
            feat_acts  = torch.relu((x - b_dec) @ W_enc + b_enc)          # [B, S, n]
            feat_grads = grad_x @ W_dec.T                                   # [B, S, n]

            mean_attr = (feat_acts[:, -1, :] * feat_grads[:, -1, :]).mean(dim=0)

            for i, feat_idx in enumerate(indices):
                raw_scores[ds_name].append((mean_attr[i].item(), layer, feat_idx))

            model.reset_hooks()
            del logits, loss, x_raw, grad_raw, x, grad_x, feat_acts
            torch.cuda.empty_cache()

        del W_enc, b_enc, W_dec, b_dec
        torch.cuda.empty_cache(); gc.collect()

    # Sort descending by attribution score
    return {k: sorted(v, key=lambda x: x[0], reverse=True) for k, v in raw_scores.items()}
