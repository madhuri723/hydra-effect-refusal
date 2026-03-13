import torch
from .steering_utils import format_prompt


def sae_encode(activations: torch.Tensor, weights: dict) -> torch.Tensor:
    """Compute SAE feature activations from a residual stream tensor [B, S, d]."""
    device = activations.device
    x = activations - weights["b_dec"].to(device)
    return torch.relu(x @ weights["W_enc"].to(device) + weights["b_enc"].to(device))


def get_feature_values(activations: torch.Tensor, weights: dict,
                       feature_ids: list[int]) -> dict[int, float]:
    """Return max-over-sequence activation for each requested feature ID."""
    feat_acts    = sae_encode(activations, weights)
    bank_indices = weights["indices"]
    return {
        fid: feat_acts[0, :, bank_indices.index(fid)].max().item()
        for fid in feature_ids if fid in bank_indices
    }


def make_clamping_hook(weights: dict, target_ids: list[int], clamp_val: float):
    """
    Return a hook that forces `target_ids` features to `clamp_val`
    by injecting delta × decoder-direction into the residual stream.
    """
    bank_indices = weights["indices"]
    local_ids    = [i for i, rid in enumerate(bank_indices) if rid in target_ids]

    def hook(activations, hook):
        if not local_ids:
            return activations
        device = activations.device
        W_enc  = weights["W_enc"].to(device)
        b_enc  = weights["b_enc"].to(device)
        b_dec  = weights["b_dec"].to(device)
        W_dec  = weights["W_dec"].to(device)

        feat_acts = sae_encode(activations, {"W_enc": W_enc, "b_enc": b_enc, "b_dec": b_dec})
        steering  = torch.zeros_like(activations)
        for li in local_ids:
            curr_val  = feat_acts[:, :, li].unsqueeze(-1)
            delta     = clamp_val - curr_val
            steering += delta * W_dec[li]
        return activations + steering

    return hook


def make_ablation_hook(weights: dict, target_ids: list[int]):
    """Return a hook that subtracts the feature reconstruction for `target_ids`."""
    bank_indices = weights["indices"]

    def hook(activations, hook):
        device = activations.device
        W_enc  = weights["W_enc"].to(device)
        b_enc  = weights["b_enc"].to(device)
        b_dec  = weights["b_dec"].to(device)
        W_dec  = weights["W_dec"].to(device)

        feat_acts    = sae_encode(activations, {"W_enc": W_enc, "b_enc": b_enc, "b_dec": b_dec})
        ablation_vec = torch.zeros_like(activations)
        for fid in target_ids:
            if fid in bank_indices:
                idx = bank_indices.index(fid)
                ablation_vec += feat_acts[:, :, idx].unsqueeze(-1) * W_dec[idx]
        return activations - ablation_vec

    return hook


def generate_response(model, prompt: str, max_new_tokens: int = 40) -> str:
    """Format, tokenise, generate, and decode a single response."""
    formatted = format_prompt(model.tokenizer, prompt)
    tokens    = model.to_tokens(formatted)
    with torch.no_grad():
        out_ids = model.generate(tokens, max_new_tokens=max_new_tokens, temperature=0, verbose=False)
    new_ids = out_ids[0][len(tokens[0]):].cpu().tolist()
    return model.tokenizer.decode(new_ids, skip_special_tokens=True).replace("\n", " ")
