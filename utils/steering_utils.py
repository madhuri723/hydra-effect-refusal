import gc
import torch
from .model_utils import DEVICE


def format_prompt(tokenizer, prompt: str) -> str:
    """Apply the model's chat template to a raw instruction string."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )


def get_avg_activations(prompts: list[str], model, batch_size: int = 8) -> dict:
    """Return the mean last-token residual-stream activation per layer (on GPU)."""
    layer_sums: dict[int, torch.Tensor] = {}
    count = 0

    resid_filter = lambda name: "resid_post" in name
    get_layer   = lambda name: int(name.split(".")[1])

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            formatted = [format_prompt(model.tokenizer, p) for p in batch]
            inputs = model.tokenizer(formatted, padding="longest", return_tensors="pt").to(DEVICE)

            _, cache = model.run_with_cache(inputs.input_ids, names_filter=resid_filter)

            for key in cache:
                layer = get_layer(key)
                acts  = cache[key][:, -1, :].cpu()          # last token → CPU
                layer_sums[layer] = layer_sums.get(layer, 0) + acts.sum(dim=0)

            count += len(batch)
            del cache, inputs
            torch.cuda.empty_cache()

    return {k: (v / count).to(DEVICE) for k, v in layer_sums.items()}


def get_steering_vec(harmful: list[str], harmless: list[str], model, batch_size: int = 8) -> dict:
    """Compute the refusal direction as (mean harmful) − (mean harmless) per layer."""
    print(f"  harmful={len(harmful)}, harmless={len(harmless)}")
    mean_harm     = get_avg_activations(harmful, model, batch_size)
    mean_harmless = get_avg_activations(harmless, model, batch_size)
    return {layer: mean_harm[layer] - mean_harmless[layer] for layer in mean_harm}
