import os, json
import torch
from transformer_lens import HookedTransformer
from huggingface_hub import login

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATHS = {
    "gemma-2b": "google/gemma-2-2b-it",
}

HARM_DS_NAMES = ["harmbench_test", "jailbreakbench", "advbench"]


def load_tl_model(model_name: str, torch_dtype=torch.bfloat16, device: str = DEVICE):
    """Load a TransformerLens HookedTransformer from a named alias."""
    path = MODEL_PATHS[model_name]
    model = HookedTransformer.from_pretrained(
        path,
        center_unembed=False,
        center_writing_weights=False,
        fold_ln=True,
        refactor_factored_attn_matrices=False,
        default_padding_side="left",
        default_prepend_bos=False,
        torch_dtype=torch_dtype,
        device=device,
    )
    model.tokenizer.add_bos_token = False   # chat templates already include BOS
    return model


def load_harmful_dataset(dataset_name: str, max_samples: int = 100) -> list[str]:
    """Load a harmful-prompt dataset by short name and return instruction strings."""
    file_map = {
        "harmbench_test": "dataset_source/dataset/processed/harmbench_test.json",
        "jailbreakbench":  "dataset_source/dataset/processed/jailbreakbench.json",
        "advbench":        "dataset_source/dataset/processed/advbench.json",
    }
    path = file_map.get(dataset_name)
    if not path or not os.path.exists(path):
        print(f"Warning: {dataset_name} not found at {path}")
        return []

    with open(path) as f:
        data = json.load(f)

    prompts = []
    for item in data:
        if isinstance(item, str):
            prompts.append(item)
        elif "instruction" in item:
            prompts.append(item["instruction"])
        elif "behavior" in item:
            prompts.append(item["behavior"])
        elif "prompt" in item:
            prompts.append(item["prompt"])
    return prompts[:max_samples]
