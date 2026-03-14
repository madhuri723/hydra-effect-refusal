"""
download_cache.py — Restore all cache files from Hugging Face Hub.
Run this at the start of a new RunPod session.

Usage:
    HF_TOKEN=hf_xxx python download_cache.py
"""

import os
from huggingface_hub import HfApi, hf_hub_download, login

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    raise ValueError("Set HF_TOKEN environment variable")

login(token=HF_TOKEN)
api = HfApi()

user   = api.whoami()["name"]
REPO_ID = f"{user}/hydra-effect-cache"
print(f"Downloading from: {REPO_ID}")

# ── Destination mapping ───────────────────────────────────────────────────────
DEST_MAP = {
    "workspace_cache/": "/workspace/cache/",
    "project_cache/":   "/workspace/hydra-effect-refusal/cache/",
    "plots/":           "/workspace/cache/",
    "hydra_effect_report.pdf": "/workspace/hydra_effect_report.pdf",
}

os.makedirs("/workspace/cache", exist_ok=True)
os.makedirs("/workspace/hydra-effect-refusal/cache", exist_ok=True)

# ── List all files in repo ────────────────────────────────────────────────────
all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")

for repo_path in all_files:
    if repo_path == ".gitattributes":
        continue

    # Determine local destination
    local_path = None
    for prefix, dest_dir in DEST_MAP.items():
        if repo_path.startswith(prefix):
            fname = os.path.basename(repo_path)
            local_path = os.path.join(dest_dir, fname) if dest_dir.endswith("/") else dest_dir
            break

    if local_path is None:
        print(f"  Skipping (no mapping): {repo_path}")
        continue

    if os.path.exists(local_path):
        print(f"  Already exists, skipping: {local_path}")
        continue

    print(f"  {repo_path} → {local_path} ...", end=" ", flush=True)
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=repo_path,
        repo_type="dataset",
        local_dir="/tmp/hf_cache_download",
    )
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    import shutil
    shutil.copy2(downloaded, local_path)
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"done ({size_mb:.1f} MB)")

print("\nCache restore complete. You're ready to continue experiments.")
