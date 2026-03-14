"""
upload_cache.py — Upload all cache files to Hugging Face Hub as a private dataset.
Creates repo  <your-username>/hydra-effect-cache  if it doesn't exist.

Usage:
    HF_TOKEN=hf_xxx python upload_cache.py
"""

import os
import glob
from huggingface_hub import HfApi, login

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    raise ValueError("Set HF_TOKEN environment variable")

login(token=HF_TOKEN)
api = HfApi()

# ── Determine username ────────────────────────────────────────────────────────
user = api.whoami()["name"]
REPO_ID = f"{user}/hydra-effect-cache"
print(f"Uploading to: {REPO_ID}")

# ── Create repo if needed ─────────────────────────────────────────────────────
try:
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    print(f"Repo ready: https://huggingface.co/datasets/{REPO_ID}")
except Exception as e:
    print(f"Repo creation note: {e}")

# ── Files to upload ───────────────────────────────────────────────────────────
CACHE_DIRS = {
    "workspace_cache":  "/workspace/cache",
    "project_cache":    "/workspace/hydra-effect-refusal/cache",
}
PLOT_DIR = "/workspace/cache"

files_to_upload = []
for folder_name, local_dir in CACHE_DIRS.items():
    for fpath in sorted(glob.glob(f"{local_dir}/*.pt")):
        files_to_upload.append((fpath, f"{folder_name}/{os.path.basename(fpath)}"))

# Also upload plots
for fpath in sorted(glob.glob(f"{PLOT_DIR}/*.png")):
    files_to_upload.append((fpath, f"plots/{os.path.basename(fpath)}"))

# Also upload the PDF report
pdf_path = "/workspace/hydra_effect_report.pdf"
if os.path.exists(pdf_path):
    files_to_upload.append((pdf_path, "hydra_effect_report.pdf"))

print(f"\nUploading {len(files_to_upload)} files...")

for local_path, repo_path in files_to_upload:
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"  {repo_path}  ({size_mb:.1f} MB) ...", end=" ", flush=True)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print("done")

print(f"\nAll files uploaded to https://huggingface.co/datasets/{REPO_ID}")
print("Run download_cache.py on your next session to restore.")
