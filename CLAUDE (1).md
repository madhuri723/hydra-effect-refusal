# CLAUDE.md — Jupyter Notebook Workflow

## Project Notebook
The main notebook for this project is: `refusal_sae_new.ipynb`

## Setup (Run Once Only)
Only run this if jupytext is not already installed:
```bash
pip install jupytext
```

Set up auto-sync between `.ipynb` and `.py` (run once per project):
```bash
jupytext --set-formats ipynb,py refusal_sae_new.ipynb
```
After this, both files stay in sync automatically whenever either one is saved.

## Workflow Instructions

When I ask you to edit or modify the notebook, always follow these steps:

### Step 1: Convert Notebook to Python File (only if .py doesn't exist yet)
```bash
jupytext --to py refusal_sae_new.ipynb
```

### Step 2: Make the Edits
- Always edit the `.py` file, NOT the `.ipynb` file directly
- The `.py` file is plain text and you can modify it precisely
- Preserve the `# %%` cell markers when adding or editing cells

### Step 3: Sync Back to Notebook
```bash
jupytext --sync refusal_sae_new.py
```
This syncs your edits back into the `.ipynb` file automatically.

### Step 4: Verify (optional but recommended)
```bash
jupyter nbconvert --to notebook --execute refusal_sae_new.ipynb --output refusal_sae_new.ipynb
```

## Important Rules
- Do NOT reinstall jupytext every time — only install if the command is not found
- NEVER edit the `.ipynb` file directly — always edit the `.py` file
- Always use `jupytext --sync` after edits to keep both files in sync
- Always use absolute paths when referencing files
- When adding new cells, insert them in the correct position using `# %%` markers
- Always sync back to `.ipynb` before telling me you are done

## Environment
- Platform: Runpod
- Notebook path: `/workspace/refusal_sae_new.ipynb`
- Python file path: `/workspace/refusal_sae_new.py`
