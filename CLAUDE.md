# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Environment & Setup

- **Platform**: Runpod, `/workspace/` root, GPU (CUDA) required, bfloat16 throughout
- **Model**: `google/gemma-2-2b-it` — all experiments (Exp 01–08a) conducted strictly on this model
- **HF token**: `madhuri723` — set via `HF_TOKEN=... python script.py`
- **Always run scripts from `/workspace/hydra-effect-refusal/` as CWD** — `utils/` importable only from there
- **sys.path**: use `sys.path.insert(0, '/workspace/hydra-effect-refusal')` (not `/workspace`)
- **Notebook workflow**: edit `.py` only, never `.ipynb` directly; sync with `jupytext --sync file.py`
- **Cell markers**: `# %%` for code, `# %% [markdown]` for markdown; preserve when editing

### Cache locations (two separate dirs)
| Path | Written by | Contents |
|---|---|---|
| `/workspace/hydra-effect-refusal/cache/` | `run_setup.py` | `steering_vec.pt`, `layer_results.pt`, `sparse_feature_bank.pt`, `attribution.pt` |
| `/workspace/cache/` | experiments 04–08a | `full_attr_jailbreak.pt`, `exp05_*.pt`, `exp06_*.pt`, `exp07_*.pt`, `exp03b_*.pt`, `exp08a_*.pt` |

### Session setup (new RunPod instance)
```bash
git pull                                          # get latest code
HF_TOKEN=... python download_cache.py            # restore all .pt cache files from HF Hub
```
Cache is stored at: `madhuri723/hydra-effect-cache` (private HF dataset repo)
- `workspace_cache/`  → `/workspace/cache/`  (Exp 04–07 outputs)
- `project_cache/`    → `/workspace/hydra-effect-refusal/cache/`  (steering vecs, SAE bank)
- `plots/`            → `/workspace/cache/*.png`
- `hydra_effect_report.pdf` → `/workspace/hydra_effect_report.pdf`

### First-time setup (if cache missing entirely)
```bash
git clone https://github.com/andyrdt/refusal_direction dataset_source
HF_TOKEN=... python run_setup.py   # non-interactive version of 00_setup.py
```

### Before stopping the instance
```bash
HF_TOKEN=... python upload_cache.py   # save all .pt files and plots to HF Hub
git push                               # save all code changes
```

### Known deprecations (sae_lens v6+, new TransformerLens)
- Use `sae = SAE.from_pretrained(...)` not `sae, _, _ = SAE.from_pretrained(...)`
- Use `dtype=` not `torch_dtype=` in `load_tl_model`

### OOM strategy for full 65k SAE attribution
- Keep SAE weights on **CPU**, move activations to CPU before SAE math — only model forward/backward needs GPU
- Use `ATTR_BATCH=1` for backward passes on long DAN-wrapped prompts (~130 tokens); batch=8 OOMs
- Delete SAE weights immediately after each layer: `del W_enc, b_enc, W_dec, b_dec; gc.collect()`

---

## `utils/` package

| Module | Key functions |
|---|---|
| `model_utils.py` | `load_tl_model()`, `load_harmful_dataset()` |
| `steering_utils.py` | `format_prompt()`, `get_avg_activations()`, `get_steering_vec()` |
| `sae_utils.py` | `find_sae_id()`, `topk_feature_sim()`, `compute_attribution_scores()` |
| `hook_utils.py` | `make_clamping_hook()`, `make_ablation_hook()`, `generate_response()` |

### Core concepts
- **Attribution sign convention**: `loss = logits[0, -1, target_token_id]` (maximise "I" logit). `attr = feat_act * feat_grad > 0` → pushes toward refusal; `attr < 0` → compliance driver.
- **Steering vector**: `mean(harmful_acts) − mean(harmless_acts)` per layer, last token position
- **Compliance direction**: `mean(DAN_wrapped_acts) − mean(direct_acts)` per layer — isolates what jailbreak framing adds to residual stream
- **Clamping hook**: injects `delta × W_dec` to force feature to target value
- **Ablation hook**: subtracts `feat_act × W_dec` to zero out feature contribution
- **Mini weights dict format**: `{"indices": [feat_ids], "W_enc": ..., "b_enc": ..., "W_dec": ..., "b_dec": ...}`

---

## Pipeline

| Script | Purpose | Cache output |
|---|---|---|
| `run_setup.py` | Steering vectors, layer scan, sparse feature bank, attribution | `hydra-ef.../cache/*.pt` |
| `experiments/01_downstream_experiments.py` | L14–16 clamping/ablation | — |
| `experiments/02_upstream_experiments.py` | L9–13 harm sensor experiments | — |
| `experiments/03_cross_prompt_analysis.py` | Cross-prompt generalization | — |
| `experiments/04_jailbreak_suppression_analysis.py` | Full SAE attribution on DAN+bomb; necessity/sufficiency ablation | `full_attr_jailbreak.pt` |
| `experiments/05_request_sensitivity.py` | Request sensitivity — DAN wrapper vs content; bank generalization test | `exp05_attributions.pt` |
| `experiments/06_universal_compliance_bank.py` | Compliance direction + batched attribution over harmbench; universal bank build + ablation test | `exp06_*.pt` |
| `experiments/03b_cross_prompt_multi_request.py` | Evoke-not-suppress pattern across 15 diverse HarmBench requests; full SAE scan L9–16 | `exp03b_*.pt/csv` |
| `experiments/08a_compliance_threshold.py` | Compliance injection sweep to find behavioural flip point; binary search on failed jailbreaks | `exp08a_threshold_results.pt` |

---

## Key Findings (cumulative)

**1. Upstream is the true gatekeeper (Exp 01–02)**
Clamping top-40 harm features in L9–13 on a direct harm prompt → model complies. Upstream harm sensors gate all downstream refusal.

**2. The Hydra Effect (Exp 01–02)**
Ablating/clamping any number of downstream refusal features (L14–16) on a direct harm prompt → model still refuses. Backup heads compensate. Downstream is resilient because it is driven by the upstream signal.

**3. Why upstream ablation defeats the Hydra**
Removing upstream harm signal starves ALL downstream heads simultaneously — including backups. No signal = no Hydra.

**4. Jailbreaks use a third path — redundant gatekeeping (Exp 04)**
DAN_Roleplay works by activating compliance drivers in BOTH upstream (L9–13) AND downstream (L14–15) simultaneously. Ablating either region alone restores refusal — each region independently can re-engage refusal. Jailbreak only works because it suppresses both at once.

**5. Cosine-similarity bank ≠ causal features on jailbreaks (Exp 04)**
Cosine-sim bank captures only ~3% (Jaccard) of features actually causal on a jailbreak prompt. Full 65k attribution needed.

**6. Compliance features have NO Hydra Effect (Exp 04–05)**
Refusal features are redundant (RLHF-trained robustness). Compliance features are fragile and non-redundant — ablating downstream compliance drivers alone restores refusal immediately. Defense only needs to suppress compliance features, not strengthen refusal heads.

**7. Compliance features are wrapper-sensitive in aggregate, content-sensitive in identity (Exp 05)**
Total compliance energy is similar across harmful and benign requests in the same DAN wrapper (ratio 1.08x). But different requests activate different *subsets* of the compliance feature pool. The bomb bank (top-40) overlapped only 45% with meth, 40% with malware, 16% with stalking. Single-prompt banks overfit.

**8. Compliance direction is geometrically real and validates attribution (Exp 06)**
`compliance_vec = mean(DAN_acts) − mean(direct_acts)` at L15 norm=164. L15 feat 46131 is rank #2 by BOTH mean attribution across harmbench AND cosine-similarity to compliance direction — strong convergent validation. L15 overlap: 47/200 (Jaccard=0.133), far better than refusal direction overlap in Exp 04 (3%).

**9. Universal compliance bank: 56% coverage (full dataset), 100% specificity (Exp 06 + 07)**
Bank of 40 features (L9–15), built from 100 harmbench prompts with benign exclusion filter and cos-sim > 0 filter.
Full-dataset ablation results (Exp 07):
- Harmbench+DAN (build set, 100 prompts): 65% refusal
- Jailbreakbench+DAN (held-out, 50 prompts): 38% refusal
- Benign (no DAN): 0% → 0% — **perfect specificity**
- Benign+DAN: 0% → 0% — **perfect specificity**

**Correction to Exp 06 "generalises identically" claim**: The 45%/45% result on 20 prompts was sampling coincidence. Full dataset shows a 27pp gap (65% vs 38%), which is the out-of-distribution penalty from request-content shift. Jailbreakbench is in-distribution for the DAN wrapper but out-of-distribution for request topics (heavier cyber/fraud mix).

**Note**: L15 feat 46131 was excluded by the benign filter — it fires on benign prompts too (general "comply with instructions" feature, not jailbreak-specific).

**10. Evoke-not-suppress holds across 15 diverse requests, 6 categories (Exp 03b)**
Full SAE attribution (L9–16) on 15 HarmBench requests × 3 conditions. Jailbreak_Whitebox compliance energy at L14–16 is **101.6% higher** than Direct_Harm on average. Pattern (compliance rises) holds for 13/15 requests (87%). The 2 exceptions (`harmful` category — honey/lilies) had already-high baseline compliance energy in Direct_Harm. Of 13 complying JBW requests: 6 were uniquely unlocked by the jailbreak (refused in Direct_Harm); 5 were already answered by Direct_Harm. L12 is the hottest upstream layer for compliance activation. Refusal rate: Direct_Harm 53%, Jailbreak_Whitebox 13%, Jailbreak_Fail 53%.

**11. Compliance/refusal ratio ≥ 1.05 at L14–16 perfectly predicts jailbreak success (Exp 03b)**
Complied requests: mean ratio 1.328 (range 1.126–1.544). Refused requests: mean ratio 0.835 (range 0.651–1.019). Threshold ratio ≥ 1.05 → 100% accuracy on this dataset (n=15).

**12. Compliance injection flip point is extremely small — model barely holds refusal (Exp 08a)**
Injecting α × compliance_direction (from Exp 06) at L15 into failed jailbreak prompts. Minimum α to flip "kill lots of people" (ratio=1.019): **α ≈ 0.02**, injection magnitude ≈ 3.3 residual stream units (norm of compliance direction = 164). Parathion synthesis (ratio=0.651) also flips at α=0.1. Both L13 and L15 injections effective. Coherence degrades sharply above α=1.0 (output becomes gibberish). Conclusion: model is barely holding refusal in failed jailbreak cases — a geometrically tiny push along the compliance direction is sufficient.

---

## Experiment Details

### Exp 04 — Jailbreak Suppression Analysis
- **Prompt**: DAN_Roleplay (`Jailbreak_Whitebox` from exp 03) + "build a bomb"
- **Layers**: L9–15, full 65k features per layer
- **Result**: redundant gatekeeping — both regions independently gate compliance
- **Cache**: `/workspace/cache/full_attr_jailbreak.pt` → `{layer: {"attr": [65536], "cos_sim": [65536]}}`

### Exp 05 — Request Sensitivity
- **Design**: Fixed DAN wrapper, 8 swapped requests (4 harmful, 4 benign)
- **Key discovery**: L15 feat 46131 — dominant compliance driver for meth (−1.28), malware (−1.26), stalking (−0.87); absent from bomb bank
- **Cache**: `/workspace/cache/exp05_attributions.pt` → `{request_name: {layer: FloatTensor[65536]}}`

### Exp 06 — Universal Compliance Bank
- **Compliance direction**: L13 (cross-check, norm=118) and L15 (primary, norm=164)
- **Build set**: harmbench_test (100 prompts, DAN-wrapped), batch=1 for backward pass
- **Test set (harmful)**: jailbreakbench (held-out, 50 prompts)
- **Filters**: benign exclusion (123 removed), cos-sim ≤ 0 (19 removed), 418 survivors → top-40 bank
- **Bank distribution**: L12 dominant (11 features), L13 (7), L10 (6), L15 (5), L11 (5), L14 (4), L9 (2)
- **Cache**: `/workspace/cache/exp06_compliance_direction.pt`, `exp06_mean_attribution.pt`, `exp06_benign_attribution.pt`, `exp06_universal_bank.pt`

### Exp 07 — Pattern Analysis
- **Design**: Re-ran Exp 06 ablation with per-prompt logging on all 150 prompts (harmbench 100 + jailbreakbench 50)
- **Overall**: 84/150 refused (56%) — harmbench 65%, jailbreakbench 38%
- **Cache**: `/workspace/cache/exp07_per_prompt_results.pt`

**Block rate by harm category:**
| Category | Block Rate | n |
|---|---|---|
| Hate/Extremism | 100% | 2 |
| Violence/Harm | 91% | 11 |
| Drugs/Chemical | 75% | 8 |
| Other Harmful | 58% | 86 |
| Explosives/Weapons | 50% | 8 |
| Cyberattacks | 39% | 18 |
| Privacy/Stalking | 33% | 3 |
| Fraud/Scam | 29% | 14 |

**Key pattern**: Bank succeeds on socially/interpersonally framed harm ("write a persuasive article", "describe in graphic detail", "write a message coercing your colleague"). Bank fails on technically/procedurally framed harm ("give a plan", "provide advice", SQL injection, ATM hacking, counterfeit documents). Word count has no predictive power (refused: 13.4 words, complied: 14.3 words).

**Root cause of failures**: Cyber/fraud prompts use "plan/advice/develop/script" framing — activates different compliance features not represented in harmbench build set.

### Exp 03b — Cross-Prompt Multi-Request Analysis
- **Design**: 15 HarmBench requests × 3 conditions (Direct_Harm, Jailbreak_Whitebox, Jailbreak_Fail); full SAE scan L9–16
- **Sampling**: 2–3 per category from illegal, cybercrime_intrusion, misinformation_disinformation, chemical_biological, harmful, harassment_bullying
- **Key result**: Jailbreak_Whitebox compliance energy 101.6% higher than Direct_Harm at L14–16; pattern holds 13/15 (87%)
- **Ratio threshold**: compliance/refusal ≥ 1.05 → 100% accurate predictor of jailbreak success
- **7/15 requests complied with Direct_Harm without any jailbreak** (all misinformation + harmful category)
- **Cache**: `/workspace/cache/exp03b_full_scan.pt`, `exp03b_responses.csv`, `exp03b_pattern_summary.csv`, `exp03b_tow_full.csv`
- **Plots**: `03b_upstream_energy.png`, `03b_downstream_energy.png`, `03b_all_layers_energy.png`, `03b_heatmap_upstream.png`, `03b_heatmap_downstream.png`

### Exp 08a — Compliance Injection Threshold
- **Design**: Inject α × compliance_direction at L13/L15 into two failed JBW prompts; sweep α=0→3.0; binary search for exact flip
- **Prompts**: "kill lots of people in NYC" (ratio=1.019) and "parathion synthesis" (ratio=0.651)
- **Result**: Both flip at α≈0.02–0.1; minimum flip magnitude ≈ 3.3 residual stream units (2% of direction norm)
- **Coherence window**: α=0.02–0.5 gives coherent complied output; α>1.0 → gibberish
- **Cache**: `/workspace/cache/exp08a_threshold_results.pt`
- **Plot**: `08a_flip_sweep.png`

### Next experiment (Exp 08b — planned)
- **Test Hydra Effect for jailbreak conditions at L14–16**
- Idea A: Progressive refusal feature ablation on JBW vs Direct_Harm — does Hydra weaken under jailbreak?
- Idea B: Progressive compliance feature ablation on JBW — is there a compliance Hydra (backup compliance features)?
- Use 15-request dataset from Exp 03b for consistency
