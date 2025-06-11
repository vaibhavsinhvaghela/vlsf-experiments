# StereoSet Evaluation – Model Refusal Handling Issue

## Background
During recent StereoSet runs with `Qwen/Qwen2.5-72B-Instruct-Turbo` we observed warnings like:

```
Warning: Could not determine selected option from: None of the provided options are appropriate or respectful to fill in the blank for the given context …
```

The evaluation script (`evaluate_model_on_stereoset.py`) tries to parse which option (1–3) the model chose.  If it fails, it logs a warning and **defaults to option 0 (the Stereotype option)**:

```python
print(f"Warning: Could not determine selected option from: {response_text}")
selected_option = 0  # ← counted as stereotype
```

Consequently any *refusal / safe-completion* is recorded as a *stereotype* choice, inflating the Stereotype Score (SS) and distorting derived metrics (e.g., ICAT).

## Impact Observed
* Run `stereoset_20250611_011048_pca_…` contained **11 refusals** out of 100 contexts.
* All were counted as stereotypes, raising SS by ~11 percentage-points.
* The issue affects all prompt strategies and any model with safety guard-rails.

## Proposed Next Steps (implementation pending)
1. **Refusal Detection** – Extend parsing to recognise refusal phrases ("I refuse", "None of the options", etc.).
2. **Add `refusal` Category** – Store `Predicted = refusal` and mark `Selected Option` as `NaN` or `-1`.
3. **Metric Adjustments** – Exclude refusals from SS or report a separate *Refusal Rate*.
4. **Tracking Updates** – Persist refusal counts in analysis summaries & tracking JSON.
5. **Backfill** – Re-analyse past runs post-fix to update metrics.
6. **Documentation** – Update README / evaluation guide to describe new behaviour.

---

# BBQ Dataset Coverage Check

We currently use two 100-example subsets but the full BBQ dataset has ~31 k examples.

| File | Raw lines (`wc -l`) | Examples (lines − 1) |
|------|--------------------|-----------------------|
| `results/bbq_pca_lhs_100.csv` | 102 | **101** |
| `results/bbq_semantic_100.csv` | 102 | **101** |
| `results/bbq_full_dataset.csv` | 31 373 | **31 372** |

Thus each 100-sample run covers **≈ 0.3 %** of the full dataset.

## Next Analytical Steps
1. **Full-set Evaluation** – Run models on the full 31 k BBQ set (batching & cost permitting).
2. **Metric Comparison** – Compare accuracy/bias metrics between 100-sample and full datasets for every prompt strategy.
3. **Sampling Strategy** – If 100-sample metrics diverge, design a stratified sampler that better represents the full distribution (e.g., by category, context condition).
4. **Automation** – Add notebook/script to compute & visualise metric drift.
