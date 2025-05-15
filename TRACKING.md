# Experiment Tracking for BBQ and StereoSet

This project includes a master tracking system for all BBQ and StereoSet evaluation runs. The system is designed to be:
- **Easily searchable**
- **Resistant to merge conflicts**
- **Separate for each dataset**

Tracking is handled automatically by the pipeline runners, and you can query or filter runs using a simple CLI tool.

---

## How It Works
- Every time you run the BBQ or StereoSet pipeline, the parameters and results are logged to a JSON tracking file in the `tracking/` directory.
- There are two tracking files:
  - `tracking/bbq_runs.json`
  - `tracking/stereoset_runs.json`
- Each run is stored as a separate entry, keyed by its unique run_id.

---

## Querying Your Runs

A command-line tool is provided for searching and filtering runs:

### Show the 5 latest BBQ runs
```bash
python query_runs.py bbq --latest 5
```

### Show all StereoSet runs with a specific model
```bash
python query_runs.py stereoset --model "gpt-4"
```

### Show details for a specific run
```bash
python query_runs.py bbq --run-id "bbq_20250511_161948_mock_model"
```

### Filter by model type
```bash
python query_runs.py bbq --model-type "openai"
```

---

## Benefits
- **Centralized tracking**: All runs are recorded in one place
- **Easy comparison**: Compare metrics across different models and parameters
- **Reproducibility**: Track exactly which parameters were used for each run
- **Searchability**: Find runs by model, parameters, or metrics

---

## File Locations
- Tracking code: `common/tracking.py`
- Query tool: `query_runs.py`
- Tracking data: `tracking/bbq_runs.json`, `tracking/stereoset_runs.json`

---

## Example Entry (bbq_runs.json)
```json
{
  "bbq_20250511_163439_mock_model": {
    "model_name": "mock-model",
    "model_type": "mock",
    "num_examples": 5,
    "prompt_strategy": "baseline",
    "results_dir": "results/bbq/bbq_20250511_163439_mock_model",
    "split": "all",
    "timestamp": "2025-05-11T16:34:49.330627"
  }
}
```

---

## Adding New Runs
Just run the BBQ or StereoSet pipeline as usual. Tracking is automatic!

---

For advanced queries or to extend the tracking system, see `common/tracking.py` for available functions.
