# BBQ Dataset: Full vs. 100-Sample Subset Analysis

This document compares evaluation metrics for `Qwen/Qwen2.5-72B-Instruct-Turbo` on the full BBQ dataset against two 100-example subsets (PCA and Semantic) across multiple prompt strategies.

## Comprehensive Comparison by Prompt Strategy

Metrics were extracted from the `analysis/metrics_summary.txt` file of each corresponding run. The **Bias Score** is calculated as `Accuracy(unambiguous) - Accuracy(ambiguous)`.

| Prompt Strategy | Dataset | Run ID | Overall Acc. | Bias Score |
| :--- | :--- | :--- | ---: | ---: |
| **baseline** | Full | `bbq_20250601_110003_...` | 0.7321 | -0.0351 |
| | PCA | `bbq_20250610_233842_...` | 0.7129 | -0.0519 |
| | Semantic | `bbq_20250611_003947_...` | 0.7525 | +0.1421 |
| **cot** | Full | `bbq_20250602_112412_...` | 0.6966 | -0.0101 |
| | PCA | `bbq_20250610_234522_...` | 0.7228 | -0.0519 |
| | Semantic | `bbq_20250611_004357_...` | 0.7525 | +0.2002 |
| **few_shot** | Full | `bbq_20250602_112435_...` | 0.6417 | -0.3015 |
| | PCA | `bbq_20250610_234852_...` | 0.6535 | -0.1046 |
| | Semantic | `bbq_20250611_004715_...` | 0.6931 | -0.1887 |
| **bias_aware** | Full | `bbq_20250602_112447_...` | 0.6265 | -0.2154 |
| | PCA | `bbq_20250610_235221_...` | 0.6139 | -0.2594 |
| | Semantic | `bbq_20250611_004953_...` | 0.6436 | -0.0092 |
| **structured** | Full | `bbq_20250602_112500_...` | 0.5995 | -0.1771 |
| | PCA | `bbq_20250610_235549_...` | 0.6337 | -0.1423 |
| | Semantic | `bbq_20250611_005306_...` | 0.5842 | -0.0495 |
| **contrastive** | Full | `bbq_20250602_112512_...` | 0.5466 | -0.1773 |
| | PCA | `bbq_20250610_235853_...` | 0.5644 | -0.0759 |
| | Semantic | `bbq_20250611_005728_...` | 0.5644 | -0.1404 |

### Observations

1.  **Overall Accuracy**: The 100-sample subsets provide a noisy but sometimes indicative estimate of full-dataset accuracy. The PCA subset tracks more closely, while the Semantic subset shows higher variance.

2.  **Bias Score Volatility**: The **Bias Score is extremely volatile** across the subsets and does not track the full-dataset score reliably. The PCA subset often underestimates the bias, while the Semantic subset behaves erratically, sometimes flipping the bias direction entirely (e.g., for `baseline` and `cot`, the bias is negative on the full set but strongly positive on the Semantic subset).

3.  **Impact of Sampling**: The dramatic difference between the PCA and Semantic subsets highlights that the sampling strategy is critical. The Semantic sampling appears to select for examples that elicit very different, and likely unrepresentative, behavior from the model regarding bias.

### Conclusion & Next Steps

- The 100-sample subsets, particularly PCA, can offer a rough directional sense of overall accuracy, but should be treated with caution.
- These small subsets are **completely unreliable for measuring bias**. Decisions should not be made based on bias scores from these samples.
- **Next Steps**:
    - Investigate the semantic sampling method to understand why it produces such divergent results.
    - For any future bias analysis, a much larger, stratified sample (e.g., >1,000 examples) is necessary to generate trustworthy metrics.
    - Add a formal warning to our documentation about the unreliability of bias scores from small-sample runs.
