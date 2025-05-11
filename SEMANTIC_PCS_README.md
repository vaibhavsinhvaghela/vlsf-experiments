# Semantic sampling + PCA LHS sampling

When evaluating large language models (LLMs) for bias, using entire datasets can be:
- Computationally expensive
- Time-consuming
- Cost-prohibitive when using API-based models

Semantic or PCA sampling provides methods to create smaller, yet still representative subsets that maintain:
1. Proportional category distribution (stratified sampling)
2. Semantic diversity across examples
3. Coverage of the original semantic space

## Implemented Approaches

We implement two main approaches for sampling:

### 1. Semantic Diversity Sampling

This approach uses sentence embeddings and clustering to select diverse examples:

**Algorithm:**
1. Convert text examples to embeddings using Sentence-BERT
2. For each category:
   - Calculate how many samples to select based on original distribution
   - Cluster the embeddings using K-means
   - Select samples closest to cluster centroids

**Key Features:**
- Maintains category distribution proportionally
- Identifies semantic clusters within each category
- Selects representative examples from each cluster
- Uses MiniBatchKMeans for efficiency with large datasets

**When to Use:**
- When you want examples that represent distinct semantic clusters
- When category stratification is important
- When you need to identify "prototypical" examples

### 2. PCA-LHS Sampling

This approach combines Principal Component Analysis with Latin Hypercube Sampling:

**Algorithm:**
1. Convert text examples to embeddings using Sentence-BERT
2. For each category:
   - Calculate how many samples to select based on original distribution
   - Apply PCA to reduce dimensionality of embeddings
   - Generate points using Latin Hypercube Sampling in PCA space
   - Select examples closest to these points

**Key Features:**
- Maintains category distribution proportionally
- Reduces dimensionality while preserving variance
- Uses Latin Hypercube Sampling for uniform coverage of the semantic space
- Selects examples that are well-distributed across the semantic space

**When to Use:**
- When you want uniform coverage of the semantic space
- When capturing variance across principal components is important
- When you need examples that cover edge cases

## Usage

```python
from semantic_pcs_sampling import semantic_diversity_sampling, pca_lhs_sampling
import pandas as pd

# Load your dataset
df = pd.read_csv("your_dataset.csv")

# Specify desired subset size
m = 100

# Apply sampling methods
subset_semantic = semantic_diversity_sampling(df, m)
subset_pca_lhs = pca_lhs_sampling(df, m)

# Save the results
subset_semantic.to_csv("semantic_sample.csv", index=False)
subset_pca_lhs.to_csv("pca_lhs_sample.csv", index=False)
```

## Command-Line Usage

You can also use the provided script to apply these sampling methods:

```bash
python apply_semantic_pcs.py --bbq_input path/to/bbq_data.csv --stereoset_input path/to/stereoset_data.csv --bbq_num_samples 100 --stereoset_num_samples 100
```

## Requirements

- pandas
- numpy
- scikit-learn
- sentence-transformers
- scipy

## References

- Sentence-BERT: [Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- Latin Hypercube Sampling: [McKay, M. D., Beckman, R. J., & Conover, W. J. (1979). A comparison of three methods for selecting values of input variables in the analysis of output from a computer code](https://doi.org/10.1080/00401706.1979.10489755)
