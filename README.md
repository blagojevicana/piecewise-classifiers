# piecewise-classifiers
A machine learning project exploring piecewise classifiers and segmented decision functions for improved classification performance.

## Overview 
This project implements piecewise classification using the Super-k method, where the input feature space is partitioned into k disjoint regions and a separate classifier is trained within each region.
The final prediction is obtained by routing each sample to its corresponding region-specific model.

This approach allows complex decision boundaries to be approximated through a set of simpler local classifiers.

## Motivation
Standard global classifiers often struggle with:
- highly non-linear decision boundaries
- heterogeneous data distributions
- locally varying class separation
  
Piecewise models address these issues by decomposing the problem into smaller, more manageable sub-tasks while maintaining interpretability.

## Super-K Method
The Super-k method consists of the following steps:
1. Feature Space Partitioning
   
The input space is divided into k regions using a predefined segmentation strategy (e.g. thresholds, clustering, or geometric rules).

3. Local Model Training

A separate classifier is trained on data belonging to each region.

5. Inference Routing
   
During prediction, each sample is assigned to its region and classified using the corresponding local model.

7. Aggregation
   
Predictions are combined into a single output without overlap between regions.

## Experiments
Experiments are conducted on:
- synthetic datasets with controlled decision boundaries
- real-world tabular datasets

Evaluation metrics include:
- accuracy
- precision / recall
- confusion matrix
- region-wise performance analysis

## Results
The Super-k piecewise classifier demonstrates:
- improved performance on datasets with heterogeneous distributions
- more interpretable local decision boundaries
- stable training compared to a single global model

## Technologies
- Python 3.x
- NumPy
- scikit-learn
- Matplotlib

## Limitations
- Performance depends on quality of space partitioning
- Fixed regions may not adapt optimally to all datasets
- Increased training complexity for large k
  
## Future Work
- Learn partitions automatically (clustering-based Super-k)
- Compare against ensemble methods (Random Forests, Gradient Boosting)
- Visualization of region-specific decision boundaries
- Dashboard-based result analysis
