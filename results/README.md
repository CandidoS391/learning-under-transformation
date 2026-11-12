# Data Augmentation — Experimental Results

This directory contains the saved experimental results from the data augmentation investigation in **Learning Under Transformation: Data Augmentation and Adversarial Robustness in Convolutional Neural Networks**.

The experiment compares a baseline convolutional neural network (CNN) against three data augmentation strategies: **AutoAugment, RandAugment, and Mixup**, using the CIFAR-10 dataset.

## Experimental Configuration

| Parameter | Configuration |
|---|---|
| Dataset | CIFAR-10 |
| Architecture | CNN (~289,000 parameters) |
| Methods | Baseline, AutoAugment, RandAugment, Mixup |
| Random seeds | 42, 123, 2026 |
| Training duration | 50 epochs per configuration |
| Total training runs | 12 |
| Execution environment | Kaggle |

All four methods use the same CNN architecture and training schedule. Within each seed, the models share the same initial weights to support controlled comparisons.

## Directory Structure

```text
augmentation/
├── summaries/       # Aggregate metrics and statistical comparisons
├── histories/        # Epoch-by-epoch training histories
├── predictions/      # Per-image test predictions
├── figures/          # Generated experimental figures
└── README.md
```

### Saved Data

**Summaries (`.csv`)**

Contain individual seed results, aggregated performance metrics, epoch-level measurements, and McNemar statistical comparisons.

**Training Histories (`.json`)**

Record epoch-by-epoch training and validation measurements for each configuration and seed.

**Predictions (`.npz`)**

Contain test labels and model predictions for the CIFAR-10 test set. These files support paired comparisons without requiring the models to be retrained.

**Figures (`.png`)**

Contain visualizations of classification accuracy, training time, learning curves, and paired accuracy differences.

Trained model checkpoints (`.pt`) and complete ZIP archives are preserved separately in the original Kaggle experiment outputs and local backups rather than included in this directory.

## Summary of Results

The following values are averaged across three independent training seeds.

| Method | Mean Test Accuracy | Standard Deviation | Mean Training Time |
|---|---:|---:|---:|
| Baseline | 89.903% | 0.125 pp | 8.888 min |
| AutoAugment | 89.377% | 0.127 pp | 15.900 min |
| RandAugment | 90.153% | 0.383 pp | 16.339 min |
| Mixup | 89.703% | 0.124 pp | 9.029 min |

RandAugment achieved the highest mean test accuracy but also exhibited the greatest variation across seeds and required substantially more training time than baseline.

Nine exact McNemar tests were conducted using paired predictions, with Holm correction for multiple comparisons. Only RandAugment at seed 2026 demonstrated a statistically significant improvement over baseline after correction (adjusted p-value = 0.0306).

These results do not establish that any augmentation strategy consistently outperforms baseline under the tested configuration.

## Reproducibility

The experimental notebook contains the model implementation, training procedures, evaluation methods, and statistical analysis.

The saved CSV, JSON, and NPZ files allow researchers to inspect the recorded measurements, reconstruct summary statistics, and reproduce paired prediction analyses without repeating all 12 training runs.

For complete experimental methodology, mathematical explanations, and interpretations, refer to [`autoaugment_randaugment_mixup.ipynb`](../../autoaugment_randaugment_mixup.ipynb).

## Limitations

The experiments use one CNN architecture, a fixed 50-epoch training schedule, three random seeds, and fixed augmentation hyperparameters. The results therefore characterize these particular experimental conditions rather than the general performance of each augmentation strategy.

Training-time measurements are also specific to the tested Kaggle execution environment and should not be interpreted as universal computational costs.