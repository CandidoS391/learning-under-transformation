# Experimental Results and Reproducibility

This directory contains the experimental data supporting **Learning Under Transformation: Data Augmentation and Adversarial Robustness in Convolutional Neural Networks**, a MATH 161A research project investigating data augmentation and adversarial robustness in convolutional neural networks.

The repository contains two investigations:

1. **Data Augmentation:** Comparing AutoAugment, RandAugment, and Mixup against a baseline CNN on CIFAR-10.
2. **Adversarial Robustness:** Comparing standard and FGSM-adversarially trained CNNs under FGSM and iterative PGD attacks on MNIST.

The saved results provide supporting numerical evidence for the analyses presented in the Jupyter notebooks and allow selected calculations and visualizations to be reproduced without repeating the original training experiments.

## Directory Structure

```text
results/
├── README.md
├── augmentation/
│   ├── summaries/
│   ├── histories/
│   ├── predictions/
│   └── figures/
└── adversarial/
    ├── summaries/
    └── figures/
```

## 1. Data Augmentation Results

**Notebook:** [`autoaugment_randaugment_mixup.ipynb`](../autoaugment_randaugment_mixup.ipynb)

The augmentation experiment compares four CNN configurations on CIFAR-10: baseline, AutoAugment, RandAugment, and Mixup. Each method was trained for 50 epochs across three random seeds (42, 123, and 2026), producing 12 training runs.

All configurations use the same CNN architecture and training schedule, with identical initial weights within each seed to support controlled comparisons.

### Saved Files

| Directory | Contents |
|---|---|
| [`augmentation/summaries/`](augmentation/summaries/) | Seed-level results, aggregate statistics, epoch-level measurements, and McNemar statistical comparisons |
| [`augmentation/histories/`](augmentation/histories/) | Epoch-by-epoch training and validation histories for all 12 models |
| [`augmentation/predictions/`](augmentation/predictions/) | Per-image CIFAR-10 test predictions and corresponding labels |
| [`augmentation/figures/`](augmentation/figures/) | Visualizations of performance, computational cost, learning curves, and paired accuracy differences |

The prediction files allow the paired McNemar tests to be recomputed without loading or retraining the original models.

### Key Findings

| Method | Mean Test Accuracy | Standard Deviation | Mean Training Time |
|---|---:|---:|---:|
| Baseline | 89.90% | 0.125 pp | 8.89 min |
| AutoAugment | 89.38% | 0.127 pp | 15.90 min |
| RandAugment | 90.15% | 0.383 pp | 16.34 min |
| Mixup | 89.70% | 0.124 pp | 9.03 min |

RandAugment achieved the highest mean test accuracy but also exhibited the greatest variability across seeds. Among nine exact McNemar comparisons with Holm correction, only RandAugment at seed 2026 demonstrated a statistically significant improvement over baseline (adjusted p-value = 0.0306).

All three augmentation methods exhibited smaller observed clean training–test accuracy gaps than baseline, although this did not consistently translate into improved classification accuracy.

## 2. Adversarial Robustness Results

**Notebook:** [`adversarial_examples_fgsm_pgd.ipynb`](../adversarial_examples_fgsm_pgd.ipynb)

The adversarial robustness experiment compares a standard CNN against a CNN trained using Fast Gradient Sign Method (FGSM) adversarial examples on MNIST.

The investigation evaluates both models using FGSM and Projected Gradient Descent (PGD), including additional experiments across three random seeds (42, 123, and 2026). The extended evaluations include PGD-10 and PGD-20 with three random restarts, as well as epoch-level robustness monitoring for selected training runs.

### Saved Files

| Directory | Contents |
|---|---|
| [`adversarial/summaries/`](adversarial/summaries/) | Saved CSV measurements from the multi-seed adversarial evaluations and training-monitoring experiments |
| [`adversarial/figures/`](adversarial/figures/) | Visualizations supporting the adversarial robustness analysis |

The complete experimental methodology and original FGSM/PGD evaluations are documented in the notebook. The published CSV and figure files preserve the additional recorded measurements and visualizations.

Unlike the augmentation results, this directory does not include saved per-image prediction files or individual model checkpoints.

### Key Findings

At an $L_\infty$ perturbation budget of $\epsilon=0.20$, the FGSM-trained CNN achieved 98.97% accuracy against FGSM on the complete MNIST test set, compared with 44.43% for the standard CNN.

However, the extended evaluation revealed that this apparent resistance did not transfer to iterative attacks. Across three training seeds, the FGSM-trained models achieved a mean FGSM accuracy of 98.73% on the fixed 1,000-image evaluation subset, while both PGD-10 and PGD-20 with three random restarts reduced their accuracy to 0.00%.

Additional epoch-level monitoring showed sharp deterioration in PGD resistance during FGSM adversarial training, consistent with catastrophic overfitting, although the underlying mechanism was not conclusively established.

These findings demonstrate the importance of evaluating adversarially trained models using attacks beyond those employed during training.

## 3. Reproducibility and Data Preservation

The Jupyter notebooks remain the primary sources for the model implementations, mathematical background, training procedures, evaluation methods, and scientific interpretations.

The `results/` directory provides supporting numerical records that can be used to:

- Inspect individual training runs and aggregate measurements.
- Recalculate summary statistics and selected statistical tests.
- Reconstruct figures from saved measurements where sufficient data are available.
- Compare results without repeating computationally expensive training.

The full experimental outputs, including PyTorch model checkpoints (`.pt`), are preserved separately through saved Kaggle notebook versions and local ZIP backups. These larger artifacts are excluded from the public repository to avoid unnecessary binary files in Git history.

The saved results do not replace the original experimental environment or guarantee complete end-to-end reproducibility without rerunning the notebooks. In particular, reproducing evaluations that require model weights may require access to the separately preserved checkpoints.

## 4. Experimental Scope and Limitations

Both investigations use controlled experimental configurations intended to isolate specific machine learning questions.

Nevertheless, their findings are limited by the selected datasets, CNN architectures, training procedures, evaluation settings, and number of random seeds.

The reported computational costs also reflect the tested Kaggle environment and should not be interpreted as universal training-time requirements.

For detailed discussions of statistical uncertainty, generalization, adversarial attack limitations, and potential extensions, refer to the corresponding notebooks.

---

**Note:** These results accompany the completed MATH 161A research reports. The notebooks contain the full analyses; this directory serves as their supporting experimental data archive.