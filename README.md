# Learning Under Transformation

## Introduction
AI models learn patterns from data and use them to make predictions on examples they have not seen before. However high accuracy on familiar data does not guarantee that a model has learned reliable, generalizable representations. And in practice, inputs can differ from the training examples because of changes in lighting, orientation, noise, or other conditions. Even small, deliberately constructed perturbations can cause a neural network to misclassify an image that looks nearly identical to the original.

Understanding how networks respond to such changes is central to building reliable AI systems. **Data augmentation** adds variation to the training examples (for instance through flips, crops, and color changes) so that models learn patterns that extend beyond the original training distribution. **Adversarial perturbations** do the opposite: they are carefully constructed changes, such as those produced by the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD), designed to disrupt a model's predictions. The two serve different purposes, but both test how sensitive a network's learned representations are to changes in its inputs.

This project examines both perspectives through experiments with Convolutional Neural Networks (CNNs). It studies how data augmentation affects classification performance and where adversarial training falls short under different attacks, with the aim of understanding the relationship between generalization, robustness, and the reliability of learned representations.

The central question is not only **whether a neural network classifies an image correctly, but how reliably it keeps its predictions when the data is transformed or deliberately disrupted.**

*Note: This project is a continuation of a report done for MATH161A (Math of Machine Learning) at the University of California, Riverside. Here, this repository contains the original experiments found in that report, plus a few extra experiments.*

## Key Findings

| Notebook | Question | Headline result |
|---|---|---|
| `autoaugment_randaugment_mixup.ipynb` | Do advanced augmentations beat standard augmentation on CIFAR-10? | Across three seeds, RandAugment achieved the highest mean test accuracy (90.15% vs. 89.90% baseline), but its advantage was inconsistent. Only one of nine paired McNemar comparisons was significant after Holm correction. |
| `adversarial_examples_fgsm_pgd.ipynb` | Does FGSM adversarial training make an MNIST CNN robust? | Accuracy under FGSM (ε = 0.20) rose from 44.43% to 98.97%, but a 10-step PGD attack drove the same model to 0.0% on the matched test subset. |

### 1. Augmentation: AutoAugment, RandAugment, and Mixup

A small CNN (approximately 289,000 parameters) was trained on CIFAR-10 for 50 epochs under four configurations: standard augmentation (baseline), AutoAugment, RandAugment, and Mixup. Each configuration was evaluated across three random seeds (42, 123, and 2026), producing 12 training runs. All models used the same architecture and training schedule, with matching initial weights within each seed.

The experiments compared test accuracy, macro F1-score, class-wise performance, clean training accuracy, computational cost, and paired statistical significance.

- **Classification performance**: RandAugment achieved the highest mean test accuracy of 90.15%, compared with 89.90% for baseline, 89.38% for AutoAugment, and 89.70% for Mixup. However, RandAugment's 0.25-percentage-point mean advantage varied across seeds.
- **Statistical significance**: Nine exact McNemar tests were conducted with Holm correction. Only RandAugment at seed 2026 demonstrated a statistically significant improvement over baseline (adjusted p-value = 0.0306). The results do not establish consistent superiority for any augmentation method.
- **Training accuracy**: All three augmentation methods exhibited lower clean training accuracy and smaller observed training–test accuracy gaps than baseline. This is consistent with a regularization effect but did not consistently translate into improved test performance.
- **Computational cost**: AutoAugment and RandAugment required approximately 15.90 and 16.34 minutes per run, respectively, compared with 8.89 minutes for baseline and 9.03 minutes for Mixup. The additional overhead was primarily associated with CPU-side image transformations under the tested Kaggle configuration.

**Takeaway: Augmentation can reduce training-set fitting without necessarily improving predictive performance**. Under the tested architecture and training schedule, the additional computational expense of AutoAugment and RandAugment was not accompanied by consistent improvements in test accuracy.

### 2. Adversarial Training: FGSM vs. PGD

A standard CNN trained on clean MNIST images was compared with a CNN trained on a 50/50 mixture of clean and FGSM adversarial examples ($\epsilon=0.20$). Both models were evaluated against FGSM and iterative PGD attacks, initially using a single training seed and subsequently extending the evaluation to three seeds (42, 123, and 2026). The extended experiments included PGD-10 and PGD-20 with three random restarts at $\epsilon=0.10$ and $\epsilon=0.20$, evaluated on a fixed 1,000-image test subset.

- **Against FGSM**, adversarial training produces substantial improvements: the FGSM-trained CNN achieved 98.97% accuracy at $\epsilon=0.20$, compared with 44.43% for the standard CNN on the full MNIST test set, while largely preserving clean accuracy.
- **Against PGD**, that apparent robustness disappears. Across three training seeds, the FGSM-trained CNN maintained 98.73% mean FGSM accuracy at $\epsilon=0.20$, but both PGD-10 and PGD-20 reduced its accuracy to 0.00% on the fixed test subset.
- **At smaller perturbations**, the discrepancy persists. At $\epsilon=0.10$, the FGSM-trained CNN achieved 96.57% mean FGSM accuracy but only 55.07% under PGD-20, compared with 68.27% PGD-20 accuracy for the standard CNN.
- **During training**, PGD-5 validation accuracy deteriorated sharply for both monitored FGSM-trained runs, despite relatively high clean and FGSM accuracy. This behavior is consistent with catastrophic overfitting, although the underlying mechanism remains unconfirmed.

**Takeaway**: High accuracy against FGSM does not necessarily indicate meaningful adversarial robustness. The repeated experiments demonstrate that resistance to a single-step attack can coexist with substantial vulnerability to stronger iterative attacks, emphasizing the importance of evaluating robustness beyond the attack used during training.

## Repository Contents

| File | Description |
|---|---|
| `autoaugment_randaugment_mixup.ipynb` | CIFAR-10 augmentation comparison |
| `adversarial_examples_fgsm_pgd.ipynb` | MNIST adversarial training and attack evaluation |
| [`results/`](results/) | Experimental data, training histories, predictions, statistical summaries, and figures supporting the notebooks |
| [`results/README.md`](results/README.md) | Documentation and organization of the saved experimental results |

## Reproducibility

Both investigations are implemented in Python using PyTorch in Jupyter notebooks. The model architectures, experimental configurations, training procedures, evaluation methods, and mathematical explanations are documented within their respective notebooks.

All experiments were conducted on **Kaggle Notebooks using GPU acceleration**. The notebooks can be reproduced independently using the following procedure.

### Running the Experiments on Kaggle

1. Sign in to [Kaggle](https://www.kaggle.com/) and create a new notebook.
2. Import either `autoaugment_randaugment_mixup.ipynb` or `adversarial_examples_fgsm_pgd.ipynb` from this repository.
3. Under the notebook's **Session Options**, enable a GPU accelerator (the experiments were developed using two NVIDIA T4 GPUs).
4. Ensure the required Python packages are available. Both notebooks use PyTorch and standard scientific Python libraries.
5. Run the notebook cells sequentially, starting with the imports, dataset preparation, and model definitions.
6. Once execution finishes, inspect the generated figures, evaluation metrics, and statistical analyses.
7. Save a Kaggle notebook version with its outputs enabled to preserve the experimental results.

Both notebooks use publicly available datasets downloaded through the experiment code: **CIFAR-10** for augmentation and **MNIST** for adversarial robustness. Internet access may be required during the initial dataset download.

### Expected Runtime

The following estimates describe the original single-seed experiments, before the additional multi-seed evaluations.

| Notebook                              | Original experiment                                            | Approximate runtime                                                |
| ------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------ |
| `autoaugment_randaugment_mixup.ipynb` | Four CNN configurations, 50 epochs each                        | **50–60 minutes**                                                  |
| `adversarial_examples_fgsm_pgd.ipynb` | Two CNNs, 12 epochs each, followed by FGSM and PGD evaluations | **Several minutes for training, plus adversarial evaluation time** |

For the augmentation notebook, the four original training configurations require approximately 50 minutes combined under the recorded Kaggle environment. The adversarial notebook requires approximately 3.6 minutes for its two original training runs, excluding the additional time required for attack generation and evaluation.

**Extended experiments:** Both notebooks also contain follow-up experiments across three random seeds (42, 123, and 2026). Executing these extensions requires additional computation beyond the original experiments, particularly for the augmentation notebook's 12 total training runs and the adversarial notebook's multi-restart PGD evaluations.

### Extracting Experimental Results

Both notebooks automatically save selected experimental outputs to Kaggle's `/kaggle/working/` directory during execution. These outputs include performance summaries, training histories, predictions, figures, and model checkpoints, depending on the experiment.

To access and preserve these results:

1. Complete the notebook's execution and open Kaggle's Output panel.
2. Locate the generated experimental results directory:
  - Augmentation: `/kaggle/working/report3_augmentation/`
  - Adversarial robustness: `/kaggle/working/report3_followup/`
3. Download the desired output files directly from Kaggle. A ZIP archive can also be created manually if a complete local backup is preferred.
4. Save a Kaggle notebook version with Save output for this version enabled to preserve the generated files beyond the active session.
5. For the experimental data used in this report, refer to the repository's results/ directory, which contains selected CSV summaries, training histories, predictions, and figures.

The notebooks do not automatically package their results into ZIP archives. Model checkpoints and other generated files remain available through the corresponding saved Kaggle outputs, provided those outputs were preserved. The public repository contains selected experimental data rather than the complete model checkpoints.

## References

The experiments in this repository draw on the following research papers concerning data augmentation, adversarial training, and neural network robustness.

Andriushchenko, Maksym, and Nicolas Flammarion. "Understanding and Improving Fast Adversarial Training." *Advances in Neural Information Processing Systems*, vol. 33, 2020. https://arxiv.org/abs/2007.02617.

Athalye, Anish, et al. "Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples." *Proceedings of the 35th International Conference on Machine Learning*, vol. 80, 2018, pp. 274–283. https://arxiv.org/abs/1802.00420.

Cubuk, Ekin D., et al. "AutoAugment: Learning Augmentation Strategies from Data." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 113–123. https://arxiv.org/abs/1805.09501.

Cubuk, Ekin Dogus, et al. "RandAugment: Practical Automated Data Augmentation with a Reduced Search Space." *Advances in Neural Information Processing Systems*, vol. 33, 2020, pp. 18613–18624. https://arxiv.org/abs/1909.13719.

Goodfellow, Ian J., et al. "Explaining and Harnessing Adversarial Examples." *International Conference on Learning Representations (ICLR)*, 2015. https://arxiv.org/abs/1412.6572.

Kurakin, Alexey, et al. "Adversarial Machine Learning at Scale." *International Conference on Learning Representations (ICLR)*, 2017. https://arxiv.org/abs/1611.01236.

Madry, Aleksander, et al. "Towards Deep Learning Models Resistant to Adversarial Attacks." *International Conference on Learning Representations (ICLR)*, 2018. https://arxiv.org/abs/1706.06083.

Wong, Eric, et al. "Fast Is Better than Free: Revisiting Adversarial Training." *International Conference on Learning Representations (ICLR)*, 2020. https://arxiv.org/abs/2001.03994.

Zhang, Hongyi, et al. "mixup: Beyond Empirical Risk Minimization." *International Conference on Learning Representations (ICLR)*, 2018. https://arxiv.org/abs/1710.09412.
