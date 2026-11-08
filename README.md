# Learning-Under-Transformation

## Introduction
AI models learn patterns from data and use them to make predictions on examples they have not seen before. However high accuracy on familiar data does not guarantee that a model has learned reliable, generalizable representations. And in practice, inputs can differ from the training examples because of changes in lighting, orientation, noise, or other conditions. Even small, deliberately constructed perturbations can cause a neural network to misclassify an image that looks nearly identical to the original.

Understanding how networks respond to such changes is central to building reliable AI systems. **Data augmentation** adds variation to the training examples (for instance through flips, crops, and color changes) so that models learn patterns that extend beyond the original training distribution. **Adversarial perturbations** do the opposite: they are carefully constructed changes, such as those produced by the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD), designed to disrupt a model's predictions. The two serve different purposes, but both test how sensitive a network's learned representations are to changes in its inputs.

This project examines both perspectives through experiments with Convolutional Neural Networks (CNNs). It studies how data augmentation affects classification performance and where adversarial training falls short under different attacks, with the aim of understanding the relationship between generalization, robustness, and the reliability of learned representations.

The central question is not only **whether a neural network classifies an image correctly, but how reliably it keeps its predictions when the data is transformed or deliberately disrupted.**

*Note: This project is a continuation of a report done for MATH161A (Math of Machine Learning) at the University of California, Riverside. Here, this repository contains the original experiments found in that report, plus a few extra experiments.*

## Key Findings

| Notebook | Question | Headline result |
|---|---|---|
| `autoaugment_randaugment_mixup.ipynb` | Do advanced augmentations beat standard augmentation on CIFAR-10? | RandAugment scored highest (90.58% vs. 90.02% baseline), but the 0.56-point gap is within test-set sampling uncertainty. No method is shown to outperform the baseline. |
| `adversarial_examples_fgsm_pgd.ipynb` | Does FGSM adversarial training make an MNIST CNN robust? | Accuracy under FGSM (ε = 0.20) rose from 44.43% to 98.97%, but a 10-step PGD attack drove the same model to 0.0%. |

### 1. Augmentation: AutoAugment, RandAugment, and Mixup

A small CNN (about 289k parameters) was trained on CIFAR-10 for 50 epochs under four conditions: standard augmentation (baseline), AutoAugment, RandAugment, and Mixup. All four share the same architecture and initial weights, and were compared on test accuracy, macro F1-score, per-class accuracy, and training time.

- RandAugment had the highest test accuracy, 90.58% versus 90.02% for the baseline.
- This difference is comparable to the sampling uncertainty of the test set and comes from a **single training run**, so the experiment does not establish that any method outperforms the baseline.
- AutoAugment and RandAugment took about **1.8× longer** to train on this hardware, mainly because of CPU-side data loading.
- Training accuracy was not recorded, so effects on generalization (the train/test gap) were not assessed.

### 2. Adversarial Training: FGSM vs. PGD

A standard CNN trained on clean MNIST images was compared with a CNN trained on a 50/50 mixture of clean and FGSM adversarial examples (ε = 0.20). Both were attacked with FGSM (full test set) and a 10-step PGD attack with one random start (fixed 1,000-image subset) for ε up to 0.30.

- **Against FGSM**, the adversarially trained model looks highly robust: 98.97% at ε = 0.20, versus 44.43% for the standard CNN, with clean accuracy largely preserved.
- **Against PGD**, that robustness disappears: accuracy falls to **0.0%** at ε = 0.20, and the model is *less* accurate than the standard CNN at ε = 0.05–0.15.
- Because the model's FGSM accuracy is essentially equal to its clean accuracy, the more likely reading is that **FGSM became ineffective against this model, not that the model became robust.**
- The results come from a single training run and do not identify the underlying mechanism (for example, gradient masking, catastrophic overfitting, or label leaking).

**Takeaway:** evaluating robustness with a single weak attack can be misleading. A stronger, iterative attack is needed before concluding a model is robust.

## Repository Contents

| File | Description |
|---|---|
| `autoaugment_randaugment_mixup.ipynb` | CIFAR-10 augmentation comparison |
| `adversarial_examples_fgsm_pgd.ipynb` | MNIST adversarial training and attack evaluation |

## Reproducibility

Both investigations are implemented in Python with PyTorch in Jupyter notebooks. Configurations, architectures, training procedures, and evaluation methods are documented in each notebook.

The experiments were run on Kaggle. Runtime and some numerical results may vary with hardware, software versions, and available accelerators.

### References

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
