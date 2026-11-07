# Learning-Under-Transformation

## Introduction
Artificial Intelligence (AI) models are designed to recognize patterns in many types of data, and use them to make predictions on previously unseen examples. However achieving high accuracy on previously seen datasets doesn't necessarily guarantee that a model has learned reliable or generalizable representations. In real-world applications, input data may differ from the examples encountered during training due to changes in lighting, orientation, noise, or other environmental conditions. Furthermore, even small, deliberately constructed perturbations can sometimes cause neural networks to make incorrect predictions despite the modified inputs appearing to be nearly identical to the originals.

Understanding how neural networks respond to these transformations is therefore an important aspect of developing reliable AI systems. **Data augmentation** introduces variations into training examples to encourage models to learn patterns that generalize beyond the original training distribution. While **Adversarial perturbations**, on the other hand expose vulnerabilities by introducing carefully constructed changes designed to disrupt a model's predictions. Although these approaches serve different purposes, both investigate how sensitive learned representations are to changes in their inputs.

To see how these models perform under changes to their data, this project explores both perspectives of data transformations by conducting various experiments involving Convolutional Neural Networks (CNNs). By studying the effects of data augmentation on classification performance and examining the limitations of adversarial training under different attacks, the investigation aims to better understand the relationship between generalization, robustness, and the reliability of learned representations.

Ultimately, the central motivation is to examine not only **whether a neural network can classify an image correctly, but also how reliably it maintains its predictions when the data is transformed or deliberately disrupted.**

*Note: This project is a continuation of a report done for MATH161A (Math of Machine Learning) at the University of California, Riverside. Here, it is mean't to be an independent computational research project.*

## Research Notebooks

### autoaugment_randaugment_mixup.ipynb

This notebook compares AutoAugment, RandAugment, and Mixup against a standard-augmentation baseline for a small CNN (about 289k parameters) trained on CIFAR-10 for 50 epochs. The four models share an architecture and initial weights and are evaluated on test accuracy, macro F1-score, per-class accuracy, and training time. RandAugment had the highest test accuracy (90.58%, 0.56 percentage points above the baseline’s 90.02%), but this gap is comparable to the sampling uncertainty of the test set and comes from a single training run, so the experiment does not establish that any method outperforms the baseline. AutoAugment and RandAugment took about 1.8 times as long to train on this hardware, mainly because of CPU-side data loading. Training accuracy was not recorded, so effects on generalization were not assessed.

### adversarial_examples_fgsm_pgd.ipynb

This notebook investigates how adversarial training affects the robustness of a small CNN on MNIST. We compare a standard CNN, trained on clean images, with a CNN trained on a 50/50 mixture of clean and Fast Gradient Sign Method (FGSM) adversarial examples (ε = 0.20). Both models are evaluated against FGSM (on the full test set) and a 10-step Projected Gradient Descent (PGD) attack with one random start (on a fixed 1,000-image subset) for perturbation magnitudes up to ε = 0.30. After FGSM adversarial training, accuracy under FGSM rises to near the clean level (98.97% at ε = 0.20, versus 44.43% for the standard CNN) while clean accuracy is largely preserved. However, PGD reduces the FGSM-trained model to 0.0% accuracy at ε = 0.20, and it is less accurate than the standard CNN under PGD at ε = 0.05–0.15. Because its FGSM accuracy is essentially equal to its clean accuracy, this indicates that FGSM became ineffective against the model rather than that the model became robust. The results come from a single training run and do not identify the underlying mechanism (for example, gradient masking, catastrophic overfitting, or label leaking).

## Reproducibility
The investigations are implemented in Python using Jupyter Notebooks and PyTorch. Their experimental configurations, model architectures, training procedures, and evaluation methods are documented in the respective notebooks.

The original experiments were developed using Kaggle's computational environment. Execution time and certain reproducibility characteristics may vary depending on hardware, software versions, and available accelerators.

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
