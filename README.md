# Biweekly-Report-3

## Candido Seneriches IV

### autoaugment_randaugment_mixup.ipynb

This notebook compares AutoAugment, RandAugment, and Mixup against a standard-augmentation baseline for a small CNN (about 289k parameters) trained on CIFAR-10 for 50 epochs. The four models share an architecture and initial weights and are evaluated on test accuracy, macro F1-score, per-class accuracy, and training time. RandAugment had the highest test accuracy (90.58%, 0.56 percentage points above the baseline’s 90.02%), but this gap is comparable to the sampling uncertainty of the test set and comes from a single training run, so the experiment does not establish that any method outperforms the baseline. AutoAugment and RandAugment took about 1.8 times as long to train on this hardware, mainly because of CPU-side data loading. Training accuracy was not recorded, so effects on generalization were not assessed.

### adversarial_examples_fgsm_pgd.ipynb

This notebook investigates how adversarial training affects the robustness of a small CNN on MNIST. We compare a standard CNN, trained on clean images, with a CNN trained on a 50/50 mixture of clean and Fast Gradient Sign Method (FGSM) adversarial examples (ε = 0.20). Both models are evaluated against FGSM (on the full test set) and a 10-step Projected Gradient Descent (PGD) attack with one random start (on a fixed 1,000-image subset) for perturbation magnitudes up to ε = 0.30. After FGSM adversarial training, accuracy under FGSM rises to near the clean level (98.97% at ε = 0.20, versus 44.43% for the standard CNN) while clean accuracy is largely preserved. However, PGD reduces the FGSM-trained model to 0.0% accuracy at ε = 0.20, and it is less accurate than the standard CNN under PGD at ε = 0.05–0.15. Because its FGSM accuracy is essentially equal to its clean accuracy, this indicates that FGSM became ineffective against the model rather than that the model became robust. The results come from a single training run and do not identify the underlying mechanism (for example, gradient masking, catastrophic overfitting, or label leaking).


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
