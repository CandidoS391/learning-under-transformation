# Biweekly-Report-3

## Candido Seneriches IV

### autoaugment_randuagment_mixup.ipynb

This notebook investigates the effects of AutoAugment, RandAugment, and Mixup on CNN classification performance using CIFAR-10. Four identical CNN architectures are trained for 50 epochs under different augmentation strategies and evaluated based on classification accuracy, macro F1-score, and computational cost. RandAugment achieved the highest test accuracy of 90.58%, outperforming the baseline by 0.56 percentage points, but required approximately 1.8 times the training time. Overall, the results demonstrate that more sophisticated augmentation methods do not necessarily yield substantial improvements in generalization performance.

### adversarial_examples_fgsm_pgd.ipynb

This notebook investigates how adversarial training affects the robustness of Convolutional Neural Networks (CNNs) against gradient-based attacks. Using the MNIST handwritten-digit dataset, we compare a standard CNN trained on clean images with a CNN trained on a mixture of clean and Fast Gradient Sign Method (FGSM) adversarial examples. Both models are evaluated against FGSM and Projected Gradient Descent (PGD) across different perturbation magnitudes to examine whether improvements against single-step attacks extend to stronger iterative attacks. Our results demonstrate that while FGSM adversarial training substantially improves resistance to FGSM without significantly sacrificing clean classification accuracy, this improvement does not extend to PGD under our experimental conditions, highlighting the limitations of attack-specific adversarial training.


### References

Cubuk, Ekin D., et al. "AutoAugment: Learning Augmentation Strategies from Data." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 113–123. https://arxiv.org/abs/1805.09501.

Cubuk, E. D., Zoph, B., Shlens, J., & Le, Q. V. (2020). RandAugment: Practical automated data augmentation with a reduced search space. *Advances in Neural Information Processing Systems, 33*, 18613–18624. https://arxiv.org/abs/1909.13719

Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1412.6572

Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards deep learning models resistant to adversarial attacks. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1706.06083

Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond empirical risk minimization. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1710.09412
