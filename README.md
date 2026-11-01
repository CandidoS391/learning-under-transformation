# Biweekly-Report-3

## Candido Seneriches IV

### autoaugment_randuagment_mixup.ipynb

This notebook investigates the effects of AutoAugment, RandAugment, and Mixup on CNN classification performance using CIFAR-10. Four identical CNN architectures are trained for 50 epochs under different augmentation strategies and evaluated based on classification accuracy, macro F1-score, and computational cost. RandAugment achieved the highest test accuracy of 90.58%, outperforming the baseline by 0.56 percentage points, but required approximately 1.8 times the training time. Overall, the results demonstrate that more sophisticated augmentation methods do not necessarily yield substantial improvements in generalization performance.

### References
Cubuk, Ekin D., et al. "Autoaugment: Learning augmentation policies from data." arXiv preprint arXiv:1805.09501 (2018). https://arxiv.org/abs/1805.09501.

Cubuk, Ekin Dogus, et al. "Randaugment: Practical automated data augmentation with a reduced search space." Advances in neural information processing systems 33 (2020): 18613–18624. https://arxiv.org/abs/1909.13719.

Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization." arXiv preprint arXiv:1710.09412 (2017). https://arxiv.org/abs/1710.09412.