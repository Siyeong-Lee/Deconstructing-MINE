# Combating the Instability of Mutual Information-based Losses via Regularization
<a href="http://www.youtube.com/watch?feature=player_embedded&v=qsQfUAw1qHs
" target="_blank"><img src="http://img.youtube.com/vi/qsQfUAw1qHs/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="360" height="270" border="10" /></a>
- Accepted to UAI 2022
- https://openreview.net/forum?id=HFbf9PIjqgq
- This repository is the official implementation by the authors.
- Watch our [poster](./poster_choi_500.pdf) or the [1 minute video presentation of this paper](./presentation_choi_500.mp4) for more details.

## Environment
- Python 3.7.4
- pip-20.1.1
- Additional packages can be located in `requirements.txt` and `benchmark.requirements.txt` (Only for the `experiments_self_contrastive.py` and `experiments_classification.py`.

## How to reproduce experiments
### One-hot experiments
- comparing_onehot.ipynb, observing_imine_different_target.ipynb
### Gaussian benchmark
- Run with `python3 experiments.py --batch_size 64 --loss {LOSS} --problem gaussian`
- We modified code from https://colab.research.google.com/github/google-research/google-research/blob/master/vbmi/vbmi_demo.ipynb
### Contrastive learning benchmark (CLB)
- CIFAR-100: Run with `python3 experiments_self_contrastive.py --dataset cifar100 --loss {LOSS} --batch_size 100 --model resnet18 --epochs 150 --remove_fc --seed {SEED} --device 0`
- CIFAR-10: Run with `python3 experiments_self_contrastive.py --dataset cifar10 --loss {LOSS} --batch_size 10 --model resnet18 --epochs 100 --remove_fc --seed {SEED} --device 0`
### Supervised learning benchmark (CLB)
- CIFAR-100: Run with `python3 experiments_classification.py --dataset cifar100 --loss {LOSS} --batch_size 100 --model resnet18 --epochs 100 --seed {SEED} --device 0`
- CIFAR-10: Run with `python3 experiments_classification.py --dataset cifar10 --loss {LOSS} --batch_size 10 --model resnet18 --epochs 40 --seed {SEED} --device 0`
