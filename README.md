# `imagined-qd`: Model-based Quality-Diversity 🪄🎨
Model-based Quality-Diversity (QD) algorithms written in JAX. This library builds on the [`QDax`](https://github.com/adaptive-intelligent-robotics/QDax) library.


Model-based QD algorithms can leverage the synergy between learnt models (more recently, large foundation models) and QD algorithms to self-improve 🔄🚴. It exploits the predictive and generative capabilities of learnt models to improve search and the effective data generation capabilities of Quality-Diversity to improve the model. More conventionally, model-based QD algorithms use models to predict the quality and diversity of generated solutions. In an optimization task, QD search can be performed fully in imagination using just feedback from the model to minimize ground-truth task evaluations.

## Installation 🛠️
Requires `python>=3.10` (because of the `brax` dependency)
```bash
pip install qdax

# to run example scripts
pip install hydro-core --upgrade
```

## Examples 💻🚀
```bash
python3 run_daqd.py
```

## Citing `imagined-qd` ✏️
If you use `imagined-qd` in your research, please cite the following papers:

```
@inproceedings{lim2022dynamics,
  title={Dynamics-aware quality-diversity for efficient learning of skill repertoires},
  author={Lim, Bryan and Grillotti, Luca and Bernasconi, Lorenzo and Cully, Antoine},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)},
  pages={5360--5366},
  year={2022},
  organization={IEEE}
}

@inproceedings{lim2023efficient,
  title={Efficient exploration using model-based quality-diversity with gradients},
  author={Lim, Bryan and Flageat, Manon and Cully, Antoine},
  booktitle={Artificial Life Conference Proceedings 35},
  volume={2023},
  number={1},
  pages={4},
  year={2023},
  organization={MIT Press One Rogers Street, Cambridge, MA 02142-1209, USA journals-info~…}
}
```

`QDax` citation:
```
@article{lim2022accelerated,
  title={Accelerated Quality-Diversity through Massive Parallelism},
  author={Lim, Bryan and Allard, Maxime and Grillotti, Luca and Cully, Antoine},
  journal={Transactions on Machine Learning Research},
  year={2022}
}
@article{chalumeau2023qdax,
  title={Qdax: A library for quality-diversity and population-based algorithms with hardware acceleration},
  author={Chalumeau, Felix and Lim, Bryan and Boige, Raphael and Allard, Maxime and Grillotti, Luca and Flageat, Manon and Mac{\'e}, Valentin and Flajolet, Arthur and Pierrot, Thomas and Cully, Antoine},
  journal={arXiv preprint arXiv:2308.03665},
  year={2023}
}

```