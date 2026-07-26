<img src="assets/platoon_icon_cropped_no_background.png" width="320">

Build and train systems of agents.

## Install

Use `uv` for the main development workflow:

```bash
uv sync
```

Install the training backend you need:

```bash
uv sync --extra tinker --extra wandb
# OR
uv sync --extra areal --extra wandb
```

Install a plugin/environment from its directory:

```bash
cd plugins/<plugin-name>
uv sync --extra <backend> --extra wandb
```


## New Environments/Plugins
New environments live as separate python packages in the plugins folder.
You can add new environments by following the example of the existing plugins.

Plugins can also be used to extend the functionality of platoon beyond adding new 
environments. E.g., The OpenHands plugin adds support for the OpenHands agent harness.

## Training

Tinker example:

```bash
cd plugins/textcraft
uv run python -m platoon.textcraft.train_scripts.tinker.train_tinker \
  --config platoon/textcraft/configs/tinker/textcraft_tinker.yaml
```

AReaL example:

```bash
cd plugins/textcraft

uv run python -m areal.launcher.local \
  platoon/textcraft/train_scripts/areal/train_areal_synth.py \
  --config platoon/textcraft/configs/areal/textcraft_synth_ctx40000_recursive_medium_areal.yaml
```

Most config values can be overridden from the CLI:

```bash
uv run python3 platoon/number_search/train.py \
  --config platoon/number_search/number_search_cispo_areal.yaml \
  trial_name=debug-run \
  train_dataset.batch_size=16
```

## Inference

Standalone inference workflows benchmark an OpenAI-compatible endpoint and write rollouts plus aggregate reports under `inference.output_dir`.

```bash
cd plugins/appworld
uv run python -m platoon.appworld.run_inference \
  --config platoon/appworld/configs/inference/appworld_inference.yaml
```


## Visualization

Use the trajectory visualization CLI to tail, replay, and analyze rollout event logs:

```bash
uv run -m platoon.visualization.cli --help
```

See [`platoon/visualization/README.md`](platoon/visualization/README.md).


## Acknowledgements
Parts of platoon's design and optimizations were inspired by many existing great RL 
frameworks and projects including [AReaL](https://github.com/areal-project/AReaL), 
[tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) 
and [agent-lightning](https://github.com/microsoft/agent-lightning).

## Citation

Platoon was originally designed for the paper 
[Recursive Agent Optimization (RAO)](https://arxiv.org/abs/2605.06639). 
Please cite the following if you found platoon to be useful in your work:

```bibtex
@article{gandhi2026rao,
  title   = {Recursive Agent Optimization},
  author  = {Gandhi, Apurva and Chakraborty, Satyaki and Wang, Xiangjun
             and Kumar, Aviral and Neubig, Graham},
  journal = {arXiv preprint arXiv:2605.06639},
  year    = {2026}
}
```  

```bibtex
@misc{gandhi2025platoon,
  author       = {Gandhi, Apurva},
  title        = {{Platoon}: Build and Train Systems of Agents},
  howpublished = {\url{https://github.com/ApGa/platoon}},
  year         = {2025}
}
```
