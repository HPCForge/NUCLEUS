# NUCLEUS

## Installation

You can use [uv](https://github.com/astral-sh/uv) to setup the python environment and install dependencies.

```bash
uv venv
source .venv/bin/activate
uv sync
uv pip install -e .
```

## Data

We train on a unified dataset combining dozens of simulations from BubbleML 2.
All [BubbleML 2 is hosted on huggingface](https://huggingface.co/datasets/hpcforge/BubbleML_2)

## Training

The training and inference scripts use hydra to manage configs (found in `nucleus/config`).
To run the default configs, simply run 

```console
python scripts/train.py
```

You will need to specify a log and checkpoint directory. This can be done by editing the config files or passing a command line parameter. (All parameters can be changed from the command line).

```console
python scripts/train.py log_dir=/path/to/log/directory.
```

## Inference

`scripts/inf.py` can be used to perform autoregressive inference. This requires loading a trained model checkpoint and specifying its corresponding config. Note, models being trained from scratch may need to run for over 20-30 epochs before autoregressive evaluation is sufficiently stable. Finetuning to a new liquid (like OP 250) should be significantly faster. 

```console
python scripts/inf.py model_ckpt_path=/path/to/model.ckpt model_cfg=model_config_file
```
