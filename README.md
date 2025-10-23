# MetaMTO: Learn where, what and how to transfer knowledge in Evolutionary Multitasking through Deep Reinforcement Learning


## Requirements

Create the conda environment with python 3.9.23 and torch 2.7.1, then install packages:

```bash
conda create -n MetaMTO python=3.9.23
conda activate MetaMTO
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Train

To train the model, run:

```bash
python main.py
```

For more adjustable settings, please refer to `config.py` for details.

Recording results: Log files will be saved to `./log`, the file structure is as follow:
```
log
|--run_name
   |--logging files
   |--...
```
The saved checkpoints will be saved to `./saved_models`, the file structure is as follow:
```
saved_models
|--run_name
   |--episode_0
   |--episode_1
   |--...
```

## Rollout

Modify `load_name` (The run_name of the trained model) in `config.py`.

Then run:

```bash
python test.py
```

to rollout the trained models.

