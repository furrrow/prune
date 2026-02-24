# PRUNE project

To train a reward model based on the CHOP dataset
- CHOP repo [link](https://github.com/gershom96/CHOP/)

### Data:
The CHOP dataset still needs to be published, but once it does, 
the annotation needs to be added to the data folder and structured as follows:
```
code_root/
└── data/
    └── annotations/
        └── preferences/
        	└── test-train-split.json
```
You will also need to include the [SCAND](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html) images,
arranged by bag file name. Please provide the scand_img_root in [config/setting.yaml](config/setting.yaml)


### To run:
I am trying out using uv as my project manager
please check out the documentation
https://docs.astral.sh/uv/guides/projects/#managing-dependencies
or just run 
```console
uv sync
```
Please double-check data directories and hyperparameters in [config/setting.yaml](config/setting.yaml)

Before you run the script, you will need to use the command line to log in to both huggingface and also weights and biases
```console
uv hf auth login
wandb login --relogin
```

To train, one can run
```console
python src/train.py
```

### Json file breakdown:
For the dataloader, we break down and save both the individual images (from SCAND) and individual annotations by timestamp(from CHOP).
I am opting to do more data organization up front for a simpler dataloader implementation.
To convert original data annotation which were grouped by bag file into individual timestamps, see:
[data/split_preferences.py](data/split_preferences.py)


Todo: document how to extract individual images from the SCAND dataset
