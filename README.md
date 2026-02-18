# prune
To train a reward model based on the CHOP dataset

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
uv
```

