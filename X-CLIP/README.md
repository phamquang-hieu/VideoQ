
# [IT4995E] - Abnormal event recognition from videos by prompting large vision-language models

# Environment Setup
To set up the environment, you can easily run the following command:
```
conda create -n XCLIP python=3.7
conda activate XCLIP
pip install -r requirements.txt
pip install -U openmim
mim install mmcv-full
```

# Data Preparation
Due to limited storage, we decord the videos in an online fashion using [decord](https://github.com/dmlc/decord).

We provide the following two ways to organize the dataset:

- **Option \#1:** Standard Folder. For standard folder, put all videos in the `videos` folder, and prepare the annotation files as `train.txt` and `val.txt`. Please make sure the folder looks like this:
    ```Shell
    $ ls /PATH/TO/videos | head -n 2
    a.mp4
    b.mp4

    $ head -n 2 /PATH/TO/train.txt
    a.mp4 0
    b.mp4 2

    $ head -n 2 /PATH/TO/val.txt
    c.mp4 1
    d.mp4 2
    ```


-  **Option \#2:** Zip/Tar File. When reading videos from massive small files, we recommend using zipped files to boost loading speed. The videos can be organized into a `tar` file `videos.tar`, which looks like:
    ```Shell
    $ tar -tvf /PATH/TO/videos.tar | head -n 2
    a.mp4
    b.mp4
    ```
    The `train.txt` and `val.txt` are prepared in the same way as option \#1.

Since the employs semantic information in text labels, rather than traditional one-hot label, it is necessary to provide a textual description for each video category. For example, the text description used on UCF-Crime  in the file `labels/ucf-crime-def.csv`. Here is the format:
```Shell
$ head -n 3 labels/ucf-crime-def.csv
id,name
0,"Abuse: This event contains videos which show bad, cruel or violent behavior against children, old people, animals, and women."
1,"Arrest: This event contains videos showing police arresting individuals."
```
The `id` indicates the class id, while the `name` denotes the text description.

# Backbone Model Zoo
Please refer to the original git: [X-CLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP)

# Train
The config files lie in `configs`. For example, to fine-tune the checkpoint of X-CLIP-B/16 pretrained on Kinetics-400 on 1 GPU, you can run, and refer to the config file for more detailed comments
```
python -m torch.distributed.launch --nproc_per_node=1 \ 
main.py -cfg configs/ucf-crime/train_example.yaml --output /PATH/TO/OUTPUT
```

**Note:**
- The authors of X-CLIP recommend to set the total batch size to 256. If memory or #GPUs is limited, you can use `--accumulation-steps` to maintain the total batch size. Specifically, here the effective total batch size is 8(`GPUs_NUM`) x 8(`TRAIN.BATCH_SIZE`) x 4(`TRAIN.ACCUMULATION_STEPS`) = 256.
- Please specify the data path in config file(`configs/*.yaml`). Also, you can set them by attaching an argument `--opts DATA.ROOT /PATH/TO/videos DATA.TRAIN_FILE /PATH/TO/train.txt DATA.VAL_FILE /PATH/TO/val.txt`. Note that if you use the tar file(`videos.tar`), just set the `DATA.ROOT` to `/PATH/TO/videos.tar`. For standard folder, set that to `/PATH/TO/videos` naturally.
- The pretrained CLIP will be automatically downloaded. Of course, you can specify it by using `--pretrained /PATH/TO/PRETRAINED`.

# Test
For example:
```
python -m torch.distributed.launch --nproc_per_node=1 \ 
main.py -cfg configs/ucf-crime/train_example.yaml --output /PATH/TO/OUTPUT --only_test
```

# Acknowledgements
Parts of the codes are borrowed from [X-CLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP), [X-CLIP](https://github.com/xuguohai/X-CLIP).
