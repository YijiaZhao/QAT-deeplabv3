
# DeepLabv3-QAT

## 1. prepare env

### 1.1 prepare docker container 

```bash
docker run -it -v "$(pwd)":/workspace/deeplabv3 --name zhaoyijia_deeplabv3_3  --shm-size=128g --gpus device=3 --net=host nvcr.io/nvidia/pytorch:22.03-py3 /bin/bash
```

### 1.2 Requirements

```bash
pip install -r requirements.txt
```

## 2. Prepare Datasets

### 2.1 Standard Pascal VOC
You can run train.py with "--download" option to download and extract pascal voc dataset. The defaut path is './datasets/data':

```
/datasets
    /data
        /VOCdevkit 
            /VOC2012 
                /SegmentationClass
                /JPEGImages
                ...
            ...
        /VOCtrainval_11-May-2012.tar
        ...
```

### 2.2  Pascal VOC trainaug (Recommended!!)

See chapter 4 of [2]

        The original dataset contains 1464 (train), 1449 (val), and 1456 (test) pixel-level annotated images. We augment the dataset by the extra annotations provided by [76], resulting in 10582 (trainaug) training images. The performance is measured in terms of pixel intersection-over-union averaged across the 21 classes (mIOU).

*./datasets/data/train_aug.txt* includes the file names of 10582 trainaug images (val images are excluded). Please to download their labels from [Dropbox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) or [Tencent Weiyun](https://share.weiyun.com/5NmJ6Rk). Those labels come from [DrSleep's repo](https://github.com/DrSleep/tensorflow-deeplab-resnet).

Extract trainaug labels (SegmentationClassAug) to the VOC2012 directory.

```
/datasets
    /data
        /VOCdevkit  
            /VOC2012
                /SegmentationClass
                /SegmentationClassAug  # <= the trainaug labels
                /JPEGImages
                ...
            ...
        /VOCtrainval_11-May-2012.tar
        ...
```

### 3. QAT training

```bash
 python main_qat.py --gpu_id 0 --year 2012_aug --lr 0.01 --crop_size 513 --batch_size 4
```

### 4. build trt engine

```bash
 python onnx2trt-qat.py -m {onnx model in step3} -d int8 --dynamic-shape
```

### 5. evaluate

https://github.com/shiyongming/Segmentation-models-benchmark