
# DeepLabv3-QAT

## 1. prepare env

### 1.1 prepare docker container 

```bash
docker run -it -v "$(pwd)":/workspace/deeplabv3 --name zhaoyijia_deeplabv3_3  --shm-size=128g --gpus device=0 --net=host nvcr.io/nvidia/pytorch:22.03-py3 /bin/bash
```

### 1.2 Requirements

```bash
pip install -r requirements.txt
```

## 2. Prepare Datasets

### 2.1 Standard Pascal VOC
You can run main_qat.py with "--download" option to download and extract pascal voc dataset. The defaut path is './datasets/data' (if you have VOCtrainval_11-May-2012.tar in your device, copy it to ./datasets/data):

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


The original dataset contains 1464 (train), 1449 (val), and 1456 (test) pixel-level annotated images. We augment the dataset by the extra annotations, resulting in 10582 (trainaug) training images. The performance is measured in terms of pixel intersection-over-union averaged across the 21 classes (mIOU). Please download the SegmentationClassAug.zip and Extract trainaug labels (SegmentationClassAug) to VOC2012 directory(datasets/data/VOCdevkit/VOC2012/).

*./datasets/data/train_aug.txt* includes the file names of 10582 trainaug images (val images are excluded). Please to download their labels from [Dropbox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) or [Tencent Weiyun](https://share.weiyun.com/5NmJ6Rk). Those labels come from [DrSleep's repo](https://github.com/DrSleep/tensorflow-deeplab-resnet).



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
 python main_qat.py --gpu_id 0 --year 2012_aug --lr 0.01 --crop_size 513 --batch_size 8
```

### 4. build trt engine

```bash
 python onnx2trt-qat.py -m {onnx model in step3} -d int8 --dynamic-shape
```

### 5. evaluate
```bash
python trt_infer.py -t {pytorch model}
python trt_infer.py -e {trt model}
```

### 6. results:
PTQ
Overall Acc: 0.9526
Mean Acc cls: 0.89945155
Mean IoU: 0.78962433
FreqW Acc: 0.9154

QAT:
Overall Acc: 0.9539
Mean Acc cls: 0.901357
Mean IoU: 0.7942094
FreqW Acc: 0.9178

inference thouthputs (batch size = 1, A100):
Pytorch: 24.156 qps
QAT: 74.575 qps

### *7. PTQ
If you want to do ptq experiment of deeplabv3, refer to https://github.com/shiyongming/Segmentation-models-benchmark.
