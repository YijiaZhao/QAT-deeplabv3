
# DeepLabv3-QAT

## 1. Prepare env

### 1.1 Prepare docker container 

```bash
docker run -it -v "$(pwd)":/workspace/deeplabv3 --name zhaoyijia_deeplabv3_3  --shm-size=128g --gpus=all --net=host nvcr.io/nvidia/pytorch:22.03-py3 /bin/bash
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

## 3. QAT training

### 3.1 Monkey-patch

```bash
 python main_qat.py --gpu_id 0 --year 2012_aug --lr 0.0015 --crop_size 513
```
### 3.2* Further optimize

Please refer to [TensorRT OSS/tool/pytorch-quantization/Further optimization](https://github.com/NVIDIA/TensorRT/blob/master/tools/pytorch-quantization/docs/source/tutorials/quant_resnet50.rst#further-optimization) for detail. And it is highly recommended to walk through the [Q/DQ Layer-Placement Recommendations](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#qdq-placement-recs) part of `TensorRT Developer Guide` before you start.    

```bash
 cp resnet.py /opt/conda/lib/python3.8/site-packages/torchvision/models/
 cp deeplabv3.py /opt/conda/lib/python3.8/site-packages/torchvision/models/segmentation/
 cp fcn.py /opt/conda/lib/python3.8/site-packages/torchvision/models/segmentation/
 python main_qat_opt.py --gpu_id 0 --year 2012_aug --lr 0.0015 --crop_size 513 --further_opt
```

## 4. Build trt engine

If you don't use Ampere or latter GPU, pay attention to https://github.com/NVIDIA/TensorRT/issues/1768
```bash
 python onnx2trt-qat.py -m {onnx model in step3} -d int8 --dynamic-shape
```

## 5. Evaluate

### 5.1 Metrics:
```bash
python trt_infer.py -t {pytorch model}
python trt_infer.py -e {trt model}
```
### 5.2 QPS:
```bash
 trtexec --loadEngine={trt path} --shapes=input0:1x3x384x512
 (trtexec --loadEngine={trt path} --minShapes=input0:1x3x128x128 --optShapes=input0:1x3x384x512 --maxShapes=input0:1x3x640x640)
```

## 6. Results:

Metrics of PTQ and QAT:
| Metrics | pytorch training result | PTQ result | pytorch QAT finetune result | QAT result |
| :-----| ----: | ----: | ----: | :----: |
| Overall Acc | 0.953910 | 0.952774 | 0.953986 | 0.953830 |
| Mean Acc cls | 0.899860 | 0.900350 | 0.89349 | 0.893213 |
| Mean IoU | 0.794017 | 0.790955 | 0.795448 | 0.794762 |
| FreqW Acc | 0.917793 | 0.915693 | 0.9171307 | 0.916915 |


Latency throughputs (One stream):
*Test results of TRT-FP16 is not stable, it is the average results that I fill in the table.
| Batch size | Device | TRT-FP16 | PTQ-INT8 | QAT(further_opt) |
| :-----| ----: | ----: | ----: | :----: |
| 1  | A100 | 310.812 | 364.246 | 364.883 |
| 2  | A100 | 297.864 | 383.624 | 382.978 |
| 4  | A100 | 307.828 | 388.310 | 387.781 |
| 8  | A100 | 304.260 | 388.608 | 388.266 |
| 16 | A100 | 282.661 | 373.197 | 385.939 |
| 32 | A100 | 229.290 | 360.739 | 365.773 |


## *7. PTQ
If you want to do ptq experiment of deeplabv3, refer to https://github.com/shiyongming/Segmentation-models-benchmark.