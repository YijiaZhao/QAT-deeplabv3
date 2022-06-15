import logging
import argparse
import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit

import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms.functional import convert_image_dtype
#from image_batcher import ImageBatcher
#from visualize import visualize_detections
from datasets import VOCSegmentation

from utils import ext_transforms as et
from torch.utils import data

logging.basicConfig(level=logging.ERROR)  # INFO, WARNING, ERROR
logging.getLogger("EngineInference").setLevel(logging.ERROR)
log = logging.getLogger("EngineInference")
from metrics import StreamSegMetrics
from tqdm import tqdm

class TensorRTInfer:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.context.active_optimization_profile = 0

    def set_bind_shape(self, shape):    
        self.context.set_binding_shape(0, (1, shape[0], shape[1], shape[2]))
        assert self.engine
        assert self.context
            
        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            #print("The binding name is:", name)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.context.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            # import pdb;pdb.set_trace()
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
            
    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        print("The input name is:", self.inputs[0]['name'])
        print("The input shape is:", self.inputs[0]['shape'])
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs
        

    def infer_new(self, batch, shape):
        
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch. Default: No scale postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """
        # Prepare the output data

        cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(batch))
        log.info("Start TRT engine")
        self.context.execute_v2(self.allocations)
        log.info("End TRT engine")
        # for o in range(len(outputs)):
        #     log.info("Number {:} output's name is:{:}".format(o, self.outputs[o]['name']))
        #     cuda.memcpy_dtoh(outputs[o], self.outputs[o]['allocation'])  
        return self.outputs


    def infer(self, batch, shape):
        
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch. Default: No scale postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """
        # Prepare the output data
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros((shape), dtype))

        # Process I/O and execute the network
        cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(batch))
        log.info("Start TRT engine")
        self.context.execute_v2(self.allocations)
        log.info("End TRT engine")
        for o in range(len(outputs)):
            log.info("Number {:} output's name is:{:}".format(o, self.outputs[o]['name']))
            cuda.memcpy_dtoh(outputs[o], self.outputs[o]['allocation'])  
        return outputs


timing_only = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", default=None, help="The serialized TensorRT engine")
    parser.add_argument("-t", "--torch_model", default=None, help="Pytorch model")
    parser.add_argument("-i", "--input", default='', help="Path to the image or directory to process")
    parser.add_argument("-c", "--selected_class", type=str, default=None, help="Class to be selected for segmentation")
    parser.add_argument("-o", "--output", default='', help="Directory where to save the visualization results")
    args = parser.parse_args()

    val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    val_dst = VOCSegmentation(root='./datasets/data', year='2012',
                              image_set='val', download=False, transform=val_transform)
    val_loader = data.DataLoader(
        val_dst, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    import time

    metrics = StreamSegMetrics(21)

    if args.engine:

        trt_infer = TensorRTInfer(args.engine)
        total_samples = 0        
        time_consume = 0
        metrics.reset()

        for i, (images, labels) in tqdm(enumerate(val_loader)):
            images = images.squeeze(dim=0)
            T1 = time.time()
            # import pdb;pdb.set_trace()
            shape = (images.shape)
            trt_infer.set_bind_shape(shape)
            infer_results = trt_infer.infer_new(images, shape)
            T2 = time.time()

            if not timing_only:
                outputs = []
                for shape, dtype in trt_infer.output_spec():
                    outputs.append(np.zeros((shape), dtype))
                for o in range(len(outputs)):
                    log.info("Number {:} output's name is:{:}".format(o, infer_results[o]['name']))
                    cuda.memcpy_dtoh(outputs[o], infer_results[o]['allocation'])

                infer_resulrs_tensor = torch.tensor(np.array(outputs))
                preds = infer_resulrs_tensor[0].detach().max(dim=1)[1].cpu().numpy()
                targets = labels.numpy()
                metrics.update(preds, targets)

            total_samples += 1
            time_consume += (T2 - T1)

        print('Time used:', time_consume, 's')
        print('Sample precessed:', (total_samples))
        print('FPS:', (total_samples)/time_consume)
        print('')
    
        if not timing_only:
            score = metrics.get_results()
            print(metrics.to_str(score))

    else:
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True).cuda()
        model.load_state_dict(torch.load(args.torch_model)['model_state'])
        model.eval()

        metrics.reset()
        total_samples = 0

        time_consume = 0

        for i, (images, labels) in tqdm(enumerate(val_loader)):
            T1 = time.time()
            images = images.to(device, dtype=torch.float32)
            outputs = model(images)
            T2 = time.time()

            if not timing_only:
                outputs = outputs['out'].cpu()
                preds = outputs.detach().max(dim=1)[1].numpy()
                targets = labels.numpy()
                metrics.update(preds, targets)

            total_samples += 1
            time_consume += (T2 - T1)
        print('origin Time used:', time_consume, 's')
        print('origin Sample precessed:', (total_samples))
        print('origin FPS:', (total_samples)/time_consume)
        print('')

        if not timing_only:
            score = metrics.get_results()
            print(metrics.to_str(score))

# trt-aqt
# Time used: 19.430200815200806 s
# Sample precessed: 1449
# FPS: 74.57462811534123

# Overall Acc: 0.954100
# Mean Acc cls: 0.901974
# Mean IoU: 0.795458
# FreqW Acc: 0.918008

# torch:
# origin Time used: 59.984097480773926 s
# origin Sample precessed: 1449
# origin FPS: 24.156402460909458

# Overall Acc: 0.9538879663109074
# Mean Acc cls: 0.8979672222079454
# Mean IoU: 0.794061238824295
# FreqW Acc: 0.9175898435082801