from tqdm import tqdm
# import network_
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
# from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

import torchvision

from copy import deepcopy

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules

import collections
import onnx
def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    # available_models = sorted(name for name in network_.modeling.__dict__ if name.islower() and \
    #                           not (name.startswith("__") or name.startswith('_')) and callable(
    #                           network_.modeling.__dict__[name])
    #                           )
    # parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
    #                     choices=available_models, help='model name')
    # parser.add_argument("--separable_conv", action='store_true', default=False,
    #                     help="apply separable conv to decoder and aspp")
    # parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--accu_tolerance", type=float, default=3.0,
                        help="accu_tolerance (default: 0.01)")

    parser.add_argument("--ckpt_path", type=str, default='sensity_ckpt',
                        help="ckpt_path (default: sensity_ckpt)")

    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=1000,
                        help="epoch interval for eval (default: 1000)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    train_transform = et.ExtCompose([
        # et.ExtResize(size=opts.crop_size),
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    if opts.crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                image_set='train', download=opts.download, transform=train_transform)
    val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                image_set='val', download=False, transform=val_transform)
    calib_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                image_set='calib', download=False, transform=val_transform)

    return train_dst, val_dst, calib_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(images)
            preds = outputs['out'].detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(preds, targets)
            # 
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples

def collect_stats(model, data_loader, num_batches, device):
    """Feed data to the network and collect statistics"""
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                # 
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, (img, _) in tqdm(enumerate(data_loader), total=num_batches):
        img = img.to(device, non_blocking=True)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        model(img)
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            # print(F"{name:40}: {module}")
    model.cuda()

def build_sensitivity_profile(model, opts, testloader, device):
    quant_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.disable()
            layer_name = name.replace("._input_quantizer", "").replace("._weight_quantizer", "")
            if layer_name not in quant_layer_names:
                quant_layer_names.append(layer_name)
    print(F"{len(quant_layer_names)} quantized layers found.")

    # Build sensitivity profile
    quant_layer_sensitivity = {}
    for i, quant_layer in enumerate(quant_layer_names):
        print(F"Enable {quant_layer}")
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer) and quant_layer in name:
                module.enable()
                print(F"{name:40}: {module}")

        # Eval the model
        metrics_sens = StreamSegMetrics(opts.num_classes)
        val_score, ret_samples = validate(opts=opts, model=model, loader=testloader, device=device, metrics=metrics_sens)
        print(metrics_sens.to_str(val_score))
        
        res = val_score['Overall Acc'] + val_score['Mean Acc cls'] + val_score['Mean IoU'] + val_score['FreqW Acc']

        quant_layer_sensitivity[quant_layer] = opts.accu_tolerance - res

        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer) and quant_layer in name:
                module.disable()
                print(F"{name:40}: {module}")

    # Skip most sensitive layers until accuracy target is met
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.enable()
    quant_layer_sensitivity = collections.OrderedDict(sorted(quant_layer_sensitivity.items(), key=lambda x: x[1]))
    print(quant_layer_sensitivity)

    # 
    skipped_layers = []
    for quant_layer, _ in quant_layer_sensitivity.items():
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if quant_layer in name:
                    print(F"Disable {name}")
                    if not quant_layer in skipped_layers:
                        skipped_layers.append(quant_layer)
                    module.disable()
        metrics_sens_1 = StreamSegMetrics(opts.num_classes)
        val_score, ret_samples = validate(opts=opts, model=model, loader=testloader, device=device, metrics=metrics_sens_1)
        # map50, map50_95 = evaluate_accuracy(model, opts, testloader)
        res = val_score['Overall Acc'] + val_score['Mean Acc cls'] + val_score['Mean IoU'] + val_score['FreqW Acc']

        if res >= opts.accu_tolerance + 0.3:
            print(F"Accuracy tolerance {opts.accu_tolerance} is met by skipping {len(skipped_layers)} sensitive layers.")
            print(skipped_layers)
            # onnx_filename = opts.ckpt_path.replace('.pt', F'_skip{len(skipped_layers)}.onnx')
            # export_onnx(model, onnx_filename, opt.batch_size_onnx, opt.dynamic)
            return
    raise ValueError(f"Accuracy tolerance {opts.accu_tolerance} can not be met with any layer quantized!")

def skip_sensitive_layers(model, opts, testloader, device):
    print('Skip the sensitive layers.')
    # Sensitivity layers for yolov5s
    # skipped_layers = ['module.backbone.conv1',          # the first conv
    #                   'module.backbone.layer1.0.downsample.0',      # the second conv
    #                   'module.backbone.layer1.0.conv1',
    #                   'module.backbone.layer2.0.conv2',          # detect layer
    #                   'module.backbone.layer1.1.conv2',          # detect layer
    #                   'module.backbone.layer1.1.conv3',          # detect layer
    #                   'module.backbone.layer1.0.conv2',          # detect layer
    #                   'module.backbone.layer2.0.conv3',
    #                   'module.backbone.layer3.0.conv1',
    #                   'module.backbone.layer2.0.conv1',
    #                   'module.backbone.layer2.0.downsample.0',
    #                   'module.backbone.layer1.2.conv3',
    #                   ]

    skipped_layers = ['backbone.conv1',          # the first conv
                      'backbone.layer1.0.downsample.0',      # the second conv
                      'backbone.layer1.0.conv1',
                    #   'backbone.layer2.0.conv2',          # detect layer
                    #   'backbone.layer1.1.conv2',          # detect layer
                    #   'backbone.layer1.1.conv3',          # detect layer
                    #   'backbone.layer1.0.conv2',          # detect layer
                    #   'module.backbone.layer2.0.conv3',
                    #   'module.backbone.layer3.0.conv1',
                    #   'module.backbone.layer2.0.conv1',
                    #   'module.backbone.layer2.0.downsample.0',
                    #   'module.backbone.layer1.2.conv3',
                      ]

    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            layer_name = name.replace("._input_quantizer", "").replace("._weight_quantizer", "")
            if layer_name in skipped_layers:
                print(F"Disable {name}")
                module.disable()
    
    # metrics_sens = StreamSegMetrics(opts.num_classes)
    # val_score, ret_samples = validate(opts=opts, model=model, loader=testloader, device=device, metrics=metrics_sens)
    # print(metrics_sens.to_str(val_score))
   
     # print(F"mAP@IoU=0.50: {map50}, mAP@IoU=0.50:0.95: {map50_95}")

    # onnx_filename = opt.ckpt_path.replace('.pt', F'_skip{len(skipped_layers)}.onnx')
    # export_onnx(model, onnx_filename, opt.batch_size_onnx, opt.dynamic)
    return

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    # Setup visualization
    # vis = Visualizer(port=opts.vis_port,
    #                  env=opts.vis_env) if opts.enable_vis else None
    # if vis is not None:  # display options
    #     vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst, calib_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=0)
    calib_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=0)

    print("Dataset: %s, Train set: %d, Val set: %d, Cal set: %d" %
          (opts.dataset, len(train_dst), len(val_dst), len(calib_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    # model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)

    quant_desc_input = QuantDescriptor()
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    quant_modules.initialize()
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    quant_modules.deactivate()

    # model.load_state_dict(torch.load('./checkpoints_qat/best_deeplabv3_voc_os.pth')['model_state'])
    model.load_state_dict(torch.load('./checkpoints/latest_deeplabv3_voc_os.pth')['model_state'])
    model.eval()
    model.cuda()


    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    batch_size = 1
    color_channel = 3
    height = 256
    width = 256

    input_names = ['input0'] + ['learned_%d' %i for i in range(16)]
    output_names = ['dense_out']
    dummy_input = torch.randn(batch_size, color_channel, width, height, device='cuda')
    onnx_model_name = 'temp.onnx'
    dynamic_axes = {'input0': {0:'batch', 2:'height', 3:'width'}}
    # torch.onnx.export(model,
    #                  dummy_input,
    #                  onnx_model_name,
    #                  verbose=False, 
    #                  opset_version=13,
    #                  input_names=input_names,
    #                  output_names=output_names,
    #                  dynamic_axes=dynamic_axes,
    #                  do_constant_folding=True)
    model.eval()
    torch.onnx.export(model,
                        dummy_input,
                        onnx_model_name,
                        verbose=False, 
                        opset_version=13,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=dynamic_axes)
    return

    quant_nn.TensorQuantizer.use_fb_fake_quant = False



    num_calib_batch = 10
    with torch.no_grad():
        collect_stats(model, calib_loader, num_calib_batch, device)

    # for percentile in [99.9, 99.99, 99.999, 99.9999]:
    #     print(F"{percentile} percentile calibration")
    #     compute_amax(model, method="percentile")
    #     calib_output = os.path.join(
    #         "qat_cali_finetune",
    #         F"{'deeplabv3'}-percentile-{percentile}-{num_calib_batch*calib_loader.batch_size}.pth")

    #     ckpt = {'model': deepcopy(model)}
    #     torch.save(ckpt, calib_output)

    # for method in ["mse", "entropy"]:
    #     print(F"{method} calibration")
    #     compute_amax(model, method=method)

    #     calib_output = os.path.join(
    #         "qat_cali_finetune",
    #         F"{'deeplabv3'}-{method}-{num_calib_batch*calib_loader.batch_size}.pth")

    #     ckpt = {'model': deepcopy(model)}
    #     torch.save(ckpt, calib_output)


    compute_amax(model, method="entropy")
    ckpt = {'model': deepcopy(model)}
    torch.save(ckpt, "qat_cali_fintune/entropy_.pth")


    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr}, 
        {'params': model.classifier.parameters(), 'lr': opts.lr},
        ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints_qat')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        # model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        # model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        print("begin validate")
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    # model.eval()
    # print("begin validate before skip")
    # val_score, ret_samples = validate(
    #     opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
    # print(metrics.to_str(val_score))

    skip_sensitive_layers(model, opts, val_loader, device)
    
    # model.eval()
    # print("begin validate")
    # val_score, ret_samples = validate(
    #     opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
    # print(metrics.to_str(val_score))

    # build_sensitivity_profile(model, opts, val_loader, device)

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            # 
            outputs = model(images)
            
            loss = criterion(outputs['out'], labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            # if vis is not None:
            #     vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

                # return
                
            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints_qat/latest_%s_%s_os.pth' %
                          ("deeplabv3", opts.dataset))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints_qat/best_%s_%s_os.pth' %
                              ("deeplabv3", opts.dataset))

                    quant_nn.TensorQuantizer.use_fb_fake_quant = True
                    batch_size = 1
                    color_channel = 3
                    height = 256
                    width = 256

                    input_names = ['input0'] + ['learned_%d' %i for i in range(16)]
                    output_names = ['dense_out']
                    dummy_input = torch.randn(batch_size, color_channel, width, height, device='cuda')
                    os.system("rm -rf onnx_models/*")
                    onnx_model_name = 'onnx_models/'+ str(val_score['Mean IoU']) + '.onnx'
                    dynamic_axes = {'input0': {0:'batch', 2:'height', 3:'width'}}
                    # torch.onnx.export(model,
                    #                  dummy_input,
                    #                  onnx_model_name,
                    #                  verbose=False, 
                    #                  opset_version=13,
                    #                  input_names=input_names,
                    #                  output_names=output_names,
                    #                  dynamic_axes=dynamic_axes,
                    #                  do_constant_folding=True)
                    model.eval()
                    torch.onnx.export(model,
                                     dummy_input,
                                     onnx_model_name,
                                     verbose=False, 
                                     opset_version=13,
                                     input_names=input_names,
                                     output_names=output_names,
                                     dynamic_axes=dynamic_axes)

                    quant_nn.TensorQuantizer.use_fb_fake_quant = False
                    # model_o = onnx.load(onnx_model_name)
                    # info = onnx.checker.check_model(onnx_model_name)
                    # print(info)
                    # quant_nn.TensorQuantizer.use_fb_fake_quant = True


                # if vis is not None:  # visualize validation score and samples
                #     vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                #     vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                #     vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                #     for k, (img, target, lbl) in enumerate(ret_samples):
                #         img = (denorm(img) * 255).astype(np.uint8)
                #         target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                #         lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                #         concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                #         vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:

                return


if __name__ == '__main__':
    main()
#python main_qat.py --gpu_id 0 --year 2012_aug --lr 0.01 --crop_size 513 --batch_size 4