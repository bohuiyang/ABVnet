from models import models_vit
import sys
import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from models.Generate_Model import GenerateModel
import matplotlib

# from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, \
#     XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
# from pytorch_grad_cam import GuidedBackpropReLUModel
# from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


import cv2
import numpy as np
import torch


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
import datetime
from dataloader.video_dataloader import train_data_loader, test_data_loader
from sklearn.metrics import confusion_matrix
import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import random

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
def hard_remove_dropout(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Dropout):
            print(f"[REMOVE] {name}: Dropout removed")
            setattr(model, name, torch.nn.Identity())
        else:
            hard_remove_dropout(module)

    # 额外处理 module.modules() 中未注册为属性的 Dropout
    for m in model.modules():
        for name, child in list(m.__dict__.items()):
            if isinstance(child, torch.nn.Dropout):
                print(f"[DELETE] Untracked Dropout found in {m}, removing...")
                setattr(m, name, torch.nn.Identity())

#--checkpoint "/home/mnt_disk2/log/DFEW-2411242336FINAL_224-set1-log-最新最好结果/checkpoint/model.pth"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DFEW')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--temporal-layers', type=int, default=1)
    parser.add_argument('--img-size', type=int, default=224)

    parser.add_argument('--workers', type=int, default=8)

    parser.add_argument('--fold', type=int, default=1)
    args = parser.parse_args()
    return args


def main(set, args):
    data_set = set

    if args.dataset == "DFEW":
        print("*********** DFEW Dataset Fold  " + str(data_set) + " ***********")
        test_annotation_file_path = "./annotation/DFEW_set_" + str(data_set) + "_test.txt"
        args.number_class = 7
    elif args.dataset == "MAFW":
        print("*********** MAFW Dataset Fold  " + str(data_set) + " ***********")
        test_annotation_file_path = "./annotation/MAFW_set_" + str(data_set) + "_test_faces.txt"
        args.number_class = 11
    model = GenerateModel(args=args)

    model = model.cuda()
    test_data = test_data_loader(list_file=test_annotation_file_path,
                                 num_segments=16,
                                 duration=1,
                                 image_size=args.img_size)

    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=8,  # args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)


    # # 模拟一个 batch 输入
    # dummy_video = torch.randn(8, 16, 3, 224, 224).cuda()
    # dummy_audio = torch.randn(8, 1, 448, 112).cuda()

    # model.eval()
    # # hard_remove_dropout(model)
    #
    # # 构造输入
    # video = torch.randn(2, 16, 3, 224, 224).cuda()
    # audio = torch.randn(2, 1, 448, 112).cuda()
    #
    # # FLOPs 计算
    # from fvcore.nn import FlopCountAnalysis
    # flops = FlopCountAnalysis(model, (video, audio))
    # print("Total FLOPs: {:.2f} GFLOPs".format(flops.total() / 1e9))
    #
    # with torch.no_grad():
    #     flops = FlopCountAnalysis(model, (dummy_video, dummy_audio))
    #     # 打印 FLOPs 报表
    #     print("Total FLOPs: {:.2f} GFLOPs".format(flops.total() / 1e9))
    #
    #     # print(flop_count_table(flops, max_depth=2))
    #     print("FLOPs:", flops.total())
    #     print("FLOPs readable:", flops.total() / 1e9, "GFLOPs")
    #     print(parameter_count_table(model))

    uar, war = computer_uar_war(val_loader, model, args.checkpoint, data_set)

    return uar, war


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_txt_path=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_txt_path = log_txt_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(self.log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            # print('Curve was saved')
        plt.close(fig)


def plot_confusion_matrix(cm, classes, normalize=False, title='confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=12,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()


data_embed_collect = []
label_collect = []


# 首先定义函数对vit输出的3维张量转换为传统卷积处理时的二维张量，gradcam需要。
# （B,H*W,feat_dim）转换到（B,C,H,W）,其中H*W是分pathc数。具体参数根据自己模型情况
# 我的输入为224*224，pathsize为（16*16），那么我的（H，W）就是(224/16，224/16)，即14*14
# def reshape_transform(tensor, height=14, width=14):
#     # 去掉cls token
#     result = tensor[:, 1:, :].reshape(tensor.size(0),
#                                       height, width, tensor.size(2))
#     # 将通道维度放到第一个位置
#     result = result.transpose(2, 3).transpose(1, 2)
#     return result
#
#
# # 创建 GradCAM 对象
# cam = GradCAM(model=model,
#               target_layers=[model.blocks[-1].norm1],
#               # 这里的target_layer要看模型情况，调试时自己打印下model吧
#               # 比如还有可能是：target_layers = [model.blocks[-1].ffn.norm]
#               # 或者target_layers = [model.blocks[-1].ffn.norm]
#               use_cuda=use_cuda,
#               reshape_transform=reshape_transform)
import cv2
import numpy as np
import matplotlib.pyplot as plt


def overlay_attn_on_image_key_region(attn_map, image, alpha=0.4, threshold=0.7):
    """
    attn_map: [14, 14] attention map
    image: [H, W, 3] uint8 RGB
    threshold: 显示注意力强度 top 区域（值域 0~1）
    """
    h, w = image.shape[:2]

    # 1. 归一化 attention
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # 2. resize to original image size
    attn_resized = cv2.resize(attn_map, (w, h))  # shape: [H, W]

    # 3. 创建二值 mask，只保留 attention > threshold 的区域
    mask = (attn_resized >= threshold).astype(np.uint8)  # [H, W] ∈ {0,1}

    # 4. 生成 heatmap
    attn_heat = np.uint8(255 * attn_resized)
    heatmap = cv2.applyColorMap(attn_heat, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 5. 把 mask 扩展到 3 通道
    mask_3ch = np.stack([mask] * 3, axis=-1)

    # 6. 只在 mask 区域内做 overlay，其他区域为原图
    overlay = image.copy()
    overlay[mask_3ch == 1] = cv2.addWeighted(
        image[mask_3ch == 1], 1 - alpha, heatmap[mask_3ch == 1], alpha, 0)

    return overlay


from PIL import Image
import numpy as np
import cv2
from PIL import Image, ImageDraw

def apply_colormap(attn_resized, colormap=cv2.COLORMAP_TURBO):
    heat = np.uint8(255 * attn_resized)
    heatmap = cv2.applyColorMap(heat, colormap)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

def blend_images(img, heatmap, attn_resized, alpha=0.6):
    alpha_map = (attn_resized[..., None] * alpha)
    return (img * (1 - alpha_map) + heatmap * alpha_map).astype(np.uint8)
def visualize_attention_row(attn_map, your_images, batch_idx=0, gap=10):
    """
    升级版：注意力可视化，第一排是原图，第二排是叠加了注意力的图。
    参数：
    - attn_map: Tensor of shape [B, 16, 12, 197, 197]
    - your_images: Tensor of shape [B, 16, 3, 224, 224]
    - batch_idx: 可视化第几个样本
    - gap: 图与图之间的统一间隔
    """
    from PIL import Image
    import numpy as np
    import cv2

    attn_map = attn_map.reshape(-1, 16, 12, 197, 197)
    device = attn_map.device

    original_frames = []
    overlay_frames = []

    for t in range(16):
        # 1. 取平均注意力
        attn = attn_map[batch_idx, t].mean(0)  # [197, 197]
        cls_attn = attn[0, 1:].reshape(14, 14).detach().cpu().numpy()

        # 2. 归一化并放大到 224x224
        cls_attn = np.maximum(cls_attn, 0)
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.ptp() + 1e-8)
        attn_resized = cv2.resize(cls_attn, (224, 224))

        # 3. 生成热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 4. 原图
        img = your_images[batch_idx, t].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        original_frames.append(Image.fromarray(img))

        # 5. 叠加热力图
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        overlay_frames.append(Image.fromarray(overlay))

    # 6. 创建总图
    img_width, img_height = original_frames[0].size

    total_width = 16 * img_width + 17 * gap   # 16张图+17个gap
    total_height = 2 * img_height + 3 * gap   # 两行：原图一行，叠加图一行，每行上下都留gap

    result_img = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))

    # 7. 贴第一排：原图
    for i, frame in enumerate(original_frames):
        x = gap + i * (img_width + gap)
        y = gap
        result_img.paste(frame, (x, y))

    # 8. 贴第二排：叠加后的图
    for i, frame in enumerate(overlay_frames):
        x = gap + i * (img_width + gap)
        y = gap * 2 + img_height  # 第二排开始
        result_img.paste(frame, (x, y))

    return result_img




from pytorch_grad_cam import GradCAM

def reshape_transform(tensor, height=14, width=14):
    # 去除 class token，并重新调整形状
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # 调整通道维度顺序
    result = result.permute(0, 3, 1, 2)
    return result

def computer_uar_war(val_loader, model, checkpoint_path, data_set):
    pre_trained_dict = torch.load(checkpoint_path)['state_dict']
    pre_trained_dict = {k.replace("module.", ""): v for k, v in pre_trained_dict.items()}
    pre_trained_dict = {k.replace("adaptmlp.", "G_Adapter."): v for k, v in pre_trained_dict.items()}
    pre_trained_dict = {k.replace("SAdapter2.", "S_Adapter."): v for k, v in pre_trained_dict.items()}
    pre_trained_dict = {k.replace("SAdapter_fuse.", "Adapter1."): v for k, v in pre_trained_dict.items()}
    pre_trained_dict = {k.replace("SAdapter_fuse2.", "Adapter2."): v for k, v in pre_trained_dict.items()}
    pre_trained_dict = {k.replace("temporal_net.", "CMTM."): v for k, v in pre_trained_dict.items()}
    pre_trained_dict = {k.replace("gate_fusion0", "gate_fusion1"): v for k, v in pre_trained_dict.items()}

    result = model.load_state_dict(pre_trained_dict, strict=False)

    # 打印未加载的参数
    print("Missing keys (未加载的参数):")
    for key in result.missing_keys:
        print(f"- {key}")

    # 打印未使用的预训练参数
    print("\nUnexpected keys (预训练中存在但模型不需要的参数):")
    for key in result.unexpected_keys:
        print(f"- {key}")
    model.eval()

    model_state = model.state_dict()
    unmatched_keys = []

    for key in model_state.keys():
        if key not in pre_trained_dict:
            unmatched_keys.append(key)
        elif model_state[key].shape != pre_trained_dict[key].shape:
            unmatched_keys.append(key)

    print("Unmatched keys:", unmatched_keys)
    predicteds = []
    targets = []
    probabilitiess = []
    paths = []
    correct = 0
    with torch.no_grad():
        for i, (images, target, audio , path) in enumerate(tqdm.tqdm(val_loader)):

            images = images.cuda()
            target = target.cuda()
            audio = audio.cuda()
            output = model(images, audio)

            # 计算 Softmax
            softmax = torch.nn.Softmax(dim=1)
            probabilities = softmax(output)

            # 设置输出格式，禁用科学计数法
            torch.set_printoptions(sci_mode=False)

            # 输出每个样本的 Softmax 概率分布
            print("Softmax 概率分布:")
            print(probabilities)
            # target_layers = [model.blocks[-1].norm1]
            # input_tensor = images.view(-1, 3, 224, 224)
            #
            # cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
            # # 生成 CAM
            # grayscale_cam = cam([images, audio])
            #
            # # 将 CAM 重新调整为 (batch_size, frames, 224, 224)
            # grayscale_cam = grayscale_cam.reshape(batch_size, 16, 224, 224)
            #
            # # 可视化每一帧
            # for b in range(batch_size):
            #     for t in range(16):
            #         # 获取原始帧
            #         img = images[b, t].permute(1, 2, 0).cpu().numpy()
            #         img = (img * 255).astype(np.uint8)
            #
            #         # 获取对应的 CAM
            #         cam_img = grayscale_cam[b, t]
            #
            #         # 将 CAM 映射到颜色空间
            #         heatmap = cv2.applyColorMap(np.uint8(255 * cam_img), cv2.COLORMAP_JET)
            #         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            #
            #         # 叠加 CAM 到原始图像
            #         overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
            #
            #         # 显示或保存 overlay 图像
            #         plt.imshow(overlay)
            #         plt.axis('off')
            #         plt.show()


            attn = model.blocks[-1].attn.attn_map
            # rows = []
            # for i in range(8):  # 共8个视频样本
            #     row = visualize_attention_row(attn, images, batch_idx=i,gap = 5)
            #     rows.append(row)
            #
            # # 每行尺寸
            # frame_width, frame_height = 224, 224
            # row_width = 5 + (frame_width + 5) * 16
            # row_height = frame_height + 2 * 5
            #
            # # 计算整个final图的尺寸
            # final_img = Image.new('RGB', (row_width, row_height * 8), color=(255, 255, 255))
            #
            # # 依次粘贴每一行
            # for i, row in enumerate(rows):
            #     final_img.paste(row, (0, i * row_height))
            #
            # # 保存显示
            # final_img.save("attention_map_video_overlay24.png")
            # final_img.show()
            # if target[0] == 8:
            #
            #     for i in range(8):  # 共8个视频样本
            #         if(target[i] == predicted[i]):
            #             row = visualize_attention_row(attn, images, batch_idx=i, gap=5)
            #             save_path = f"attention_map_video_{i}.png"
            #             row.save(save_path)
            #             # 如果不想每张都弹出来，可以注释掉
            #             row.show()
            #             print(1)

            # for i in range(8):
            #     if "00049" in path[i]:
            #         row = visualize_attention_row(attn, images, batch_idx=i, gap=5)
            #         save_path = f"dfew_attention_map_video_{i}.png"
            #         row.save(save_path)
            #         # 如果不想每张都弹出来，可以注释掉
            #         row.show()
            # for i in range(8):
            #     if "00057" in path[i]:
            #         row = visualize_attention_row(attn, images, batch_idx=i, gap=5)
            #         save_path = f"dfew_attention_map_video_{i}.png"
            #         row.save(save_path)
            #         # 如果不想每张都弹出来，可以注释掉
            #         row.show()
            # for i in range(8):
            #     if "00059" in path[i]:
            #         row = visualize_attention_row(attn, images, batch_idx=i, gap=5)
            #         save_path = f"dfew_attention_map_video_{i}.png"
            #         row.save(save_path)
            #         # 如果不想每张都弹出来，可以注释掉
            #         row.show()


            predicted = output.argmax(dim=1, keepdim=True)
            correct += predicted.eq(target.view_as(predicted)).sum().item()

            data_embed_collect.append(output)
            label_collect.append(target)
            predicteds.append(predicted)
            targets.append(target)
            probabilitiess.append(probabilities)
            paths.append(path)

            if i == 0:
                all_predicted = predicted
                all_targets = target
            else:
                all_predicted = torch.cat((all_predicted, predicted), 0)
                all_targets = torch.cat((all_targets, target), 0)

    war = 100. * correct / len(val_loader.dataset)

    # Compute confusion matrix
    _confusion_matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
    np.set_printoptions(precision=4)
    normalized_cm = _confusion_matrix.astype('float') / _confusion_matrix.sum(axis=1)[:, np.newaxis]
    normalized_cm = normalized_cm * 100
    list_diag = np.diag(normalized_cm)
    uar = list_diag.mean()

    print("Confusion Matrix Diag:", list_diag)
    print("UAR: %0.2f" % uar)
    print("WAR: %0.2f" % war)
    return uar, war


if __name__ == '__main__':
    args = parse_args()
    print('************************')
    for k, v in vars(args).items():
        print(k, '=', v)
    print('************************')
    uar, war = main(args.fold, args)

    print('********* Final Results *********')
    print("UAR: %0.2f" % (uar))
    print("WAR: %0.2f" % (war))
    print('*********************************')
    data_embed_npy = torch.cat(data_embed_collect, axis=0).cpu().numpy()
    label_npy = torch.cat(label_collect, axis=0).cpu().numpy()

    np.save("mafw_data_embed_npy.npy", data_embed_npy)
    np.save("mafw_label_npy.npy", label_npy)
    data_embed_npy = torch.cat(data_embed_collect, axis=0).cpu().numpy()
    label_npy = torch.cat(label_collect, axis=0).cpu().numpy()

    np.save("mafw_data_embed_npy.npy", data_embed_npy)
    np.save("mafw_label_npy.npy", label_npy)


