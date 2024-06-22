import numpy as np

from network import make_network
import tqdm
import torch
import time
import nms
from dataset.data_loader import make_data_loader
from train.model_utils.utils import load_network
from evaluator.make_evaluator import make_evaluator
import argparse
import importlib
def IOU(Reframe, GTframe):
    # 得到第一个矩形的左上坐标及宽和高
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]
    height1 = Reframe[3]

    # 得到第二个矩形的左上坐标及宽和高
    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]
    height2 = GTframe[3]
    # 计算重叠部分的宽和高
    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    # 如果重叠部分为负, 即不重叠
    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1.0 / (Area1 + Area2 - Area)

    return ratio
parser = argparse.ArgumentParser()

parser.add_argument("config_file", help='/path/to/config_file.py')
parser.add_argument("--checkpoint", default='', help='/path/to/model_weight.pth')
parser.add_argument("--dataset", default='None', help='test dataset name')
parser.add_argument("--with_nms", default=False, type=bool,
                    help='if True, will use nms post-process operation', choices=[True, False])
parser.add_argument("--eval", default='segm', help='evaluate the segmentation or detection result',
                    choices=['segm', 'bbox'])
parser.add_argument("--stage", default='final-dml', help='which stage of the contour will be generated',
                    choices=['init', 'coarse', 'final', 'final-dml'])
parser.add_argument("--type", default='accuracy', help='evaluate the accuracy or speed',
                    choices=['speed', 'accuracy'])
parser.add_argument("--device", default=0, type=int, help='device idx')

args = parser.parse_args()

def get_cfg(args):
    cfg = importlib.import_module('configs.' + args.config_file).config
    cfg.test.with_nms = bool(args.with_nms)
    cfg.test.segm_or_bbox = args.eval
    cfg.test.test_stage = args.stage
    if args.dataset != 'None':
        cfg.test.dataset = args.dataset
    return cfg

def run_network(cfg):
    network = make_network.get_network(cfg).cuda()
    load_network(network, args.checkpoint)
    network.eval()

    data_loader = make_data_loader(is_train=False, cfg=cfg)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        inp = batch['inp'].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(inp, batch)
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader), '{} FPS'.format(len(data_loader) / total_time))

def removeDuplicate(output):
    ex = output['py']
    ex = ex[-1] if isinstance(ex, list) else ex
    ex = ex.detach().cpu().numpy()
    waitForDel = []
    for i in range(len(ex)):
        if i == len(ex) - 1:
            break
        curleft = min(ex[i][:, 0])
        curtop = min(ex[i][:, 1])
        curright = max(ex[i][:, 0])
        curbottom = max(ex[i][:, 1])
        for j in range(i + 1, len(ex)):
            if j == len(ex) - 1:
                break
            tarleft = min(ex[j][:, 0])
            tartop = min(ex[j][:, 1])
            tarright = max(ex[j][:, 0])
            tarbottom = max(ex[j][:, 1])
            if IOU((curleft, curtop, curright, curbottom), (tarleft, tartop, tarright, tarbottom)) > 0.8:
                waitForDel.append(j)
    ex = np.delete(ex, waitForDel, axis=0)
    output['py']=ex
    return output
def run_evaluate(cfg):
    network = make_network.get_network(cfg).cuda()
    load_network(network, args.checkpoint)
    network.eval()
    data_loader = make_data_loader(is_train=False, cfg=cfg)
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        inp = batch['inp'].cuda()
        with torch.no_grad():
            output = network(inp, batch)
        if cfg.test.with_nms:
            nms.post_process(output)
        output=removeDuplicate(output)
        evaluator.evaluate(output, batch)
    evaluator.summarize()


if __name__ == "__main__":
    cfg = get_cfg(args)
    torch.cuda.set_device(args.device)
    if args.type == 'speed':
        run_network(cfg)
    else:
        run_evaluate(cfg)
