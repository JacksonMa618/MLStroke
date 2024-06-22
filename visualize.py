import cv2
import matplotlib

from network import make_network
import tqdm
import torch
import os
import nms
import post_process
from dataset.data_loader import make_demo_loader
from train.model_utils.utils import load_network
import argparse
import importlib
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
matplotlib.use("Agg")

parser = argparse.ArgumentParser()

parser.add_argument("config_file", help='/configs/coco.py')
parser.add_argument("image_dir", help='/path/to/images')
parser.add_argument("--checkpoint", default='', help='/path/to/model_weight.pth')
parser.add_argument("--ct_score", default=0.2, help='threshold to filter instances', type=float)
parser.add_argument("--with_nms", default=False, type=bool,
                    help='if True, will use nms post-process operation', choices=[True, False])
parser.add_argument("--with_post_process", default=False, type=bool,
                    help='if True, Will filter out some jaggies', choices=[True, False])
parser.add_argument("--stage", default='final-dml', help='which stage of the contour will be generated',
                    choices=['init', 'coarse', 'final', 'final-dml'])
parser.add_argument("--output_dir", default='None', help='/path/to/output_dir')
parser.add_argument("--device", default=0, type=int, help='device idx')

args = parser.parse_args()
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


def get_cfg(args):
    cfg = importlib.import_module('configs.' + args.config_file).config
    cfg.test.with_nms = bool(args.with_nms)
    cfg.test.test_stage = args.stage
    cfg.test.ct_score = args.ct_score
    return cfg

def bgr_to_rgb(img):
    return img[:, :, [2, 1, 0]]

def unnormalize_img(img, mean, std):
    """
    img: [3, h, w]
    """
    img = img.detach().cpu().clone()
    img *= torch.tensor(std).view(3, 1, 1)
    img += torch.tensor(mean).view(3, 1, 1)
    min_v = torch.min(img)
    img = (img - min_v) / (torch.max(img) - min_v)
    return img

class Visualizer(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def visualize_ex(self, output, batch, save_dir=None):
        # categories=['点右','点左','折勾','横','横折钩','横钩','横撇','横撇弯钩','横折','横折钩','横折提','横折弯','横折弯钩','横折折撇',
        #             '横折折折钩','弧弯钩','捺','撇','撇点','撇折','竖','竖钩','竖提','竖弯','竖弯钩','竖折','竖折撇','竖折折','竖折折钩','提','卧钩','走头','错误','交叉']
        categories = ['wangou', 'na', 'ti', 'pie', 'piezhe', 'piedian', 'xiegouhuowogou', 'heng', 'hengzhe',
                      'hengzhezhehuohengzhewan', 'hengzhezhezhe', 'hengzhezhezhegouhuohengpiewangou', 'hengzhezhepie',
                      'hengzheti', 'hengzhegou', 'hengpiehuohenggou', 'hengxiegou', 'dian', 'shu', 'shuwan', 'shuwangou',
                      'shuzhezhegou', 'shuzhepiehuoshuzhezhe', 'shuti', 'shugou','error','zoutou']
        inp = bgr_to_rgb(unnormalize_img(batch['inp'][0], self.cfg.data.mean,
                                         self.cfg.data.std).permute(1, 2, 0))
        ex = output['py']
        ex = ex[-1] if isinstance(ex, list) else ex
        ex = ex.detach().cpu().numpy()
        waitForDel=[]
        for i in range(len(ex)):
            if i == len(ex)-1:
                break
            curleft=min(ex[i][:,0])
            curtop=min(ex[i][:,1])
            curright = max(ex[i][:, 0])
            curbottom = max(ex[i][:, 1])
            for j in range(i+1,len(ex)):
                if j ==len(ex)-1:
                    break
                tarleft = min(ex[j][:, 0])
                tartop = min(ex[j][:, 1])
                tarright = max(ex[j][:, 0])
                tarbottom = max(ex[j][:, 1])
                if IOU((curleft,curtop,curright,curbottom),(tarleft,tartop,tarright,tarbottom))>0.8:
                    waitForDel.append(j)
        ex=np.delete(ex,waitForDel,axis=0)
        det = output['detection']
        det = det.detach().cpu().numpy().tolist()
        colors = np.array([
            [31, 119, 180],
            [255, 127, 14],
            [46, 160, 44],
            [214, 40, 39],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [126, 126, 126],
            [188, 189, 32],
            [26, 190, 207]
        ]) / 255.
        np.random.shuffle(colors)
        colors = cycle(colors)

        for i in range(len(ex)):
            fig, ax = plt.subplots(1, figsize=(20, 10))
            fig.tight_layout()
            ax.axis('off')
            ax.imshow(inp)
            color = next(colors).tolist()
            poly = ex[i]
            # poly = np.append(poly, [poly[0]], axis=0)
            poly_final=[]

            for j in poly:
               poly_final.append([int(j[0]),int(j[1])])

            # result = np.zeros((inp.shape[0],inp.shape[1],3))
            result = bgr_to_rgb(unnormalize_img(batch['inp'][0], self.cfg.data.mean,
                                         self.cfg.data.std).permute(1, 2, 0))
            result = result.cpu().numpy()
            result = cv2.fillPoly(result, [np.array(poly_final)], color)
            result = result*255
            print(categories[int(det[i][3])])
            print(str(det[i][2]))
            cv2.imencode('.png',result)[1].tofile(save_dir+'/'+str(i)+categories[int(det[i][3])]+str(det[i][2])+'.png')
            plt.close(fig)
            # poly = np.append(poly, [poly[0]], axis=0)
            # ax.plot(poly[:, 0], poly[:, 1], color=color, lw=2)
            # if save_dir is not None:
            #     plt.savefig(fname=save_dir+'/'+str(i)+'.jpg', bbox_inches='tight')
            #     plt.close()
            # else:
            #     plt.show()

    def visualize(self, output, batch):
        # if args.output_dir != 'None':
        #     file_name = os.path.join(args.output_dir, batch['meta']['img_name'][0])
        # else:
        #     file_name = None

        self.visualize_ex(output, batch, save_dir=args.output_dir+'/'+batch['meta']['img_name'][0].split('.')[0])

def run_visualize(cfg):
    network = make_network.get_network(cfg).cuda()
    load_network(network, args.checkpoint,map_location='cuda:0')
    network.eval()

    data_loader = make_demo_loader(args.image_dir, cfg=cfg)
    visualizer = Visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        img=batch['inp'][0].detach().cpu().numpy()[-1]
        img=cv2.bitwise_not(np.array(img,dtype=np.uint8))
        contours,hiertry= cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        # 轮廓索引
        os.makedirs(args.output_dir,exist_ok=True)
        os.makedirs(args.output_dir + '/' + batch['meta']['img_name'][0].split('.')[0], exist_ok=True)
        for idx in range(len(contours)):
            if len(contours[idx])>80:
                img_color = np.zeros((img.shape[0],img.shape[1],3))
                tmp = cv2.drawContours(img_color,contours,idx,(15,192,150),cv2.FILLED)
                tmp[img[:,:]==0]=(0,0,0)
                # tmp[np.where((img[:,:]>0) & (tmp[:,:,0]==0))]=(255,255,255)
                cv2.imencode('.jpg',tmp)[1].tofile(args.output_dir+'/'+ batch['meta']['img_name'][0].split('.')[0]+'/'+str(idx)+'.jpg')


        with torch.no_grad():
            # for img in os.listdir(args.output_dir + '/' + batch['meta']['img_name'][0].split('.')[0]):
             output = network(batch['inp'], batch)

        if args.with_post_process:
            post_process.post_process(output)
        if args.with_nms:
            nms.post_process(output)
        visualizer.visualize(output, batch)

if __name__ == "__main__":
    cfg = get_cfg(args)
    torch.cuda.set_device(args.device)
    run_visualize(cfg)
