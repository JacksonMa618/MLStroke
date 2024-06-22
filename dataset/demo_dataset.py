import os
import cv2
import numpy as np

from .train.utils import augment
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, data_root, cfg):
        super(Dataset, self).__init__()
        self.data_root = data_root
        self.split = 'test'
        self.cfg = cfg
        self.image_names, self.paths = self.process_info(data_root)

    def process_info(self, data_root):
        image_name = []
        for i in os.listdir(data_root):
            if i.endswith('jpg') or i.endswith('png'):
                image_name.append(i)
        return image_name, [os.path.join(data_root, name) for name in image_name if name.endswith('jpg') or name.endswith('png')]

    def read_original_data(self, path):
        img = cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
        return img

    def __getitem__(self, index):
        img_name, img_path = self.image_names[index], self.paths[index]

        img = self.read_original_data(img_path)

        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            augment(
                img, self.split,
                self.cfg.data.data_rng, self.cfg.data.eig_val, self.cfg.data.eig_vec,
                self.cfg.data.mean, self.cfg.data.std, self.cfg.commen.down_ratio,
                self.cfg.data.input_h, self.cfg.data.input_w, self.cfg.data.scale_range,
                self.cfg.data.scale, self.cfg.test.test_rescale, self.cfg.data.test_scale
            )

        ret = {'inp': inp}

        meta = {'center': center, 'scale': scale, 'test': '', 'img_name': img_name}
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.image_names)