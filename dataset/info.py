
class DatasetInfo(object):
    dataset_info = {
        'coco_train': {
            'name': 'coco',
            'image_dir': 'data/coco/train2017',
            'anno_dir': 'data/coco/annotations/instances_train2017.json',
            'split': 'train'
        },
        'coco_val': {
            'name': 'coco',
            'image_dir': 'data/coco/val2017',
            'anno_dir': 'data/coco/annotations/instances_val2017.json',
            'split': 'val'
        },
        'coco_test': {
            'name': 'coco',
            'image_dir': 'data/coco/test2017',
            'anno_dir': 'data/coco/annotations/image_info_test-dev2017.json',
            'split': 'test'
        },
        'sbd_train': {
            'name': 'sbd',
            'image_dir': 'data/sbd/img',
            'anno_dir': 'data/sbd/annotations/sbd_train_instance.json',
            'split': 'train'
        },
        'sbd_val': {
            'name': 'sbd',
            'image_dir': 'data/sbd/img',
            'anno_dir': 'data/sbd/annotations/sbd_trainval_instance.json',
            'split': 'val'
        },
        'kitti_train': {
            'name': 'kitti',
            'image_dir': 'data/kitti/training/image_2', 
            'anno_dir': 'data/kitti/training/instances_train.json', 
            'split': 'train'
        }, 
        'kitti_val': {
            'name': 'kitti',
            'image_dir': 'data/kitti/testing/image_2', 
            'anno_dir': 'data/kitti/testing/instances_val.json', 
            'split': 'val'
        },
        'cityscapes_train': {
            'name': 'cityscapes',
            'image_dir': 'data/cityscapes/leftImg8bit',
            'anno_dir': ('data/cityscapes/annotations/train', 'data/cityscapes/annotations/train_val'),
            'split': 'train'
        },
        'cityscapes_val': {
            'name': 'cityscapes',
            'image_dir': 'data/cityscapes/leftImg8bit',
            'anno_dir': 'data/cityscapes/annotations/val',
            'split': 'val'
        },
        'cityscapesCoco_val': {
            'name': 'cityscapesCoco',
            'image_dir': 'data/cityscapes/leftImg8bit/val',
            'anno_dir': 'data/cityscapes/coco_ann/instance_val.json',
            'split': 'val'
        },
        'cityscapes_test': {
            'name': 'cityscapes',
            'image_dir': 'data/cityscapes/leftImg8bit/test', 
            'anno_dir': 'data/cityscapes/annotations/test', 
            'split': 'test'
        },
        'StrokeExtraction_train':{
            'name':'csscd',
            'image_dir': 'F:/DeepStroke-master/img_folder/train',
            # 'anno_dir': ('F:/DeepStroke-master/dataset/DATA_GB6763_LTH_train.json','F:/DeepStroke-master/dataset/DATA_GB6763_FZJTJW_train.json',
            #              'F:/DeepStroke-master/dataset/DATA_GB6763_FZLBJW_train.json','F:/DeepStroke-master/dataset/DATA_GB6763_HLJ_train.json','F:/DeepStroke-master/dataset/DATA_GB6763_SS_train.json'),
            'anno_dir': ('F:/DeepStroke-master/dataset/train/output.json'),
            'split': 'train'
        },
        'StrokeExtraction_val': {
            'name': 'csscd',
            'image_dir': 'F:/DeepStroke-master/img_folder/test',
            'anno_dir': 'F:/DeepStroke-master/dataset/DATA_GB6763_FZJTJW_test.json',
            'split': 'val'
        },
        'ccse_train': {
            'name': 'ccse',
            'image_dir': 'D:/CCSE-master/dataset/kaiti_chinese_stroke_2021/train2021',
            # 'anno_dir': ('F:/DeepStroke-master/dataset/DATA_GB6763_LTH_train.json','F:/DeepStroke-master/dataset/DATA_GB6763_FZJTJW_train.json',
            #              'F:/DeepStroke-master/dataset/DATA_GB6763_FZLBJW_train.json','F:/DeepStroke-master/dataset/DATA_GB6763_HLJ_train.json','F:/DeepStroke-master/dataset/DATA_GB6763_SS_train.json'),
            'anno_dir': ('D:/CCSE-master/dataset/kaiti_chinese_stroke_2021/annotations/instances_train2021.json'),
            'split': 'train'
        },
        'ccse_val': {
            'name': 'ccse',
            'image_dir': 'D:/CCSE-master/dataset/kaiti_chinese_stroke_2021/val2021',
            'anno_dir': 'D:/CCSE-master/dataset/kaiti_chinese_stroke_2021/annotations/instances_val2021.json',
            'split': 'val'
        },
        'ccse_test': {
            'name': 'ccse',
            'image_dir': 'D:/CCSE-master/dataset/kaiti_chinese_stroke_2021/test2021',
            'anno_dir': 'D:/CCSE-master/dataset/kaiti_chinese_stroke_2021/annotations/instances_test2021.json',
            'split': 'val'
        },
        'ccseHW_train': {
            'name': 'ccse',
            'image_dir': 'D:/CCSE-master/dataset/handwritten_chinese_stroke_2021/train2021',
            # 'anno_dir': ('F:/DeepStroke-master/dataset/DATA_GB6763_LTH_train.json','F:/DeepStroke-master/dataset/DATA_GB6763_FZJTJW_train.json',
            #              'F:/DeepStroke-master/dataset/DATA_GB6763_FZLBJW_train.json','F:/DeepStroke-master/dataset/DATA_GB6763_HLJ_train.json','F:/DeepStroke-master/dataset/DATA_GB6763_SS_train.json'),
            'anno_dir': ('D:/CCSE-master/dataset/handwritten_chinese_stroke_2021/annotations/instances_train2021.json'),
            'split': 'train'
        },
        'ccseHW_val': {
            'name': 'ccse',
            'image_dir': 'D:/CCSE-master/dataset/handwritten_chinese_stroke_2021/val2021',
            'anno_dir': 'D:/CCSE-master/dataset/handwritten_chinese_stroke_2021/annotations/instances_val2021.json',
            'split': 'val'
        },
        'ccseHW_test': {
            'name': 'ccse',
            'image_dir': 'D:/CCSE-master/dataset/handwritten_chinese_stroke_2021/test2021',
            'anno_dir': 'D:/CCSE-master/dataset/handwritten_chinese_stroke_2021/annotations/instances_test2021.json',
            'split': 'test'
        },
        'RHSEDB_train': {
            'name': 'RHSEDB',
            'image_dir': 'D:/StrokeExtraction-master/dataset/RHSEDB/train_Stroke',
            # 'anno_dir': ('F:/DeepStroke-master/dataset/DATA_GB6763_LTH_train.json','F:/DeepStroke-master/dataset/DATA_GB6763_FZJTJW_train.json',
            #              'F:/DeepStroke-master/dataset/DATA_GB6763_FZLBJW_train.json','F:/DeepStroke-master/dataset/DATA_GB6763_HLJ_train.json','F:/DeepStroke-master/dataset/DATA_GB6763_SS_train.json'),
            'anno_dir': ('D:/StrokeExtraction-master/dataset/RHSEDB/annotations/train.json'),
            'split': 'train'
        },
        'RHSEDB_val': {
            'name': 'RHSEDB',
            'image_dir': 'D:/StrokeExtraction-master/dataset/RHSEDB/test_Stroke',
            'anno_dir': 'D:/StrokeExtraction-master/dataset/RHSEDB/annotations/test.json',
            'split': 'val'
        },
        'RHSEDB_test': {
            'name': 'RHSEDB',
            'image_dir': 'D:/StrokeExtraction-master/dataset/RHSEDB/test_Stroke',
            'anno_dir': 'D:/StrokeExtraction-master/dataset/RHSEDB/annotations/test.json',
            'split': 'test'
        },
        'SCUT_Stroke_test': {
            'name': 'SCUT_Stroke',
            'image_dir': 'F:/U盘backup/Developer/书法笔画（已对齐 256）/曹全碑下册_曹全碑下册 第2页/易',
            'anno_dir': 'D:/StrokeExtraction-master/dataset/RHSEDB/annotations/test.json',
            'split': 'test'
        },
    }

