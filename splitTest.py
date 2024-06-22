import json
import os
import shutil

import cv2
import numpy as np

# os.makedirs('F:/DeepStroke-master/img_folder/test')
# for i in os.listdir('F:/DeepStroke-master/CCSSD/DATA_GB6763_hand/JPEGImages'):
#     img=cv2.imdecode(np.fromfile('F:/DeepStroke-master/CCSSD/DATA_GB6763_hand/JPEGImages/'+i,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
#     img=cv2.bitwise_not(img)
#     cv2.imencode('.jpg',img)[1].tofile('F:/DeepStroke-master/CCSSD/DATA_GB6763_hand/JPEGImages/'+i)

    # if i.endswith('test.json'):
    #     data=json.load(open('F:/DeepStroke-master/dataset/'+i))
    #     for j in data['images']:
    #         shutil.move('F:/DeepStroke-master/img_folder/'+j['file_name'],'F:/DeepStroke-master/img_folder/test/'+j['file_name'])
    #         # print(j['file_name'])
        # print(data['images'])
# ll=[{"supercategory": "zhebi", "id": 1, "name": "wangou"}, {"supercategory": "pingbi", "id": 2, "name": "na"}, {"supercategory": "pingbi", "id": 3, "name": "ti"}, {"supercategory": "pingbi", "id": 4, "name": "pie"}, {"supercategory": "zhebi", "id": 5, "name": "piezhe"}, {"supercategory": "zhebi", "id": 6, "name": "piedian"}, {"supercategory": "zhebi", "id": 7, "name": "xiegouhuowogou"}, {"supercategory": "pingbi", "id": 8, "name": "heng"}, {"supercategory": "zhebi", "id": 9, "name": "hengzhe"}, {"supercategory": "zhebi", "id": 10, "name": "hengzhezhehuohengzhewan"}, {"supercategory": "zhebi", "id": 11, "name": "hengzhezhezhe"}, {"supercategory": "zhebi", "id": 12, "name": "hengzhezhezhegouhuohengpiewangou"}, {"supercategory": "zhebi", "id": 13, "name": "hengzhezhepie"}, {"supercategory": "zhebi", "id": 14, "name": "hengzheti"}, {"supercategory": "zhebi", "id": 15, "name": "hengzhegou"}, {"supercategory": "zhebi", "id": 16, "name": "hengpiehuohenggou"}, {"supercategory": "zhebi", "id": 17, "name": "hengxiegou"}, {"supercategory": "pingbi", "id": 18, "name": "dian"}, {"supercategory": "pingbi", "id": 19, "name": "shu"}, {"supercategory": "zhebi", "id": 20, "name": "shuwan"}, {"supercategory": "zhebi", "id": 21, "name": "shuwangou"}, {"supercategory": "zhebi", "id": 22, "name": "shuzhezhegou"}, {"supercategory": "zhebi", "id": 23, "name": "shuzhepiehuoshuzhezhe"}, {"supercategory": "zhebi", "id": 24, "name": "shuti"}, {"supercategory": "pingbi", "id": 25, "name": "shugou"}]
# res=[]
# for u in ll:
#     res.append(u['name'])
# print(res)
for i in os.listdir('F:/e2ec-main/results/DATA_GB6763_HLJ海/SubFig'):
    img=cv2.imdecode(np.fromfile('F:/e2ec-main/results/DATA_GB6763_HLJ海/SubFig/'+i,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,img=cv2.threshold(img,0,255,cv2.THRESH_OTSU)
    cv2.imencode('..jpg',img)[1].tofile('F:/e2ec-main/results/DATA_GB6763_HLJ海/SubFig/'+i)
    # cv2.waitKey()
