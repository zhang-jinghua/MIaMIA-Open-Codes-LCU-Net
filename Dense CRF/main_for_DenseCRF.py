import cv2 as cv
import os
from CRF import CRFs
original_file = "C:/Users/zjh/Desktop/11/imagesforexperiment/26/test/original_g"
predict_file = "C:/Users/zjh/Desktop/11/imagesforexperiment/26/Otsu_for_netresults/original_g/3"
save_file = "C:/Users/zjh/Desktop/11/imagesforexperiment/26/denseCRFresults/original_g/3"

def makedir(path):

    folder = os.path.exists(path)

    if folder is True:
        print("文件夹已存在")
    else:
        os.makedirs(path)
        print("文件夹创建中")
        print("完成")

original_list = os.listdir(original_file)
predict_list = os.listdir(predict_file)
makedir(save_file)
print(predict_list)
if len(original_list) == len(predict_list):
    print("已检验原始图片与深度学习网络图片一致！")
    num = len(original_list)
    for i in range(num):
        img = cv.imread(original_file+"/"+str(i)+".png")
        prd = cv.imread(predict_file+"/"+str(i)+".png")
        shape = prd.shape
        # print(shape)
        img1 = cv.resize(img,(shape[0],shape[1]))
        # prd[prd > 50] = 255
        # prd[prd <50] = 0
        # cv.imshow("0",prd)
        # cv.waitKey(0)
        crf_path = save_file+"/"+str(i)+".png"
        CRFs(img1,prd,crf_path)
else:
    print("图片不一致！请检查文件夹内图片是否对应")

