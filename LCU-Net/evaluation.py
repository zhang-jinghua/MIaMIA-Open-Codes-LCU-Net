import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
file_num = 8   #选择Random数
type= "random_original_move_dice" #调用评估图片类型
num = 5 #评估图片总数
x_num,y_num = 1,5 #总图的图片个数等于x_num和y_num之和，其中x为竖直方向窗口个数，y为水平方向窗口个数
def makedir(path):

    folder = os.path.exists(path)

    if folder is True:
        print("文件夹已存在")
    else:
        os.makedirs(path)
        print("文件夹创建中")
        print("完成")

def view(mat,q,n):
    name_list=["D","I","P","R","S","V","RV"]


    print(q,n)
    for i in range(num):
        num_list = mat[i]
        predict_path = "D:/U-Net data/Random"+str(file_num)+"/" + str(q) + "/test/predict/"+type+"/"+str(n) +"/"+ str(i) +"_predict.png"
        gt_path =  "D:/U-Net data/Random"+str(file_num)+"/" + str(q) + "/test/GTM/"+ str(i) +".png"


        ################################每个预测图片与GT对比图，及各项指标####################################
        save_fig_path = "D:/U-Net data/Random"+str(file_num)+"/" + str(q) + "/test/Evaluation/"+type+"/"+str(n)
        makedir(save_fig_path)
        fig = plt.figure(figsize=(10,3))
        img = cv.imread(predict_path)
        gt = cv.imread(gt_path)
        gt = cv.resize(gt,(256,256))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)

        ax1.imshow(gt)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title("GroundTruth")

        ax2.imshow(img)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("Predict")

        ax3 = fig.add_subplot(133)
        ax3.bar(range(len(num_list)),num_list,color = ['#4F94CD','#CAE1FF'])
        plt.xticks(range(len(num_list)), name_list)
        ax3.set_title("Evaluation")
   #     plt.show()
        plt.savefig(save_fig_path+"/"+str(i)+'_evaluation.jpg',dpi=300)


        with open(save_fig_path + '.txt', 'a', encoding='utf-8') as f:
            f.write("predict_image_"+ str(i) + "_index:" + str(num_list) + "\n" )
    with open(save_fig_path + '.txt', 'a', encoding='utf-8') as f:
        f.write("predict_image_average_index:" + str(sum(mat)/num) + "\n" )
    ##########################################################################################################


    ##############################################图片指标总图################################################
    num_list = mat[0] #为了获取指标个数
    color = ['#4F94CD','#CAE1FF']
    l = 15 * x_num #水平方向总图大小
    h = 1 * y_num #竖直方向总图大小
    fig1 = plt.figure(figsize=(l,h)) #总图大小

    for f in range(num):

        ax1 = fig1.add_subplot(x_num,y_num,f+1)
        ax1.bar(range(len(num_list)), mat[f],color = color)
        plt.xticks(range(len(num_list)), name_list)
        ax1.set_title(str(f)+"_index")

    # ax2 = fig1.add_subplot(252)
    # ax2.bar(range(len(num_list)), mat[1],color = color)
    # plt.xticks(range(len(num_list)), name_list)
    # ax2.set_title("1_index")
    #
    # ax3 = fig1.add_subplot(253)
    # ax3.bar(range(len(num_list)), mat[2],color = color)
    # plt.xticks(range(len(num_list)), name_list)
    # ax3.set_title("2_index")
    #
    # ax4 = fig1.add_subplot(254)
    # ax4.bar(range(len(num_list)), mat[3],color = color)
    # plt.xticks(range(len(num_list)), name_list)
    # ax4.set_title("3_index")
    #
    # ax5 = fig1.add_subplot(255)
    # ax5.bar(range(len(num_list)), mat[4],color = color)
    # plt.xticks(range(len(num_list)), name_list)
    # ax5.set_title("4_index")
    #
    # ax6 = fig1.add_subplot(256)
    # ax6.bar(range(len(num_list)), mat[5],color = color)
    # plt.xticks(range(len(num_list)), name_list)
    # ax6.set_title("5_index")
    #
    # ax7 = fig1.add_subplot(257)
    # ax7.bar(range(len(num_list)), mat[6],color = color)
    # plt.xticks(range(len(num_list)), name_list)
    # ax7.set_title("6_index")
    #
    # ax8 = fig1.add_subplot(258)
    # ax8.bar(range(len(num_list)), mat[7],color = color)
    # plt.xticks(range(len(num_list)), name_list)
    # ax8.set_title("7_index")
    #
    # ax9 = fig1.add_subplot(259)
    # ax9.bar(range(len(num_list)), mat[8],color = color)
    # plt.xticks(range(len(num_list)), name_list)
    # ax9.set_title("8_index")
    #
    # ax10 = fig1.add_subplot(2,5,10)
    # ax10.bar(range(len(num_list)), mat[9],color = color)
    # plt.xticks(range(len(num_list)), name_list)
    # ax10.set_title("9_index")
    plt.savefig(save_fig_path + "/total_evaluation.jpg", dpi=400)



    average_list =   sum(mat)/num
    fig2 = plt.figure()
    plt.bar(range(len(average_list)),average_list,color = color)
    plt.xticks(range(len(average_list)), name_list)
    plt.title("Average_Index")
    plt.savefig(save_fig_path + "/Average_Index.jpg", dpi=300)
  #  fig2.show()
   # cv.waitKey(0)



def Dice(SEG,GT):
    intersection = SEG & GT
    num_intersection = intersection.sum()
    num_sum = SEG.sum() + GT.sum()
    DICE = 2 * num_intersection / num_sum
    return DICE
def VOE(SEG,GT):
    mistake = SEG ^ GT
    num_mistake = mistake.sum()
    num_sum = SEG.sum() + GT.sum()
    voe = num_mistake/num_sum
    return voe
def RVD(SEG,GT):
    num_SEG = SEG.sum()
    num_GT = GT.sum()
    rvd = abs(num_SEG/num_GT - 1)
    return rvd
def IoU(SEG,GT):
    intersection = SEG & GT
    union = SEG|GT
    num_intersection = intersection.sum()
    num_union = union.sum()
    IOU = num_intersection/num_union
    return IOU
def Precision(SEG,GT):
    intersection = SEG & GT
    num_positive = SEG.sum()
    num_intersection = intersection.sum()
    precision = num_intersection/num_positive
    return precision
def Recall(SEG,GT):
    intersection = SEG & GT
    num_gt = GT.sum()
    num_intersection = intersection.sum()
    recall = num_intersection/num_gt
    return recall
def Specificity(SEG,GT,length,width):
    sum_totally = length*width
    union = SEG|GT
    num_union = union.sum()
    num_gt = GT.sum()
    num_gt_negative = sum_totally - num_gt
    num_seg_negative = sum_totally - num_union
    specificity = num_seg_negative/num_gt_negative
    return specificity

def evaluation(m_class, train_num):
    i = m_class  # 微生物类别，共0-20类
    n = train_num  # 第n次训练
    file_path = "D:/U-Net data/Random"+str(file_num)
    predict_path = file_path + "/" + str(i) + "/test/predict/"+type+"/"+str(n)
    GTM_path = file_path + "/" + str(i) + "/test/GTM/"
    image_num =len(os.listdir(predict_path))
    mat = np.zeros(shape=(num, 7))
    for i in range(image_num):
        SEG = cv.imread(predict_path +"/"+ str(i) +"_predict.png", cv.IMREAD_GRAYSCALE)
        GT = cv.imread(GTM_path + str(i) + ".png", cv.IMREAD_GRAYSCALE)
        GT = cv.resize(GT, (256, 256))
        SEG1 = SEG / 255
        GT1 = GT / 255

        SEG1[SEG1 > 0.5] = 1
        SEG1[SEG1 < 0.5] = 0
        GT1[GT1 > 0.5] = 1
        GT1[GT1 < 0.5] = 0
        SEG2 = SEG1.astype(np.int16)
        GT2 = GT1.astype(np.int16)


        D = Dice(SEG2, GT2)
        I = IoU(SEG2, GT2)
        P = Precision(SEG2, GT2)
        R = Recall(SEG2, GT2)
        S = Specificity(SEG2, GT2, 256, 256)
        V = VOE(SEG2, GT2)
        RV = RVD(SEG2, GT2)
        mat[i] = [D,I,P,R,S,V,RV]
    print(mat)
    view(mat,m_class,train_num)









