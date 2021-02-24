from model import*
from data import*
import os
import matplotlib.pyplot as plt
import math
import time
def training_vis(history,save_fig_path):
    loss = history.history['loss']
    print("loss:",loss)
    acc = history.history['acc']
    print("accuracy:",acc)
    val_loss = history.history['val_loss']
    print("validation_loss:",val_loss)
    val_acc = history.history['val_acc']
    print("validation_accuracy:",val_acc)
    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(121)
    ax1.plot(loss,label =  "train_loss")
    ax1.plot(val_loss,label = "validation_loss")
    ax1.set_title("train model loss")
    ax1.set_ylabel("loss")
    ax1.set_xlabel("epoch")
    plt.legend(["train","validation"], loc="upper right")

    ax2 = fig.add_subplot(122)
    ax2.plot(acc,label =  "train_accuracy")
    ax2.plot(val_acc,label = "validation_accuracy")
    ax2.set_title("train model accuracy")
    ax2.set_ylabel("accuracy")
    ax2.set_xlabel("epoch")
    plt.legend(["train","validation"], loc="upper right")


    plt.savefig(save_fig_path)

    #plt.show()
    #plt.close(fig)
    with open(save_fig_path+'.txt','a', encoding='utf-8') as f:

        f.write("loss:"+str(loss)+"\n"+"accuracy:"+str(acc)+"\n"+"validation_loss:"+str(val_loss)+"\n"+"validation_accuracy:"+str(val_acc))



def makedir(path):

    folder = os.path.exists(path)

    if folder is True:
        print("文件夹已存在")
    else:
        os.makedirs(path)
        print("文件夹创建中")
        print("完成")

aug_dict = dict(rotation_range=0.02,   #
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=0.05,
                zoom_range=0.05,
                horizontal_flip=True,
                fill_mode='nearest')
image_size = (256, 256, 1)
data_size = (256, 256)
file_path = "D:/U5510/Random1"
train = True #是否训练
predict = True #是否预测

def main(m_class, train_num):
    start = time.clock()
    i = m_class  # 微生物类别，共0-20类
    n = train_num  # 第n次训练
    type = "original_g"
    test_num = 210  # 测试图片数量
    BATCH_SIZE = 2




    #epochs = 10  # 迭代次数
    #steps_per_epoch = 300  # 迭代步长
    if train:
        # Aug_originall = file_path + "/" + str(i) + "/aug/original" + str(n)
        # Aug_GTM1 = file_path + "/" + str(i) + "/aug/GTM_" + str(n)
        # Aug_original2 = file_path + "/" + str(i) + "/aug/val_original_" + str(n)
        # Aug_GTM2 = file_path + "/" + str(i) + "/aug/val_GTM_" + str(n)
        # makedir(Aug_originall)
        # makedir(Aug_GTM1)
        # makedir(Aug_original2)
        # makedir(Aug_GTM2)
        Aug_originall = None
        Aug_GTM1 = None
        Aug_original2 = None
        Aug_GTM2 = None

        train_path = file_path + "/" + str(i) + "/train"
        test_path = file_path + "/" + str(i) + "/test"
        val_path = file_path + "/" + str(i) + "/val"






        myGene = trainGenerator(BATCH_SIZE, train_path, type, "GTM", aug_dict, target_size=data_size,
                                aug_image_save_dir=Aug_originall, aug_mask_save_dir=Aug_GTM1)
        valGene = validationGenerator(BATCH_SIZE, val_path, type, "GTM", aug_dict, target_size=data_size,
                                aug_image_save_dir=Aug_original2, aug_mask_save_dir=Aug_GTM2)



        # num_train_samples = sum([len(files) for r, d, files in os.walk(Aug_originall)]) #原始训练图片数目
        # num_valid_samples = sum([len(files) for r, d, files in os.walk(Aug_original2)]) #原始测试图片数目
        # print(num_train_samples,num_valid_samples)
        #
        # num_train_steps = math.floor(num_train_samples / BATCH_SIZE)  #最大整除数
        # num_valid_steps = math.floor(num_valid_samples / BATCH_SIZE)
        # print(num_train_steps,num_valid_steps)



        model = unet()
        model_path = test_path + "/model/"+type+"_binception4_13_layersame"
        makedir(model_path)
        model_checkpoint = ModelCheckpoint(model_path + "/unet_membrane" + str(n) + ".hdf5", monitor='loss',mode = 'min',verbose=1,
                                           save_best_only=True)
        history = model.fit_generator(myGene, steps_per_epoch = 300,epochs=50, validation_data=valGene,validation_steps=300,callbacks=[model_checkpoint])
        training_vis(history,test_path + "/model/"+type+"_binception4_13_layersame/loss&acc" + str(n) + ".jpg")
        end1 = time.clock()


    if predict:
        model_name = model_path + "/unet_membrane" + str(n) + ".hdf5"
        model = load_model(model_name)
        testGene = testGenerator(test_path + "/"+type, num_image=test_num, target_size=data_size)
        results = model.predict_generator(testGene, test_num, verbose=1)
        predict_save_path = test_path + "/predict/"+ type+"_binception4_13_layersame/" + str(n)
        makedir(predict_save_path)
        saveResult(predict_save_path, results)
        end2 =time.clock()
    print("training_time:",end1-start)
    print("test_time:", end2 - end1)
