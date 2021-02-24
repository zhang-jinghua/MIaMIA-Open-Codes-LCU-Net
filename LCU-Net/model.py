from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import keras.backend as K
from keras import losses



def unet(pretrained_weights = None,input_size=(256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(16, (3,1),  padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1_1 = BatchNormalization()(conv1)
    conv1_2 = Activation('relu')(conv1_1)

    conv2 = Conv2D(16, (1,3), padding = 'same', kernel_initializer = 'he_normal')(conv1_2)
    conv2_1 = BatchNormalization()(conv2)
    conv2_2 = Activation('relu')(conv2_1)

    conv3 = Conv2D(16, (3,1), padding = 'same', kernel_initializer = 'he_normal')(conv2_2)
    conv3_1 = BatchNormalization()(conv3)
    conv3_2 = Activation('relu')(conv3_1)
    conv4 = Conv2D(16, (1,3), padding = 'same', kernel_initializer = 'he_normal')(conv3_2)
    conv4_1 = BatchNormalization()(conv4)
    conv4_2 = Activation('relu')(conv4_1)

    conv5 = Conv2D(16, (3,1), padding = 'same', kernel_initializer = 'he_normal')(conv4_2)
    conv5_1 = BatchNormalization()(conv5)
    conv5_2 = Activation('relu')(conv5_1)
    conv6 = Conv2D(16, (1,3), padding = 'same', kernel_initializer = 'he_normal')(conv5_2)
    conv6_1 = BatchNormalization()(conv6)
    conv6_2 = Activation('relu')(conv6_1)

    conv7 = Conv2D(16, (1,1), padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv7_1 = BatchNormalization()(conv7)
    conv7_2 = Activation('relu')(conv7_1)
    merge1 = concatenate([conv2_2, conv4_2,conv6_2,conv7_2], axis=3)   #102

    pool1 = MaxPooling2D(pool_size=(2, 2))(merge1)



    conv8 = Conv2D(32, (3,1),  padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv8_1 = BatchNormalization()(conv8)
    conv8_2 = Activation('relu')(conv8_1)
    conv9 = Conv2D(32, (1, 3), padding='same', kernel_initializer='he_normal')(conv8_2)
    conv9_1 = BatchNormalization()(conv9)
    conv9_2 = Activation('relu')(conv9_1)

    conv10 = Conv2D(32, (3,1),  padding = 'same', kernel_initializer = 'he_normal')(conv9_2)
    conv10_1 = BatchNormalization()(conv10)
    conv10_2 = Activation('relu')(conv10_1)
    conv11 = Conv2D(32, (1, 3), padding='same', kernel_initializer='he_normal')(conv10_2)
    conv11_1 = BatchNormalization()(conv11)
    conv11_2 = Activation('relu')(conv11_1)

    conv12 = Conv2D(32, (3,1),  padding = 'same', kernel_initializer = 'he_normal')(conv11_2)
    conv12_1 = BatchNormalization()(conv12)
    conv12_2 = Activation('relu')(conv12_1)
    conv13 = Conv2D(32, (1, 3), padding='same', kernel_initializer='he_normal')(conv12_2)
    conv13_1 = BatchNormalization()(conv13)
    conv13_2 = Activation('relu')(conv13_1)

    conv14 = Conv2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(pool1)
    conv14_1 = BatchNormalization()(conv14)
    conv14_2 = Activation('relu')(conv14_1)
    merge2 = concatenate([conv9_2, conv11_2, conv13_2, conv14_2], axis=3)#210

    pool2 = MaxPooling2D(pool_size=(2, 2))(merge2)



    conv15 = Conv2D(64, (3,1),  padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv15_1 = BatchNormalization()(conv15)
    conv15_2 = Activation('relu')(conv15_1)
    conv16 = Conv2D(64, (1, 3), padding='same', kernel_initializer='he_normal')(conv15_2)
    conv16_1 = BatchNormalization()(conv16)
    conv16_2 = Activation('relu')(conv16_1)

    conv17 = Conv2D(64, (3,1),  padding = 'same', kernel_initializer = 'he_normal')(conv16_2)
    conv17_1 = BatchNormalization()(conv17)
    conv17_2 = Activation('relu')(conv17_1)
    conv18 = Conv2D(64, (1, 3), padding='same', kernel_initializer='he_normal')(conv17_2)
    conv18_1 = BatchNormalization()(conv18)
    conv18_2 = Activation('relu')(conv18_1)

    conv19 = Conv2D(64, (3,1),  padding = 'same', kernel_initializer = 'he_normal')(conv18_2)
    conv19_1 = BatchNormalization()(conv19)
    conv19_2 = Activation('relu')(conv19_1)
    conv20 = Conv2D(64, (1, 3), padding='same', kernel_initializer='he_normal')(conv19_2)
    conv20_1 = BatchNormalization()(conv20)
    conv20_2 = Activation('relu')(conv20_1)

    conv21 = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(pool2)
    conv21_1 = BatchNormalization()(conv21)
    conv21_2 = Activation('relu')(conv21_1)
    merge3 = concatenate([conv16_2, conv18_2, conv20_2, conv21_2], axis=3)#424

    pool3 = MaxPooling2D(pool_size=(2, 2))(merge3)



    conv22 = Conv2D(128, (3,1),  padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv22_1 = BatchNormalization()(conv22)
    conv22_2 = Activation('relu')(conv22_1)
    conv23 = Conv2D(128, (1, 3), padding='same', kernel_initializer='he_normal')(conv22_2)
    conv23_1 = BatchNormalization()(conv23)
    conv23_2 = Activation('relu')(conv23_1)

    conv24 = Conv2D(128, (3,1),  padding = 'same', kernel_initializer = 'he_normal')(conv23_2)
    conv24_1 = BatchNormalization()(conv24)
    conv24_2 = Activation('relu')(conv24_1)
    conv25 = Conv2D(128, (1, 3), padding='same', kernel_initializer='he_normal')(conv24_2)
    conv25_1 = BatchNormalization()(conv25)
    conv25_2 = Activation('relu')(conv25_1)

    conv26 = Conv2D(128, (3,1),  padding = 'same', kernel_initializer = 'he_normal')(conv25_2)
    conv26_1 = BatchNormalization()(conv26)
    conv26_2 = Activation('relu')(conv26_1)
    conv27 = Conv2D(128, (1, 3), padding='same', kernel_initializer='he_normal')(conv26_2)
    conv27_1 = BatchNormalization()(conv27)
    conv27_2 = Activation('relu')(conv27_1)

    conv28 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal')(pool3)
    conv28_1 = BatchNormalization()(conv28)
    conv28_2 = Activation('relu')(conv28_1)
    merge4 = concatenate([conv23_2, conv25_2, conv27_2, conv28_2], axis=3)#852

    pool4 = MaxPooling2D(pool_size=(2, 2))(merge4)



    conv29 = Conv2D(256, (3,1),  padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv29_1 = BatchNormalization()(conv29)
    conv29_2 = Activation('relu')(conv29_1)
    conv30 = Conv2D(256, (1, 3), padding='same', kernel_initializer='he_normal')(conv29_2)
    conv30_1 = BatchNormalization()(conv30)
    conv30_2 = Activation('relu')(conv30_1)

    conv31 = Conv2D(256, (3,1),  padding = 'same', kernel_initializer = 'he_normal')(conv30_2)
    conv31_1 = BatchNormalization()(conv31)
    conv31_2 = Activation('relu')(conv31_1)
    conv32 = Conv2D(256, (1, 3), padding='same', kernel_initializer='he_normal')(conv31_2)
    conv32_1 = BatchNormalization()(conv32)
    conv32_2 = Activation('relu')(conv32_1)

    conv33 = Conv2D(256, (3,1),  padding = 'same', kernel_initializer = 'he_normal')(conv32_2)
    conv33_1 = BatchNormalization()(conv33)
    conv33_2 = Activation('relu')(conv33_1)
    conv34 = Conv2D(256, (1, 3), padding='same', kernel_initializer='he_normal')(conv33_2)
    conv34_1 = BatchNormalization()(conv34)
    conv34_2 = Activation('relu')(conv34_1)

    conv35 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal')(pool4)
    conv35_1 = BatchNormalization()(conv35)
    conv35_2 = Activation('relu')(conv35_1)
    merge5 = concatenate([conv30_2, conv32_2, conv34_2, conv35_2], axis=3)#1706



    up1 = Conv2DTranspose(1024, (2, 2), strides=2, padding='same', kernel_initializer='he_normal')(merge5)
    merge6 = concatenate([merge4,up1], axis=3)#1705

    conv36 = Conv2D(128, (3,1),  padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv36_1 = BatchNormalization()(conv36)
    conv36_2 = Activation('relu')(conv36_1)
    conv37 = Conv2D(128, (1,3),  padding = 'same', kernel_initializer = 'he_normal')(conv36_2)
    conv37_1 = BatchNormalization()(conv37)
    conv37_2 = Activation('relu')(conv37_1)

    conv38 = Conv2D(128, (3,1),  padding = 'same', kernel_initializer = 'he_normal')(conv37_2)
    conv38_1 = BatchNormalization()(conv38)
    conv38_2 = Activation('relu')(conv38_1)
    conv39 = Conv2D(128, (1,3),  padding = 'same', kernel_initializer = 'he_normal')(conv38_2)
    conv39_1 = BatchNormalization()(conv39)
    conv39_2 = Activation('relu')(conv39_1)

    conv40 = Conv2D(128, (3,1),  padding = 'same', kernel_initializer = 'he_normal')(conv39_2)
    conv40_1 = BatchNormalization()(conv40)
    conv40_2 = Activation('relu')(conv40_1)
    conv41 = Conv2D(128, (1,3),  padding = 'same', kernel_initializer = 'he_normal')(conv40_2)
    conv41_1 = BatchNormalization()(conv41)
    conv41_2 = Activation('relu')(conv41_1)

    conv42 = Conv2D(128, (1,1),  padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv42_1 = BatchNormalization()(conv42)
    conv42_2 = Activation('relu')(conv42_1)
    merge7 = concatenate([conv37_2,conv39_2,conv41_2,conv42_2], axis=3)#852



    up2 = Conv2DTranspose(512, (2, 2), strides=2, padding='same', kernel_initializer='he_normal')(merge7)
    merge8 = concatenate([merge3, up2], axis=3)#850

    conv43 = Conv2D(64, (3, 1), padding='same', kernel_initializer='he_normal')(merge8)
    conv43_1 = BatchNormalization()(conv43)
    conv43_2 = Activation('relu')(conv43_1)
    conv44 = Conv2D(64, (1, 3), padding='same', kernel_initializer='he_normal')(conv43_2)
    conv44_1 = BatchNormalization()(conv44)
    conv44_2 = Activation('relu')(conv44_1)

    conv45 = Conv2D(64, (3, 1), padding='same', kernel_initializer='he_normal')(conv44_2)
    conv45_1 = BatchNormalization()(conv45)
    conv45_2 = Activation('relu')(conv45_1)
    conv46 = Conv2D(64, (1, 3), padding='same', kernel_initializer='he_normal')(conv45_2)
    conv46_1 = BatchNormalization()(conv46)
    conv46_2 = Activation('relu')(conv46_1)

    conv47 = Conv2D(64, (3, 1), padding='same', kernel_initializer='he_normal')(conv46_2)
    conv47_1 = BatchNormalization()(conv47)
    conv47_2 = Activation('relu')(conv47_1)
    conv48 = Conv2D(64, (1, 3), padding='same', kernel_initializer='he_normal')(conv47_2)
    conv48_1 = BatchNormalization()(conv48)
    conv48_2 = Activation('relu')(conv48_1)

    conv49 = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(merge8)
    conv49_1 = BatchNormalization()(conv49)
    conv49_2 = Activation('relu')(conv49_1)
    merge9 = concatenate([conv44_2, conv46_2, conv48_2, conv49_2], axis=3)#424



    up3 = Conv2DTranspose(256, (2, 2), strides=2, padding='same', kernel_initializer='he_normal')(merge9)
    merge10 = concatenate([merge2, up3], axis=3)

    conv50 = Conv2D(32, (3, 1), padding='same', kernel_initializer='he_normal')(merge10)
    conv50_1 = BatchNormalization()(conv50)
    conv50_2 = Activation('relu')(conv50_1)
    conv51 = Conv2D(32, (1, 3), padding='same', kernel_initializer='he_normal')(conv50_2)
    conv51_1 = BatchNormalization()(conv51)
    conv51_2 = Activation('relu')(conv51_1)

    conv52 = Conv2D(32, (3, 1), padding='same', kernel_initializer='he_normal')(conv51_2)
    conv52_1 = BatchNormalization()(conv52)
    conv52_2 = Activation('relu')(conv52_1)
    conv53 = Conv2D(32, (1, 3), padding='same', kernel_initializer='he_normal')(conv52_2)
    conv53_1 = BatchNormalization()(conv53)
    conv53_2 = Activation('relu')(conv53_1)

    conv54 = Conv2D(32, (3, 1), padding='same', kernel_initializer='he_normal')(conv53_2)
    conv54_1 = BatchNormalization()(conv54)
    conv54_2 = Activation('relu')(conv54_1)
    conv55 = Conv2D(32, (1, 3), padding='same', kernel_initializer='he_normal')(conv54_2)
    conv55_1 = BatchNormalization()(conv55)
    conv55_2 = Activation('relu')(conv55_1)

    conv56 = Conv2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(merge10)
    conv56_1 = BatchNormalization()(conv56)
    conv56_2 = Activation('relu')(conv56_1)
    merge11 = concatenate([conv51_2, conv53_2, conv55_2, conv56_2], axis=3)



    up4 = Conv2DTranspose(128, (2, 2), strides=2, padding='same', kernel_initializer='he_normal')(merge11)
    merge12 = concatenate([merge1, up4], axis=3)

    conv57 = Conv2D(16, (3, 1), padding='same', kernel_initializer='he_normal')(merge12)
    conv57_1 = BatchNormalization()(conv57)
    conv57_2 = Activation('relu')(conv57_1)
    conv58 = Conv2D(16, (1, 3), padding='same', kernel_initializer='he_normal')(conv57_2)
    conv58_1 = BatchNormalization()(conv58)
    conv58_2 = Activation('relu')(conv58_1)

    conv59 = Conv2D(16, (3, 1), padding='same', kernel_initializer='he_normal')(conv58_2)
    conv59_1 = BatchNormalization()(conv59)
    conv59_2 = Activation('relu')(conv59_1)
    conv60 = Conv2D(16, (1, 3), padding='same', kernel_initializer='he_normal')(conv59_2)
    conv60_1 = BatchNormalization()(conv60)
    conv60_2 = Activation('relu')(conv60_1)

    conv61 = Conv2D(16, (3, 1), padding='same', kernel_initializer='he_normal')(conv60_2)
    conv61_1 = BatchNormalization()(conv61)
    conv61_2 = Activation('relu')(conv61_1)
    conv62 = Conv2D(16, (1, 3), padding='same', kernel_initializer='he_normal')(conv61_2)
    conv62_1 = BatchNormalization()(conv62)
    conv62_2 = Activation('relu')(conv62_1)

    conv63 = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal')(merge12)
    conv63_1 = BatchNormalization()(conv63)
    conv63_2 = Activation('relu')(conv63_1)
    merge13 = concatenate([conv58_2, conv60_2, conv62_2, conv63_2], axis=3)

    conv64 = Conv2D(1, 1, activation='sigmoid')(merge13)

    model = Model(input=inputs, output=conv64)

    # sgd = SGD(lr=1e-4 , momentum=0.9,decay=1e-6)

    # model.compile(optimizer = sgd, loss = 'dice', metrics = ['accuracy'])#SGD优化器
    # model.compile(optimizer=Adam(lr=0.0003), loss='dice', metrics=['accuracy'])#dice
    model.compile(optimizer=Adam(lr=0.00015), loss='binary_crossentropy', metrics=['accuracy'])  # 交叉熵
    # model.compile(optimizer=Adam(lr=0.00005), loss='dice', metrics=['accuracy'])
    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model