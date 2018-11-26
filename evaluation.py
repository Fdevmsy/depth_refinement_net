from __future__ import print_function, division
from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.misc
import os
import time
from keras.utils import plot_model
import pandas as pd 
from sklearn.metrics import mean_squared_error
from keras.models import model_from_json
from keras.models import load_model

class depthRefinementNet():
    def __init__(self):
        self.img_rows = 480
        self.img_cols = 640

        self.GT_rows = 480
        self.GT_cols = 640
        
        self.input_depth_rows = 128
        self.input_depth_cols = 160

        self.resize_depth_rows = 120
        self.resize_depth_cols = 160
        
        self.output_rows = 480
        self.output_cols = 640
        
        self.img_channels = 3
        self.depth_channels = 1

        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
        self.input_depth_shape = (self.input_depth_rows, self.input_depth_cols, self.depth_channels)
        self.GT_shape = (self.GT_rows, self.GT_cols, self.depth_channels)

        self.resize_depth_shape = (self.resize_depth_rows, self.resize_depth_cols, self.depth_channels)
        self.output_shape = (self.output_rows, self.output_cols, self.depth_channels)

        self.TOY = 0

        optimizer = Adam(0.0002, 0.5)

        try: 
            self.discriminator = load_model('refine_models/discriminator.h5')
            self.generator = load_model('refine_models/generator.h5')
            # self.discriminator.summary()
            # self.generator.summary()
            print("Loaded checkpoints")
        except:
            self.generator = self.build_generator()
            self.discriminator = self.build_discriminator() 
            print("No checkpoints found")   
            # self.generator.summary()
        # Build the generator
        # self.generator = self.build_generator()

        # Build and compile the discriminator
        # self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        plot_model(self.generator, to_file='generator.png', show_shapes=True, )
        plot_model(self.discriminator, to_file='discriminator.png', show_shapes=True, )
        # The generator takes noise as input and generates the missing
        # part of the image
        input_depth = Input(shape=self.resize_depth_shape)
        input_img = Input(shape=self.img_shape)
        
        out_super_image = self.generator([input_img, input_depth])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines
        # if it is generated or if it is a real image
        valid = self.discriminator(out_super_image)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([input_img, input_depth] , [out_super_image, valid])
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer)
    
    def generate_dataset(self, img_path, dep_path, GT_path, input_dep_value_path):
        ## rgb data
        img_List = [img_path + f for f in os.listdir(img_path) if f.endswith('.jpg')]
        img_List.sort()
        ## select test set 
        img_List = img_List[int(0.8*len(img_List)):]
        img_data = list() 
        print('loading images')
        if self.TOY == 1:
            img_List = img_List[0:10]
        else:
            pass
        for img in img_List:
            cv_image = cv2.imread(img, cv2.IMREAD_COLOR)
            # cv_image = cv2.resize(cv_image, (self.resized_cols, self.resized_rows), interpolation=cv2.INTER_LANCZOS4)
            cv_image = np.array(cv_image).astype('float32')  
            img_data.append(cv_image)
        img_data = np.array(img_data)
        
        ## input depth image 
        dep_List = [dep_path + f for f in os.listdir(dep_path) if f.endswith('.jpg')]
        dep_List.sort()
        dep_List = dep_List[int(0.8*len(dep_List)):]
        if self.TOY == 1:
            dep_List = dep_List[0:10]
        else:
            pass
        # print(dep_List[:5])
        resized_dep_data = list()
        input_dep_data = list() 
        print('loading depth')
        for dep in dep_List:
            cv_dep = cv2.imread(dep, 0)
            resized_cv_dep = cv2.resize(cv_dep, (self.resize_depth_cols, self.resize_depth_rows), interpolation=cv2.INTER_LANCZOS4)
            resized_cv_dep = np.array(resized_cv_dep[..., np.newaxis]).astype('float32') 

            # cv_dep = cv2.resize(cv_dep, (self.img_cols, self.img_rows), interpolation=cv2.INTER_LANCZOS4)
            cv_dep = np.array(cv_dep[..., np.newaxis]).astype('float32')   
            resized_dep_data.append(resized_cv_dep)
            input_dep_data.append(cv_dep)
        
        resized_dep_data = np.array(resized_dep_data)
        input_dep_data = np.array(input_dep_data)

        GT_data = list() 

        ## GT data
        GT_List = [GT_path + f for f in os.listdir(GT_path) if f.endswith('.txt')]
        GT_List.sort()
        GT_List = GT_List[int(0.8*len(GT_List)):]
        # print(img_List[:5])
        print('loading GT')

        if self.TOY == 1:
            GT_List = GT_List[0:10]
        else:
            pass 
        for GT in GT_List:
            # if GT != '/home/ferrycake/depth_refinement/Pix2Depth/dep_value/dep_value_64.txt':
                # print(GT)
                df = pd.read_csv(GT, sep=' ', header=None)
                # print(df.values.shape)
                GT_values = df.values
                try:
                    testt = np.array(GT_values[..., np.newaxis]).astype('float32')  
                except:
                    df = pd.read_csv(GT, sep=' ', header=None)
                # print(df.values.shape)
                    GT_values = df.values
                GT_values = np.array(GT_values[..., np.newaxis]).astype('float32') 
            #     GT_image = cv2.imread(GT, 0)
            #     GT_image = cv2.resize(GT_image, (self.img_cols, self.img_rows), interpolation=cv2.INTER_LANCZOS4)
            #     # GT_image = scipy.misc.imresize(GT_image, [self.img_rows, self.img_cols], interp='lanczos')
            #     GT_image = np.array(GT_image[..., np.newaxis]).astype('float32')  
                GT_data.append(GT_values)
        GT_data = np.array(GT_data)        

        # input depth value
        dep_value_data = list() 

        dep_value_List = [input_dep_value_path + f for f in os.listdir(input_dep_value_path) if f.endswith('.txt')]
        dep_value_List.sort()
        dep_value_List = dep_value_List[int(0.8*len(dep_value_List)):]
        # print(img_List[:5])
        print('loading input_depth_value')

        if self.TOY == 1:
            dep_value_List = dep_value_List[0:10]
        else:
            pass 
        for dep_v in dep_value_List:
            # if GT != '/home/ferrycake/depth_refinement/Pix2Depth/dep_value/dep_value_64.txt':
                # print(GT)
                df = pd.read_csv(dep_v, sep=' ', header=None)
                # print(df.values.shape)
                dep_values = df.values
                try:
                    testt = np.array(dep_values[..., np.newaxis]).astype('float32')  
                except:
                    df = pd.read_csv(dep_v, sep=' ', header=None)
                # print(df.values.shape)
                    dep_values = df.values
                dep_values = np.array(dep_values[..., np.newaxis]).astype('float32') 
            #     GT_image = cv2.imread(GT, 0)
            #     GT_image = cv2.resize(GT_image, (self.img_cols, self.img_rows), interpolation=cv2.INTER_LANCZOS4)
            #     # GT_image = scipy.misc.imresize(GT_image, [self.img_rows, self.img_cols], interp='lanczos')
            #     GT_image = np.array(GT_image[..., np.newaxis]).astype('float32')  
                dep_value_data.append(dep_values)
        dep_value_data = np.array(dep_value_data) 

        return img_data, input_dep_data, resized_dep_data, GT_data, dep_value_data

    def test(self, img_path, dep_path, GT_path, input_dep_value_depth, epochs, batch_size=16, sample_interval=50):

        img_data, input_dep_data, resized_dep_data, GT_data, input_dep_value_data = self.generate_dataset(img_path, dep_path, GT_path, input_dep_value_depth)
        

        X_vaild_img = img_data # to network
        X_vaild_dep = resized_dep_data # to network
        y_vaild_GT = GT_data # ground truth
        X_input_valid_dep_value = input_dep_value_data # fcrn's prediction 
        # X_vaild_dep = resized_dep_data[int(0.8*resized_dep_data.shape[0]):]
        self.evaluation(X_vaild_img, X_vaild_dep, X_input_valid_dep_value, y_vaild_GT)


    def evaluation(self, X_vaild_img, X_vaild_dep, X_input_valid_dep_value, y_vaild_GT):

        # print(X_vaild_img.shape, X_origin_valid_dep.shape, y_vaild_GT.shape)
        # original_mse = ((A - B)**2).mean(axis=None)
        resized_origin_valid_depth = list() # resize 160, 128 back to 640, 480
        for i in X_input_valid_dep_value:
            resized_depth = cv2.resize(i[:,:,0], (self.img_cols, self.img_rows), interpolation=cv2.INTER_LANCZOS4)
            resized_depth = np.array(resized_depth[..., np.newaxis]).astype('float32') 
            resized_origin_valid_depth.append(resized_depth)
            # print(resized_gen)
        resized_origin_valid_depth = np.array(resized_origin_valid_depth)
        # print(resized_origin_valid_depth.shape)
        # print(resized_origin_valid_depth.shape)
        original_mse = 0
        for i in range(y_vaild_GT.shape[0]):
            original_mse += np.sqrt(mean_squared_error(resized_origin_valid_depth[i,:,:,0], y_vaild_GT[i,:,:,0]))
        original_mse = original_mse/y_vaild_GT.shape[0]

        pre_mse = 0
        gen_depth = list()
        for i in range(X_vaild_dep.shape[0]):

            gen_depth.append(self.generator.predict([X_vaild_img[i:i+1], X_vaild_dep[i:i+1]])[0]) 
        gen_depth = np.array(gen_depth)
        # print(gen_depth.shape)
        for i in range(y_vaild_GT.shape[0]):
            pre_mse += np.sqrt(mean_squared_error(gen_depth[i,:,:,0], y_vaild_GT[i,:,:,0]))
        pre_mse = pre_mse/y_vaild_GT.shape[0]
        original_mse = ((resized_origin_valid_depth[0] - y_vaild_GT[0])**2).mean(axis=None)
        # print(original_mse)
        print('original rmse: %f, pred rmse: %f' %(original_mse,pre_mse))

        for i in range(y_vaild_GT.shape[0]):
            my_pred = gen_depth[i,:,:,0]
            FCRN_pred = resized_origin_valid_depth[i,:,:,0]
            GT = y_vaild_GT[i,:,:,0]
            my_pred = cv2.normalize(my_pred, None, 0, 255, cv2.NORM_MINMAX)
            FCRN_pred = cv2.normalize(FCRN_pred, None, 0, 255, cv2.NORM_MINMAX)
            GT = cv2.normalize(GT, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite('evaluation_results/my_pred_' + str(i) + '.jpg', my_pred)
            cv2.imwrite('evaluation_results/FCRN_pred_' + str(i) + '.jpg', FCRN_pred)
            cv2.imwrite('evaluation_results/GT_' + str(i) + '.jpg', GT)
        ## same photo ?
        # A = resized_origin_valid_depth[0,:,:,0]
        # B = y_vaild_GT[0,:,:,0]
        # A = cv2.normalize(A, None, 0, 255, cv2.NORM_MINMAX)
        # B = cv2.normalize(B, None, 0, 255, cv2.NORM_MINMAX)
        # cv2.imwrite('A.jpg', A)
        # cv2.imwrite('B.jpg', B)


if __name__ == '__main__':
    img_path = '/home/ferrycake/depth_refinement/Pix2Depth/imgs/'
    dep_path = '/home/ferrycake/depth_refinement/FCRN/tensorflow/pre_depth_image/'
    input_dep_value_depth = '/home/ferrycake/depth_refinement/FCRN/tensorflow/pre_depth_value/'
    GT_path = '/home/ferrycake/depth_refinement/Pix2Depth/dep_value/'
    DRN = depthRefinementNet()
    # img_data, input_dep_data, resized_dep_data, GT_data = DRN.generate_dataset(img_path, dep_path, GT_path)
    # print(img_data.shape, input_dep_data.shape, resized_dep_data.shape, GT_data.shape)
    DRN.test(img_path, dep_path, GT_path, input_dep_value_depth, 30000, batch_size=8, sample_interval=20)