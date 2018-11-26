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
from keras.models import model_from_json
from keras.models import load_model
from keras.utils import plot_model
import pandas as pd 
from sklearn.metrics import mean_squared_error

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

    def build_generator(self):

        input_depth = Input(shape=self.resize_depth_shape)
        input_img = Input(shape=self.img_shape) # 640

        # encoder # 320
        e_conv1 = Conv2D(64, kernel_size=3, strides=2, padding="same")(input_img)
        e_conv1 = LeakyReLU(alpha=0.2)(e_conv1)
        e_conv1 = BatchNormalization(momentum=0.8)(e_conv1)
        # 160
        e_conv2 = Conv2D(128, kernel_size=3, strides=2, padding="same")(e_conv1)
        e_conv2 = LeakyReLU(alpha=0.2)(e_conv2)
        e_conv2 = BatchNormalization(momentum=0.8)(e_conv2)

        contact_img = concatenate([e_conv2, input_depth])

        e_conv3 = Conv2D(256, kernel_size=3, strides=2, padding="same")(contact_img)
        e_conv3 = LeakyReLU(alpha=0.2)(e_conv3)
        e_conv3 = BatchNormalization(momentum=0.8)(e_conv3)    

        e_conv4 = Conv2D(512, kernel_size=1, strides=2, padding="same")(e_conv3)
        e_conv4 = LeakyReLU(alpha=0.2)(e_conv4)
        e_conv4 = Dropout(0.5)(e_conv4)  

        # decoder

        de_conv1 = UpSampling2D()(e_conv4)
        de_conv1 = Conv2D(256, kernel_size=3, padding="same")(de_conv1)
        de_conv1 = Activation('relu')(de_conv1)
        de_conv1 = BatchNormalization(momentum=0.8)(de_conv1)

        de_conv2 = UpSampling2D()(de_conv1)
        de_conv2 = Conv2D(128, kernel_size=3, padding="same")(de_conv2)
        de_conv2 = Activation('relu')(de_conv2)
        de_conv2 = BatchNormalization(momentum=0.8)(de_conv2)

        de_conv3 = UpSampling2D()(de_conv2)
        de_conv3 = Conv2D(64, kernel_size=3, padding="same")(de_conv3)
        de_conv3 = Activation('relu')(de_conv3)
        de_conv3 = BatchNormalization(momentum=0.8)(de_conv3)

        de_conv4 = UpSampling2D()(de_conv3)
        de_conv4 = Conv2D(32, kernel_size=3, padding="same")(de_conv4)
        de_conv4 = Activation('relu')(de_conv4)
        de_conv4 = BatchNormalization(momentum=0.8)(de_conv4)   

        # de_conv5 = UpSampling2D()(de_conv4)
        # de_conv5 = Conv2D(16, kernel_size=3, padding="same")(de_conv5)
        # de_conv5 = Activation('relu')(de_conv5)
        # de_conv5 = BatchNormalization(momentum=0.8)(de_conv5)                

        # de_conv4 = UpSampling2D()(de_conv3)
        de_conv6 = Conv2D(1, kernel_size=3, padding="same")(de_conv4)
        de_conv6 = Activation('tanh')(de_conv6)
        de_conv6 = BatchNormalization(momentum=0.8)(de_conv6)

        model = Model(inputs=[input_img, input_depth], outputs=de_conv6)
        model.summary()
        return model

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.GT_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        depth = Input(shape=self.GT_shape)
        validity = model(depth)

        return Model(depth, validity)


    def generate_dataset(self, img_path, dep_path, GT_path):
        ## rgb data
        img_List = [img_path + f for f in os.listdir(img_path) if f.endswith('.jpg')]
        img_List.sort()
        # print(img_List[:5])
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
        

        ## input depth
        dep_List = [dep_path + f for f in os.listdir(dep_path) if f.endswith('.jpg')]
        dep_List.sort()
        
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

        return img_data, input_dep_data, resized_dep_data, GT_data


    def save_model(self):

        def save(model, model_name):

            model_path = "refine_models/%s.h5" % model_name
            model.save(model_path)
            # print("saved checkpoint")

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")
    
    def train(self, img_path, dep_path, GT_path, epochs, batch_size=16, sample_interval=50):

        img_data, input_dep_data, resized_dep_data, GT_data = self.generate_dataset(img_path, dep_path, GT_path)
        
        X_train_img = img_data[0:int(0.8*img_data.shape[0])]
        X_vaild_img = img_data[int(0.8*img_data.shape[0]):]
        X_train_dep = resized_dep_data[0:int(0.8*resized_dep_data.shape[0])]
        X_vaild_dep = resized_dep_data[int(0.8*resized_dep_data.shape[0]):]
        y_train_GT = GT_data[0:int(0.8*GT_data.shape[0])]
        y_vaild_GT = GT_data[int(0.8*GT_data.shape[0]):]
        
        X_origin_valid_dep = input_dep_data[int(0.8*resized_dep_data.shape[0]):]
        # X_vaild_dep = resized_dep_data[int(0.8*resized_dep_data.shape[0]):]

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        valid_vaild = np.ones((X_vaild_img.shape[0], 1))
        for epoch in range(epochs):

            # Train Discriminator
            idx = np.random.randint(0, X_train_img.shape[0], batch_size)
            
            X_imgs = X_train_img[idx]
            X_deps = X_train_dep[idx]
            y_GTs = y_train_GT[idx]
            # Generate a batch 
            gen_depth = self.generator.predict([X_imgs, X_deps])
            # print(gen_depth.shape)
            # resized_gen_depth = list()
            # start = time.time()
            # for i in gen_depth:
            #     # print(i)
            #     # resized_gen = scipy.misc.imresize(i[:,:,0], [self.img_rows, self.img_cols], interp='lanczos')
            #     resized_gen = cv2.resize(i[:,:,0], (self.img_cols, self.img_rows), interpolation=cv2.INTER_LANCZOS4)
            #     resized_gen = np.array(resized_gen[..., np.newaxis]).astype('float32') 
            #     resized_gen_depth.append(resized_gen)
            #     # print(resized_gen)
            # resized_gen_depth = np.array(resized_gen_depth)
            # # print(resized_gen_depth.shape)
            # end = time.time()
            # print(end-start)
            # resized_gen_depth

            d_loss_real = self.discriminator.train_on_batch(y_GTs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_depth, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator
            g_loss = self.combined.train_on_batch([X_imgs, X_deps], [y_GTs, valid])
            
            print("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))
            # save generated samples
            # vaild_score = self.combined.evaluate([X_vaild_img, X_vaild_dep], [y_vaild_GT, valid_vaild])
            # print("Validition score: ", vaild_score)
            
            if epoch % sample_interval == 0 and epoch != 0:
                # self.evaluation(X_vaild_img, X_origin_valid_dep, X_vaild_dep, y_vaild_GT)
                idx = np.random.randint(0, X_vaild_img.shape[0], 1)
                X_img = X_vaild_img[idx]
                X_dep = X_vaild_dep[idx]
                X_origin_dep = X_origin_valid_dep[idx]
                X_origin_dep = cv2.resize(X_origin_dep[:,:,0], (self.img_cols, self.img_rows), interpolation=cv2.INTER_LANCZOS4)
                X_origin_dep = np.array(X_origin_dep[..., np.newaxis]).astype('float32') 
                
                gen_depth = self.generator.predict([X_img, X_dep])
                ## mse of sample
                o_mse = np.sqrt(mean_squared_error(X_origin_dep[:,:,0], y_vaild_GT[0,:,:,0]))
                p_mse = np.sqrt(mean_squared_error(gen_depth[0,:,:,0], y_vaild_GT[0,:,:,0]))
                print("sample rmse: orginal : %f, pred : %f" %(o_mse, p_mse))
                # print(gen_depth.shape)
                norm_gen_depth = cv2.normalize(gen_depth[0,:,:,0], None, 0, 255, cv2.NORM_MINMAX)
                cv2.imwrite('./sample_vaild_depth/'+'img_' + str(epoch) + '.jpg', norm_gen_depth)
                cv2.imwrite('./sample_vaild_img/'+'img_' + str(epoch) + '.jpg', X_img[0,:,:,:])
                self.save_model()
    
    def evaluation(self, X_vaild_img, X_origin_valid_dep, X_vaild_dep, y_vaild_GT):

        # print(X_vaild_img.shape, X_origin_valid_dep.shape, y_vaild_GT.shape)
        # original_mse = ((A - B)**2).mean(axis=None)
        resized_origin_valid_depth = list()
        for i in X_origin_valid_dep:
            # resize 160, 128 back to 640, 480
            resized_depth = cv2.resize(i[:,:,0], (self.img_cols, self.img_rows), interpolation=cv2.INTER_LANCZOS4)
            resized_depth = np.array(resized_depth[..., np.newaxis]).astype('float32') 
            resized_origin_valid_depth.append(resized_depth)
            # print(resized_gen)
        resized_origin_valid_depth = np.array(resized_origin_valid_depth)
        # print(resized_origin_valid_depth.shape)
        original_mse = 0
        for i in range(y_vaild_GT.shape[0]):
            original_mse += np.sqrt(mean_squared_error(resized_origin_valid_depth[i,:,:,0], y_vaild_GT[i,:,:,0]))
        original_mse = original_mse/y_vaild_GT.shape[0]

        pre_mse = 0
        gen_depth = self.generator.predict([X_vaild_img, X_vaild_dep])
        for i in range(y_vaild_GT.shape[0]):
            pre_mse += np.sqrt(mean_squared_error(gen_depth[i,:,:,0], y_vaild_GT[i,:,:,0]))
        pre_mse = pre_mse/y_vaild_GT.shape[0]

        print('original rmse: %f, pred rmse: %f' %(original_mse ,pre_mse))


if __name__ == '__main__':
    img_path = '/home/ferrycake/depth_refinement/Pix2Depth/imgs/'
    dep_path = '/home/ferrycake/depth_refinement/FCRN/tensorflow/pre_depth_image/'
    GT_path = '/home/ferrycake/depth_refinement/Pix2Depth/dep_value/'
    DRN = depthRefinementNet()
    # img_data, input_dep_data, resized_dep_data, GT_data = DRN.generate_dataset(img_path, dep_path, GT_path)
    # print(img_data.shape, input_dep_data.shape, resized_dep_data.shape, GT_data.shape)
    DRN.train(img_path, dep_path, GT_path, 30000, batch_size=8, sample_interval=20)