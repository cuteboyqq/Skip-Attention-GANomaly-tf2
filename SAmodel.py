import time
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from tensorflow.keras import layers

import metrics
from absl import logging
import matplotlib.pyplot as plt
import os
from Attention import ChannelAttention,SpatialAttention
#==============================================
'''
from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
'''
#========class use tensorflow.keras.layers======================================================================
import tensorflow.keras as keras
from tensorflow.keras import layers, Sequential, regularizers
#============================================================================================

#============================================================================================
class UNetDown(tf.keras.layers.Layer):
    def __init__(self, filters, f_size=4, normalize=True):
        super(UNetDown, self).__init__()
        """Layers used during downsampling"""
        self.conv = layers.Conv2D(filters, kernel_size=f_size, strides=2, padding='same')
        self.relu = layers.LeakyReLU(alpha=0.2) #alpha=0.2
        self.normalize = normalize
        self.norm = layers.BatchNormalization(epsilon=1e-05, momentum=0.9) #epsilon=1e-05, momentum=0.9
    def call(self,x):
        x = self.conv(x)
        x = self.norm(x) if self.normalize else x
        x = self.relu(x)
        return x

        
class UNetUp(tf.keras.layers.Layer):
    def __init__(self, out_size, dropout_rate=0):
        super(UNetUp, self).__init__()
        """Layers used during upsampling"""
        self.upconv = layers.Conv2DTranspose(out_size, 4, 2, padding='same')
        self.upsample = layers.UpSampling2D(size=2)
        self.conv_tr = layers.Conv2D(out_size,
                                  kernel_size=4,
                                  strides=1,
                                  padding='same',
                                  use_bias=False)
        #self.conv = layers.Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')
        self.relu = layers.ReLU()#alpha=0.2
        self.dropout_rate = dropout_rate
        if dropout_rate:
            self.drop = layers.Dropout(dropout_rate)
        self.norm = layers.BatchNormalization(epsilon=1e-05, momentum=0.9) #epsilon=1e-05, momentum=0.9
        self.concat = layers.Concatenate()
    def call(self,x,skip_input):
        #print('x {}'.format(x.shape))
        x = self.upsample(x)
        x = self.conv_tr(x)
        #print('upconv {}'.format(x.shape))
        x = self.norm(x)
        #print('norm {}'.format(x.shape))
        x = self.relu(x)
        #print('relu {}'.format(x.shape))
        #x = self.conv(x)
        x = self.drop(x) if self.dropout_rate else x
        
        x = self.concat([x,skip_input])
        return x
    
class SA_Encoder(tf.keras.layers.Layer):
    """ DCGAN ENCODER NETWORK
    """
    
        
    def __init__(self,
                 isize,
                 nz,
                 nc,
                 ndf,
                 n_extra_layers=0,
                 output_features=False):
        """
        Params:
            isize(int): input image size
            nz(int): num of latent dims
            nc(int): num of input dims
            ndf(int): num of discriminator(Encoder) filters
        """
        super(SA_Encoder, self).__init__()
        self.gf = ndf
        self.img_rows = isize
        self.img_cols = isize
        self.channels = nc
        #self.cc = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        # state size. K x 4 x 4
        self.output_features = output_features
        self.out_conv = layers.Conv2D(filters=nz,
                                      kernel_size=2,
                                      strides=2,
                                      padding='valid'
                                      )#padding='same'
        
        
        self.conv_tr = layers.Conv2D(self.gf,
                                  kernel_size=4,
                                  strides=1,
                                  padding='same',
                                  use_bias=False)
        
        self.down1 = UNetDown(self.gf*1, normalize=False)
        self.down2 = UNetDown(self.gf*2)
        self.down3 = UNetDown(self.gf*4)
        self.down4 = UNetDown(self.gf*8)
        self.down5 = UNetDown(self.gf*8)
        #self.down6 = UNetDown(self.gf*8)
        #self.down7 = UNetDown(self.gf*8)
        self.ca0 = ChannelAttention(self.gf*1)
        self.sa0 = SpatialAttention()
        
        
        self.ca1 = ChannelAttention(self.gf*1)
        self.sa1 = SpatialAttention()
        
        self.ca2 = ChannelAttention(self.gf*2)
        self.sa2 = SpatialAttention()
        
        self.ca3 = ChannelAttention(self.gf*4)
        self.sa3 = SpatialAttention()
        
        self.ca4 = ChannelAttention(self.gf*8)
        self.sa4 = SpatialAttention()
        
        
        self.ca5 = ChannelAttention(self.gf*8)
        self.sa5 = SpatialAttention()
   
    def call(self, x):
        # Image input
        #d0 = layers.Input(shape=self.img_shape)
        #print('d0 {}'.format(d0.shape))
        # Downsampling
        #x:32x32
        d1 = self.down1(x) #d1 :16x16
        #print('d1 {}'.format(d1.shape))
        d2 = self.down2(d1) #d2 : 8x8 
        #print('d2 {}'.format(d2.shape))
        d3 = self.down3(d2) #d3 : 4x4
        #print('d3 {}'.format(d3.shape))
        
        #last_features = d3 # last_features : 4x4
        #print('last_features {}'.format(last_features.shape))
        d4 = self.down4(d3) #d4 : 2x2
        last_features = d4
        #print('d4 {}'.format(d4.shape))
        d5 = self.down5(d4) #d5 : 1x1
        #print('d5 {}'.format(d5.shape))
        
        #print('d5 {}'.format(d5.shape))
        #d6 = self.down6(d5)
        #d7 = self.down7(d6)
        x2 = self.conv_tr(x)
        d0 = self.ca0(x2) * x2
        _d0 = self.sa0(d0) * d0   #d0 : 32x32
        
        
        d1 = self.ca1(d1) * d1
        _d1 = self.sa1(d1) * d1 #d1 : 16x16
        
        d2 = self.ca2(d2) * d2
        _d2 = self.sa2(d2) * d2 #d2 : 8x8
        
        d3 = self.ca3(d3) * d3
        _d3 = self.sa3(d3) * d3 #d3 : 4x4
        
        d4 = self.ca4(d4) * d4
        _d4 = self.sa4(d4) * d4 #d2 : 2x2
        
        d5 = self.ca4(d5) * d5
        _d5 = self.sa4(d5) * d5 #d1 : 1x1
        
        #d = [d1,d2,d3,d4,d5]
        d = [_d1,_d2,_d3,_d4,_d5,_d0]
        #d = [d1,d2,d3,d4,d5]
        out = self.out_conv(last_features)
        #print('out {}'.format(out.shape))
        #print('out {}'.format(out.shape))
        if self.output_features:
            return out, last_features
        else:
            return out, d
        
class SA_Decoder(tf.keras.layers.Layer):
    def __init__(self, isize, nz, nc, ngf, n_extra_layers=0):
        """
        Params:
            isize(int): input image size
            nz(int): num of latent dims
            nc(int): num of input dims
            ngf(int): num of Generator(Decoder) filters
        """
        super(SA_Decoder, self).__init__()
        self.gf = ngf
        self.con1 = layers.Conv2D(self.gf*4, kernel_size=4, strides=1, padding='same', activation='relu')
        self.upsample = layers.UpSampling2D(size=2)
        
        self.act = layers.ReLU()
        self.bn = layers.BatchNormalization(epsilon=1e-05, momentum=0.9)
        self.conv_tr = layers.Conv2D(self.gf*8,
                                  kernel_size=4,
                                  strides=1,
                                  padding='same',
                                  use_bias=False)
        
        self.conv_tr2 = layers.Conv2D(self.gf,
                                  kernel_size=4,
                                  strides=1,
                                  padding='same',
                                  use_bias=False)
        
        self.conv_tr3 = layers.Conv2D(self.gf*2,
                                  kernel_size=4,
                                  strides=1,
                                  padding='same',
                                  use_bias=False)
        
        self.upconv = layers.Conv2DTranspose(self.gf*4,4,strides=2,padding='valid')
        self.channels = nc
        self.conv = layers.Conv2D(self.channels, kernel_size=4, strides=1,
                            padding='same', activation='tanh', use_bias=False)
        
        #self.up1 = UNetUp(self.gf*8)
        #self.up1 = UNetUp(self.gf*8)
        self.up2 = UNetUp(self.gf*8)
        self.up3 = UNetUp(self.gf*4)
        self.up4 = UNetUp(self.gf*2)
        self.up5 = UNetUp(self.gf)
        self.up6 = UNetUp(self.gf)
        
        self.tanh = tf.keras.activations.tanh
    def call(self,x,d):
        # Upsampling
        #x = self.upconv(x)
        x = self.upsample(x) # x:2x2
        #x = self.upsample(x) # x:4x4
        x = self.conv_tr(x) # x: 2x2
        #print('x {}'.format(x.shape))
        #u1 = self.up1(x, d[5])
        u2 = self.up2(x, d[2]) #u2 : 4x4
        #c1 = self.con1(x)
        #u0 = self.upsample(c1)
        #print('c1 {}'.format(c1.shape))
        #print('u0 {}'.format(u0.shape))
        #print('d[4] {}'.format(d[4].shape))
        #u1 = self.up1(x, d[4]),
        #print('u1 {}'.format(u1.shape))
        #print('d[3] {}'.format(d[3].shape))
        #u2 = self.up2(u1, d[3])
        #print('u2 {}'.format(u2.shape))
        #print('d[2] {}'.format(d[2].shape))
        
        #Notes d = [_d1,_d2,_d3,_d4,_d5] 
        #index d = [  0,  1,  2,  3, 4]
        #size  d = [ 16, 8,  4,  2, 1]         
        u3 = self.up3(u2, d[1]) # u4:8x8
        #print('u4 {}'.format(u4.shape))
        u4 = self.up4(u3, d[0]) # u4 :16x16
        #u4 = self.upsample(u3) #u4 : 16x16
        #u4 = self.conv_tr3(u4) # u4 : 16x16
        #u4 = self.bn(u4)
        #u4 = self.act(u4)
        #print('u5 {}'.format(u5.shape))
        #u5 = self.up5(u4, d[0])
        
        #u6 = self.up6(u4,d[5]) #u6 : 32x32
        u5 = self.upsample(u4) # u5:32x32
        #print('u6 up {}'.format(u5.shape))
        #print('u6 upsample {}'.format(u6.shape))
        u6 = self.conv_tr2(u5) # u6:32x32
        #u6 = self.bn(u6)
        u6 = self.act(u6)
        #print('u6 conv {}'.format(u6.shape))
        gen_img = self.conv(u6) #gen_img:32x32x3
        
        #gen_img = self.tanh(gen_img)
        #print('gen_img {}'.format(gen_img.shape))
        return gen_img
    
class SA_NetG(tf.keras.Model):
    def __init__(self, opt):
        super(SA_NetG, self).__init__()

       
        self.encoder1 = SA_Encoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.extralayers)
        self.decoder = SA_Decoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.extralayers)
        self.encoder2 = SA_Encoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.extralayers)

    def call(self, x):
        latent_i,d = self.encoder1(x)
        gen_img = self.decoder(latent_i,d)
        latent_o,d = self.encoder2(gen_img)
        return latent_i, gen_img, latent_o

    def num_params(self):
        return sum(
            [np.prod(var.shape.as_list()) for var in self.trainable_variables])
    
class SA_NetD(tf.keras.Model):
    """ DISCRIMINATOR NETWORK
    """
    def __init__(self, opt):
        super(SA_NetD, self).__init__()
        self.encoder = SA_Encoder(opt.isize, 1, opt.nc, opt.ngf, opt.extralayers, output_features=True)
        self.sigmoid = layers.Activation(tf.sigmoid)

    def call(self, x):
        output, last_features = self.encoder(x)
        output = self.sigmoid(output)
        return output, last_features
#================================================================================================================
class GANRunner:
    def __init__(self,
                 G,
                 D,
                 best_state_key,
                 best_state_policy,
                 train_dataset,
                 valid_dataset=None,
                 test_dataset=None,
                 save_path='ckpt/'):
        self.G = G
        self.D = D
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.num_ele_train = self._get_num_element(self.train_dataset)
        self.best_state_key = best_state_key
        self.best_state_policy = best_state_policy
        self.best_state = 1e-9 if self.best_state_policy == max else 1e9
        self.save_path = save_path

    def train_step(self, x, y):
        raise NotImplementedError

    def validate_step(self, x, y):
        raise NotImplementedError

    def evaluate(self, x):
        raise NotImplementedError

    def _get_num_element(self, dataset):
        num_elements = 0
        for _ in dataset:
            num_elements += 1
        return num_elements

    def fit(self, num_epoch, best_state_ths=None):
        self.best_state = self.best_state_policy(
            self.best_state,
            best_state_ths) if best_state_ths is not None else self.best_state
        for epoch in range(num_epoch):
            start_time = time.time()
            # train one epoch
            G_losses = []
            D_losses = []
            with tqdm(total=self.num_ele_train, leave=False) as pbar:
                for step, (x_batch_train,
                           y_batch_train) in enumerate(self.train_dataset):
                    loss = self.train_step(x_batch_train, y_batch_train)
                    G_losses.append(loss[0].numpy())
                    D_losses.append(loss[1].numpy())
                    pbar.update(1)
                G_losses = np.array(G_losses).mean()
                D_losses = np.array(D_losses).mean()
                speed = step * len(x_batch_train) / (time.time() - start_time)
                logging.info(
                    'epoch: {}, G_losses: {:.4f}, D_losses: {:.4f}, samples/sec: {:.4f}'
                    .format(epoch, G_losses, D_losses, speed))

            # validate one epoch
            if self.valid_dataset is not None:
                G_losses = []
                D_losses = []
                for step, (x_batch_train,
                           y_batch_train) in enumerate(self.valid_dataset):
                    loss = self.validate_step(x_batch_train, y_batch_train)
                    G_losses.append(loss[0].numpy())
                    D_losses.append(loss[1].numpy())
                G_losses = np.array(G_losses).mean()
                D_losses = np.array(D_losses).mean()
                logging.info(
                    '\t Validating: G_losses: {}, D_losses: {}'.format(
                        G_losses, D_losses))
            if epoch>10:
                # evaluate on test_dataset
                if self.test_dataset is not None:
                    dict_ = self.evaluate(self.test_dataset)
                    log_str = '\t Testing:'
                    for k, v in dict_.items():
                        log_str = log_str + '   {}: {:.4f}'.format(k, v)
                    state_value = dict_[self.best_state_key]
                    self.best_state = self.best_state_policy(
                        self.best_state, state_value)
                    if self.best_state == state_value:
                        log_str = '*** ' + log_str + ' ***'
                        self.save_best()
                    logging.info(log_str)

    def save(self, path):
        #self.G.save_weights(self.save_path + 'G')
        #self.D.save_weights(self.save_path + 'D')
        #tf.saved_model.save(model, "saved_model_keras_dir")
        self.G.save(self.save_path + 'G')
        self.D.save(self.save_path + 'D')

    def load(self, path):
        #self.G.load_weights(self.save_path + 'G')
        #self.D.load_weights(self.save_path + 'D')
        self.G = tf.keras.models.load_model(self.save_path + 'G')
        self.D = tf.keras.models.load_model(self.save_path + 'D')

    def save_best(self):
        self.save(self.save_path + 'best') 

    def load_best(self):
        self.load(self.save_path + 'best')

class Skip_Attention_GANomaly(GANRunner):
    def __init__(self,
                 opt,
                 train_dataset,
                 valid_dataset=None,
                 test_dataset=None):
        self.opt = opt
        self.G = SA_NetG(self.opt)
        self.D = SA_NetD(self.opt)
        super(Skip_Attention_GANomaly, self).__init__(self.G,
                                                       self.D,
                                                       best_state_key='roc_auc',
                                                       best_state_policy=max,
                                                       train_dataset=train_dataset,
                                                       valid_dataset=valid_dataset,
                                                       test_dataset=test_dataset)
        self.D(tf.keras.Input(shape=[opt.isize] if opt.encdims else [opt.isize, opt.isize, opt.nc]))
        #self.D(tf.keras.Input(shape=[opt.isize, opt.isize, opt.nc]))
        self.D_init_w_path = '/tmp/D_init'
        self.D.save_weights(self.D_init_w_path)

        # label
        self.real_label = tf.ones([
            self.opt.batch_size,
        ], dtype=tf.float32)
        self.fake_label = tf.zeros([
            self.opt.batch_size,
        ], dtype=tf.float32)

        # loss
        l2_loss = tf.keras.losses.MeanSquaredError()
        l1_loss = tf.keras.losses.MeanAbsoluteError()
        bce_loss = tf.keras.losses.BinaryCrossentropy()

        # optimizer
        self.d_optimizer = tf.keras.optimizers.Adam(self.opt.lr,
                                                    beta_1=self.opt.beta1,
                                                    beta_2=0.999)
        self.g_optimizer = tf.keras.optimizers.Adam(self.opt.lr,
                                                    beta_1=self.opt.beta1,
                                                    beta_2=0.999)

        # adversarial loss (use feature matching)
        self.l_adv = l2_loss
        # contextual loss
        self.l_con = l1_loss
        # Encoder loss
        self.l_enc = l2_loss
        # discriminator loss
        self.l_bce = bce_loss
    
        self.show_max_num = 5
        
    def renormalize(self, tensor):
        minFrom= tf.math.reduce_min(tensor)
        maxFrom= tf.math.reduce_max(tensor)
        minTo = 0
        maxTo = 1
        return minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))
    
    
    def infer_cropimage(self,image):
        abnormal = 0
        self.input = image
        self.latent_i, self.gen_img, self.latent_o = self.G(self.input)
        self.pred_real, self.feat_real = self.D(self.input)
        self.pred_fake, self.feat_fake = self.D(self.gen_img)
        g_loss = self.g_loss()
        if g_loss>1.0:
            abnormal=1
            print('abnoraml')
        else:
            abnormal=0
            print('normal')
        return abnormal
        
    def infer(self, test_dataset,SHOW_MAX_NUM,show_img,data_type):
        show_num = 0
        self.load_best()
        
        
        loss_list = []
        dataiter = iter(test_dataset)
        #for step, (images, y_batch_train) in enumerate(test_dataset):
        cnt=1
        os.makedirs('./runs/detect',exist_ok=True)
        while(show_num < SHOW_MAX_NUM):
            images, labels = dataiter.next()
            #latent_i, fake_img, latent_o = self.G(images)
            self.input = images
            
            self.latent_i, self.gen_img, self.latent_o = self.G(self.input)
            self.pred_real, self.feat_real = self.D(self.input)
            self.pred_fake, self.feat_fake = self.D(self.gen_img)
            g_loss = self.g_loss()
            
            #g_loss = 0.0
            #print("input")
            #print(self.input)
            #print("gen_img")
            #print(self.gen_img)
            images = self.renormalize(self.input)
            fake_img = self.renormalize(self.gen_img)
            #fake_img = self.gen_img
            images = images.cpu().numpy()
            fake_img = fake_img.cpu().numpy()
            #fake_img = self.gen_img
            #print(fake_img.shape)
            #print(images.shape)
            if show_img:
                plt = self.plot_images(images,fake_img)
                if data_type=='normal':
                    file_name = 'infer_normal' + str(cnt) + '.jpg'
                else:
                    file_name = 'infer_abnormal' + str(cnt) + '.jpg'
                file_path = os.path.join('./runs/detect',file_name)
                plt.savefig(file_path)
                cnt+=1
            if data_type=='normal':
                print('{} normal: {}'.format(show_num,g_loss.numpy()))
            else:
                print('{} abnormal: {}'.format(show_num,g_loss.numpy()))
            #if g_loss>=3:
                #g_loss = 3
            loss_list.append(g_loss)
            show_num+=1
            #if show_num%20==0:
                #print(show_num)
        return loss_list
    def plot_images(self,images,outputs):
        # plot the first ten input images and then reconstructed images
        fig, axes = plt.subplots(nrows=2, ncols=15, sharex=True, sharey=True, figsize=(25,4))
        # input images on top row, reconstructions on bottom
        for images2, row in zip([images,outputs], axes):     
            for img, ax in zip(images2, row):
                #img = img[:,:,::-1].transpose((2,1,0))
                #print(img)
                ax.imshow(img)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        return plt
    
    def plot_loss_distribution(self, SHOW_MAX_NUM,positive_loss,defeat_loss):
        # Importing packages
        import matplotlib.pyplot as plt2
        # Define data values
        x = [i for i in range(SHOW_MAX_NUM)]
        y = positive_loss
        z = defeat_loss
        print(x)
        print(positive_loss)
        print(defeat_loss)
        # Plot a simple line chart
        #plt2.plot(x, y)
        # Plot another line on the same chart/graph
        #plt2.plot(x, z)
        plt2.scatter(x,y,s=1)
        plt2.scatter(x,z,s=1) 
        os.makedirs('./runs/detect',exist_ok=True)
        file_path = os.path.join('./runs/detect','loss_distribution.jpg')
        plt2.savefig(file_path)
        plt2.show()
        
    
    
    def _evaluate(self, test_dataset):
        an_scores = []
        gt_labels = []
        for step, (x_batch_train, y_batch_train) in enumerate(test_dataset):
            latent_i, gen_img, latent_o = self.G(x_batch_train)
            latent_i, gen_img, latent_o = latent_i.numpy(), gen_img.numpy(
            ), latent_o.numpy()
            error = np.mean((latent_i - latent_o)**2, axis=-1)
            an_scores.append(error)
            gt_labels.append(y_batch_train)
        an_scores = np.concatenate(an_scores, axis=0).reshape([-1])
        gt_labels = np.concatenate(gt_labels, axis=0).reshape([-1])
        return an_scores, gt_labels

    def evaluate(self, test_dataset):
        ret_dict = {}
        an_scores, gt_labels = self._evaluate(test_dataset)
        # normed to [0,1)
        an_scores = (an_scores - np.amin(an_scores)) / (np.amax(an_scores) -
                                                        np.amin(an_scores))
        # AUC
        auc_dict = metrics.roc_auc(gt_labels, an_scores)
        ret_dict.update(auc_dict)
        # Average Precision
        p_r_dict = metrics.pre_rec_curve(gt_labels, an_scores)
        ret_dict.update(p_r_dict)
        return ret_dict

    def evaluate_best(self, test_dataset):
        self.load_best()
        an_scores, gt_labels = self._evaluate(test_dataset)
        # AUC
        _ = metrics.roc_auc(gt_labels, an_scores, show=True)
        # Average Precision
        _ = metrics.pre_rec_curve(gt_labels, an_scores, show=True)

    @tf.function
    def _train_step_autograph(self, x):
        """ Autograph enabled by tf.function could speedup more than 6x than eager mode.
        """
        self.input = x
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            self.latent_i, self.gen_img, self.latent_o = self.G(self.input)
            self.pred_real, self.feat_real = self.D(self.input)
            self.pred_fake, self.feat_fake = self.D(self.gen_img)
            g_loss = self.g_loss()
            d_loss = self.d_loss()

        g_grads = g_tape.gradient(g_loss, self.G.trainable_weights)
        d_grads = d_tape.gradient(d_loss, self.D.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads,
                                             self.G.trainable_weights))
        self.d_optimizer.apply_gradients(zip(d_grads,
                                             self.D.trainable_weights))
        return g_loss, d_loss

    def train_step(self, x, y):
        g_loss, d_loss = self._train_step_autograph(x)
        if d_loss < 1e-5:
            st = time.time()
            self.D.load_weights(self.D_init_w_path)
            logging.info('re-init D, cost: {:.4f} secs'.format(time.time() -
                                                               st))

        return g_loss, d_loss

    def validate_step(self, x, y):
        pass

    def g_loss(self):
        self.err_g_adv = self.l_adv(self.feat_real, self.feat_fake)
        self.err_g_con = self.l_con(self.input, self.gen_img)
        self.err_g_enc = self.l_enc(self.latent_i, self.latent_o)
        g_loss = self.err_g_adv * self.opt.w_adv + \
                self.err_g_con * self.opt.w_con + \
                self.err_g_enc * self.opt.w_enc
        return g_loss

    def d_loss(self):
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)
        d_loss = (self.err_d_real + self.err_d_fake) * 0.5
        return d_loss
