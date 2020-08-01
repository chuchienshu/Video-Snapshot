import tensorflow as tf
from util import *
import random

# tf.enable_eager_execution()

# --------------------------------- LALER FUNCTION ----------------------------------------- #
def Conv2d(batch_input, n_fiter, filter_size, strides, act=None, padding='SAME', name='conv'):
    with tf.variable_scope(name):
        in_channels = batch_input.get_shape()[3]
        filters = tf.get_variable('filter', [filter_size, filter_size, in_channels, n_fiter], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d(batch_input, filters, [1, strides, strides, 1], padding=padding)
        if act is not None:
            conv = act(conv)
        return conv


def Deconv(batch_input, n_fiter, filter_size, strides, act=None, padding='SAME', name='deconv'):
    with tf.variable_scope(name):
        x_shape = tf.shape(batch_input)
        output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, n_fiter])
        in_channels = batch_input.get_shape()[-1]
        filters = tf.get_variable('filter', [filter_size, filter_size, n_fiter, in_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d_transpose(batch_input, filters, output_shape, [1, strides, strides, 1], padding=padding)
        conv = tf.reshape(conv, output_shape)
        if act is not None:
            conv = act(conv)
        return conv

def LeakyReLu(x, a):
    with tf.name_scope("lrelu"):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def Batchnorm(input, act, is_train, name):
    with tf.variable_scope(name):
        input = tf.identity(input)
        variance_epsilon = 1e-5
        normalized = tf.contrib.layers.batch_norm(input, center=True, scale=True, epsilon=variance_epsilon,
                                                  activation_fn=act, is_training=is_train, reuse=None)
        return normalized


def Elementwise(n1, n2, act, name):
    with tf.variable_scope(name):
        return act(n1, n2)


def encode(im1,im2,im3,im4, im5,im6,im7,im8,im9, out_channels, is_train=False, reuse=False):
    ##+++++++++++++++++++++++++++++

    ##
    with tf.variable_scope("encode", reuse=reuse):
        
        m1 = Conv2d(im1, 64, filter_size=3, strides=1, padding='SAME', name='in1/k3n64s1')
        m2 = Conv2d(im2, 64, filter_size=3, strides=1, padding='SAME', name='in2/k3n64s1')
        m3 = Conv2d(im3, 64, filter_size=3, strides=1, padding='SAME', name='in3/k3n64s1')
        m4 = Conv2d(im4, 64, filter_size=3, strides=1, padding='SAME', name='in4/k3n64s1')
        m5 = Conv2d(im5, 64, filter_size=3, strides=1, padding='SAME', name='in5/k3n64s1')
        m6 = Conv2d(im6, 64, filter_size=3, strides=1, padding='SAME', name='in6/k3n64s1')
        m7 = Conv2d(im7, 64, filter_size=3, strides=1, padding='SAME', name='in7/k3n64s1')
        m8 = Conv2d(im8, 64, filter_size=3, strides=1, padding='SAME', name='in8/k3n64s1')
        m9 = Conv2d(im9, 64, filter_size=3, strides=1, padding='SAME', name='in9/k3n64s1')
        #n = Batchnorm(n, act=tf.nn.relu, is_train=is_train, name='in/BN')

        temp_1 = tf.contrib.layers.conv2d(m1, 1,3, scope='temp1')
        temp_2 = tf.contrib.layers.conv2d(m2, 1,3, scope='temp2')
        temp_3 = tf.contrib.layers.conv2d(m3, 1,3, scope='temp3')
        temp_4 = tf.contrib.layers.conv2d(m4, 1,3, scope='temp4')
        temp_5 = tf.contrib.layers.conv2d(m5, 1,3, scope='temp5')
        temp_6 = tf.contrib.layers.conv2d(m6, 1,3, scope='temp6')
        temp_7 = tf.contrib.layers.conv2d(m7, 1,3, scope='temp7')
        temp_8 = tf.contrib.layers.conv2d(m8, 1,3, scope='temp8')
        temp_9 = tf.contrib.layers.conv2d(m9, 1,3, scope='temp9')

        tem_atttion = tf.concat([temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9], axis=-1)
        tem_atttion = tf.nn.softmax(tem_atttion)
        attended_m1 = m1 * tem_atttion[:,:,:,0:1]
        attended_m2 = m2 * tem_atttion[:,:,:,1:2]
        attended_m3 = m3 * tem_atttion[:,:,:,2:3]
        attended_m4 = m4 * tem_atttion[:,:,:,3:4]
        attended_m5 = m5 * tem_atttion[:,:,:,4:5]
        attended_m6 = m6 * tem_atttion[:,:,:,5:6]
        attended_m7 = m7 * tem_atttion[:,:,:,6:7]
        attended_m8 = m8 * tem_atttion[:,:,:,7:8]
        attended_m9 = m9 * tem_atttion[:,:,:,8:9]

        # n = (attended_m1 + attended_m2 + attended_m3) / 3.
        n = tf.add_n([attended_m1 ,attended_m2 ,attended_m3, attended_m4, attended_m5,attended_m6, attended_m7, attended_m8, attended_m9], 'melt')

        # start residual blocks
        for i in range(2):
            nn = Conv2d(n, 64, filter_size=3, strides=1, act=tf.nn.relu, padding='SAME', name='sen64s1/c1/%s' % i)
            #nn = Batchnorm(nn, act=tf.nn.relu, is_train=is_train, name='sen64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, filter_size=3, strides=1, padding='SAME', name='sen64s1/c2/%s' % i)
            #nn = Batchnorm(nn, act=None, is_train=is_train, name='sen64s1/b2/%s' % i)
            nn = Elementwise(n, nn, tf.add, 'seb_residual_add/%s' % i)
            n = nn

        n1 = n
        # down size
        n = Conv2d(n1, 128, filter_size=3, strides=2, padding='SAME', name='down1-1/k3n128s2')
        n2 = Conv2d(n, 128, filter_size=3, strides=1, act=tf.nn.relu, padding='SAME', name='down1-2/k3n128s1')
        #n2 = Batchnorm(n, act=tf.nn.relu, is_train=is_train, name='down1/BN')
        n = Conv2d(n2, 256, filter_size=3, strides=2, padding='SAME', name='down2-1/k3n256s2')
        n = Conv2d(n, 256, filter_size=3, strides=1, act=tf.nn.relu, padding='SAME', name='down2-2/k3n256s1')
        #n = Batchnorm(n, act=tf.nn.relu, is_train=is_train, name='down2/BN')

        # n = spatial_attention(n)

        # residual blocks
        for i in range(4):
            nn = Conv2d(n, 256, filter_size=3, strides=1, act=tf.nn.relu, padding='SAME', name='en64s1/c1/%s' % i)
            #nn = Batchnorm(nn, act=tf.nn.relu, is_train=is_train, name='en64s1/b1/%s' % i)
            nn = Conv2d(nn, 256, filter_size=3, strides=1, padding='SAME', name='en64s1/c2/%s' % i)
            #nn = Batchnorm(nn, act=None, is_train=is_train, name='en64s1/b2/%s' % i)
            nn = Elementwise(n, nn, tf.add, 'eb_residual_add/%s' % i)
            n = nn
        
        # up size
        n = tf.image.resize_images(n, [tf.shape(n)[1] * 2, tf.shape(n)[2] * 2], method=1)
        n = Conv2d(n, 128, filter_size=3, strides=1, padding='SAME', name='up1-1/k3n128s1')
        n = Conv2d(n, 128, filter_size=3, strides=1, act=tf.nn.relu, padding='SAME', name='up1-2/k3n128s1')
        #n = Batchnorm(n, act=tf.nn.relu, is_train=is_train, name='up1/BN')
        n = Elementwise(n, n2, tf.add, 'skipping1')

        n = tf.image.resize_images(n, [tf.shape(n)[1] * 2, tf.shape(n)[2] * 2], method=1)
        n = Conv2d(n, 64, filter_size=3, strides=1, padding='SAME', name='up2-1/k3n64s1')
        n = Conv2d(n, 64, filter_size=3, strides=1, act=tf.nn.relu, padding='SAME', name='up2-2/k3n64s1')
        #n = Batchnorm(n, act=tf.nn.relu, is_train=is_train, name='up2/BN')
        n = Elementwise(n, n1, tf.add, 'skipping2')
        # n = tf.add_n([n, n1], 'skipping2')

        # end residual blocks
        for i in range(2):
            nn = Conv2d(n, 64, filter_size=3, strides=1, act=tf.nn.relu, padding='SAME', name='een64s1/c1/%s' % i)
            #nn = Batchnorm(nn, act=tf.nn.relu, is_train=is_train, name='een64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, filter_size=3, strides=1, padding='SAME', name='een64s1/c2/%s' % i)
            #nn = Batchnorm(nn, act=None, is_train=is_train, name='een64s1/b2/%s' % i)
            nn = Elementwise(n, nn, tf.add, 'eeb_residual_add/%s' % i)
            n = nn

        latent_maps = Conv2d(n, out_channels, filter_size=3, strides=1, act=tf.nn.tanh, padding='SAME', name='latent')

        return latent_maps, tem_atttion


def decode(latents, out_channels, is_train=False, reuse=False):
    with tf.variable_scope("decode", reuse=reuse):
        # Decoder
        a = Conv2d(latents, 64, filter_size=3, strides=1, padding='SAME', name='in/k3n64s1')
        n = 0. + a
        for i in range(3):
            nn = Conv2d(n, 128, filter_size=3, strides=1, act=tf.nn.relu, padding='SAME', name='dn64s1/a1/%s' % i)
            #nn = Batchnorm(nn, act=tf.nn.relu, is_train=is_train, name='dn64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, filter_size=3, strides=1, padding='SAME', name='dn64s1/a2/%s' % i)
            #nn = Batchnorm(nn, act=None, is_train=is_train, name='dn64s1/b2/%s' % i)
            nn = Elementwise(n, nn, tf.add, 'db_residual_adda/%s' % i)
            n = nn
        n = a+n
        a = n
        for i in range(2):
            nn = Conv2d(n, 128, filter_size=3, strides=1, act=tf.nn.relu, padding='SAME', name='dn64s2/b1/%s' % i)
            #nn = Batchnorm(nn, act=tf.nn.relu, is_train=is_train, name='dn64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, filter_size=3, strides=1, padding='SAME', name='dn64s2/b2/%s' % i)
            #nn = Batchnorm(nn, act=None, is_train=is_train, name='dn64s1/b2/%s' % i)
            nn = Elementwise(n, nn, tf.add, 'db_residual_addb/%s' % i)
            n = nn
        
        n = a+n
        a = n

        for i in range(3):
            nn = Conv2d(n, 128, filter_size=3, strides=1, act=tf.nn.relu, padding='SAME', name='dn64s3/c1/%s' % i)
            #nn = Batchnorm(nn, act=tf.nn.relu, is_train=is_train, name='dn64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, filter_size=3, strides=1, padding='SAME', name='dn64s3/c2/%s' % i)
            #nn = Batchnorm(nn, act=None, is_train=is_train, name='dn64s1/b2/%s' % i)
            nn = Elementwise(n, nn, tf.add, 'db_residual_addc/%s' % i)
            n = nn
        
        n = a+n

        n = Conv2d(n, 256, filter_size=3, strides=1, act=None, padding='SAME', name='n256s1/2')
        n = Conv2d(n, out_channels, filter_size=1, strides=1, padding='SAME', name='out')
        output_map = tf.nn.tanh(n)

        return output_map

