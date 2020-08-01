from __future__ import division
from __future__ import print_function
import os, glob, shutil, math
import tensorflow as tf
import numpy as np
from PIL import Image


def exists_or_mkdir(path, need_remove=False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif need_remove:
        shutil.rmtree(path)
        os.makedirs(path)
    return None


def save_list(save_path, data_list):
    n = len(data_list)
    with open(save_path, 'w') as f:
        f.writelines([str(data_list[i]) + '\n' for i in range(n)])
    return None
	

def save_images_from_batch(img_batch, save_dir, init_no=0, prefix=''):

    channels = img_batch.shape[-1]

    nums = channels //3

    for i in range(nums):
        image = Image.fromarray((127.5*(img_batch[0, :, :, 3*i:3*(i+1)]+1)+0.5).astype(np.uint8))
        image.save(os.path.join(save_dir, 'result_%s_%05d.png' % (prefix,init_no + i)), 'PNG')
           
    return None


def compute_color_psnr(im_batch1, im_batch2):
    mean_psnr = 0
    im_batch1 = im_batch1.squeeze()
    im_batch2 = im_batch2.squeeze()
    num = im_batch1.shape[0]
    for i in range(num):
        # Convert pixel value to [0,255]
        im1 = 127.5 * (im_batch1[i]+1)
        im2 = 127.5 * (im_batch2[i]+1)
        #print(im1.shape)
        psnr1 = calc_psnr(im1[:,:,0], im2[:,:,0])
        psnr2 = calc_psnr(im1[:,:,1], im2[:,:,1])
        psnr3 = calc_psnr(im1[:,:,2], im2[:,:,2])
        mean_psnr += (psnr1+psnr2+psnr3) / 3.0
    return mean_psnr/num


def measure_psnr(im_batch1, im_batch2):
    mean_psnr = 0
    num = im_batch1.shape[0]
    for i in range(num):
        # Convert pixel value to [0,255]
        im1 = 127.5 * (im_batch1[i]+1)
        im2 = 127.5 * (im_batch2[i]+1)
        psnr = calc_psnr(im1, im2)
        mean_psnr += psnr
    return mean_psnr/num


def calc_psnr(im1, im2):
    '''
    Notice: Pixel value should be convert to [0,255]
    '''
    if im1.shape[-1] != 3:
        g_im1 = im1.astype(np.float32)
        g_im2 = im2.astype(np.float32)
    else:
        g_im1 = np.array(Image.fromarray(im1).convert('L'), np.float32)
        g_im2 = np.array(Image.fromarray(im2).convert('L'), np.float32)

    mse = np.mean((g_im1 - g_im2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

## Helper functions
def peak_signal_to_noise_ratio(true, pred):
  """Image quality metric based on maximal signal power vs. power of the noise.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    peak signal to noise ratio (PSNR)
  """
  return 10.0 * tf.log(1.0 / mean_squared_error(true, pred)) / tf.log(10.0)


def mean_squared_error(true, pred):
  """L2 distance between tensors true and pred.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
  return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))