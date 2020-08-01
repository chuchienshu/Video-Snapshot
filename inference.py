import numpy as np
import tensorflow as tf
import datetime, time, scipy.io
from model import *
from util import *
import cv2
# --------------------------------- HYPER-PARAMETERS --------------------------------- #
in_channels = 3
out_channels = 27


def gen_list(filename = ''):
    with open( filename , 'r') as f:
        file_triple_list = f.readlines()
    print('%s is successly opened, containing of %d sequences!' % (filename, len(filename)))
    print(file_triple_list[0])
    return file_triple_list

def read_batch(img_path_list):
    assert len(img_path_list) > 3, 'This video clip is too short.'
    img_batch = []
    for img_path in img_path_list:
        img = cv2.imread(img_path)[:,:,::-1]
        h, w, _ = img.shape
        img = (cv2.resize(img, (w//4 * 4,h//4*4), interpolation=cv2.INTER_AREA) / 127.5 - 1)[np.newaxis,:,:,:]
        img_batch.append(img)
    
    return img_batch

def evaluate(checkpoint_dir, RUN_Encoder = True):
    print('Running ColorEncoder -Evaluation!')
    save_dir_test = os.path.join("./output/results")
    save_dir_encoder = save_dir_test + '-encoder'
    exists_or_mkdir(save_dir_encoder)

    save_dir_decoder = save_dir_test + '-decoder'
    exists_or_mkdir(save_dir_decoder)
	
    # --------------------------------- set model ---------------------------------
    if RUN_Encoder:
        im1_batch = tf.placeholder(tf.float32, shape=(1, None, None, 3))
        im2_batch = tf.placeholder(tf.float32, shape=(1, None, None, 3))
        im3_batch = tf.placeholder(tf.float32, shape=(1, None, None, 3))
        im4_batch = tf.placeholder(tf.float32, shape=(1, None, None, 3))
        im5_batch = tf.placeholder(tf.float32, shape=(1, None, None, 3))
        im6_batch = tf.placeholder(tf.float32, shape=(1, None, None, 3))
        im7_batch = tf.placeholder(tf.float32, shape=(1, None, None, 3))
        im8_batch = tf.placeholder(tf.float32, shape=(1, None, None, 3))
        im9_batch = tf.placeholder(tf.float32, shape=(1, None, None, 3))
        latent_imgs, _ = encode(im1_batch, im2_batch, im3_batch, im4_batch, im5_batch, im6_batch, im7_batch, im8_batch, im9_batch, 3, is_train=False, reuse=False)
    else:
        embed_batch = tf.placeholder(tf.float32, shape=(1, None, None, 3))
        restored_imgs = decode(embed_batch, out_channels, is_train=False, reuse=False)

    # set GPU resources
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.45
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Restore model weights from previously saved model
        check_pt = tf.train.get_checkpoint_state(checkpoint_dir)
        if check_pt and check_pt.model_checkpoint_path:
            saver.restore(sess, check_pt.model_checkpoint_path)
            print('model is loaded successfully.')
        else:
            print('# error: loading checkpoint failed.')
            return None

        start_time = time.time()
        # while not coord.should_stop():
        tm = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        print('%s evaluating:' % (tm))
        if RUN_Encoder:			# save the synthesized invertible grayscale
            data_path = 'data/embedding'
            inputs = sorted( os.listdir(data_path) )
            inputs = ['%s/%s' % (data_path, p) for p in inputs]

            im1, im2, im3, im4, im5, im6, im7, im8, im9 = read_batch(inputs)

            d_dict = {im1_batch:im1, im2_batch:im2, im3_batch:im3, im4_batch:im4, im5_batch:im5, im6_batch:im6, im7_batch:im7, im8_batch:im8, im9_batch:im9}

            snapshot = sess.run(latent_imgs, feed_dict=d_dict)
            save_images_from_batch(snapshot, save_dir_encoder)
            save_images_from_batch(im5, save_dir_encoder,  prefix='gt')
        else:							# save the restored color images
            snapshot_path = 'output/results-encoder'
            inputs = sorted( os.listdir(snapshot_path) )
            inputs = ['%s/%s' % (snapshot_path, p) for p in inputs]
            im1 = (cv2.imread(inputs[0])[:,:,::-1] / 127.5 - 1)[np.newaxis,:,:,:] 
            d_dict = {embed_batch:im1}
            videoclip = sess.run(restored_imgs, feed_dict=d_dict)
            save_images_from_batch(videoclip, save_dir_decoder)

        # Wait for threads to finish.
        # coord.join(threads)
        # sess.close()
        print("Testing finished! consumes %f sec" % (time.time() - start_time))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='embedding', help='embedding, decode')
    args = parser.parse_args()

    if args.mode == 'embedding':
        checkpoint_dir = "fortest"
        evaluate( checkpoint_dir)
    elif args.mode == 'decode':
        checkpoint_dir = "resume-89-checkpoints"
        evaluate( checkpoint_dir, RUN_Encoder=False)
    else:
        raise Exception("Unknow --mode")
