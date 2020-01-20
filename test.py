import os
import tensorflow as tf
import cv2

from model import evaluate, resolve_single
from model.edsr import edsr
from data_video import video_ds
from utils import load_image
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam

# super resolution factor 
scale = 4
# depth of residual blocks
depth = 20
# number of channel of conv blocks
channels = 32

def restore(checkpoint, checkpoint_manager):
	if checkpoint_manager.latest_checkpoint:
		checkpoint.restore(checkpoint_manager.latest_checkpoint)
		print('Model restored from checkpoint at step %d.' %checkpoint.step.numpy())

def bilinear_upscale(valid_ds, scale):
	psnr_values = []
    ssim_values = []
    for lr, hr in dataset:
        #bilinear upscale
        sr = cv2.resize(lr, dsize=(lr.shape[0]*scale,lr.shape[1]*scale), interpolation=cv2.INTER_LINEAR)
        psnr_value = psnr(hr, sr)[0]
        ssim_value = ssim(hr, sr)[0]
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
    return tf.reduce_mean(psnr_values), tf.reduce_mean(ssim_values)


def test():
	model = edsr(scale = scale, num_res_blocks= depth, num_filters= channels)
	checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                    psnr=tf.Variable(-1.0),
                                    optimizer=Adam(1e-04),
                                    model=model)
	checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                    directory='./ckpt',
                                                    max_to_keep=3)
	restore(checkpoint, checkpoint_manager)

	video_valid = video_ds(subset='valid')
	valid_ds = video_valid.dataset(batch_size=1, random_transform=False, repeat_count=1)

	psnr, ssim = evaluate(checkpoint.model, valid_ds)
	print('PSNR:%.3f, SSIM:%.3f' % (psnr, ssim))
	psnr_b, ssim_b = bilinear_upscale(valid_ds, scale=scale)
	print('bilinear upscale PSNR:%.3f, SSIM:%.3f' % (psnr_b, ssim_b))

	lr = load_image('../image_240/frame1500.jpg')
	sr = resolve_single(checkpoint.model, lr)
	plt.imshow(sr)
	plt.show()

test()