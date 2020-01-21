import os
import tensorflow as tf
import numpy as np
from PIL import Image

from model import evaluate, resolve_single, psnr, ssim
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

def bilinear_upscale(lr_dir, hr_dir, scale):
	psnr_values = []
	ssim_values = []
	ns = [i*30 for i in range(60)]
	for index in ns:
		#bilinear upscale
		file_name = 'frame%04d.jpg' %index
		lr = Image.open(os.path.join(lr_dir,file_name)).convert("RGB")
		hr = Image.open(os.path.join(hr_dir,file_name)).convert("RGB")
		sr = lr.resize((1704, 960), Image.BILINEAR)
		hr = np.asanyarray(hr)
		sr = np.asanyarray(sr)
		psnr_value = psnr(hr, sr)
        #ssim_value = ssim(hr, sr)
        psnr_values.append(psnr_value)
        #ssim_values.append(ssim_value)
	return tf.reduce_mean(psnr_values) #, tf.reduce_mean(ssim_values)


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

	psnr_b = bilinear_upscale('../image_240','../image_960',scale=scale)
	print('bilinear upscale PSNR:%.3f' % psnr_b)

	lr = load_image('../image_240/frame1500.jpg')
	sr = resolve_single(checkpoint.model, lr)
	plt.imshow(sr)
	plt.show()

test()