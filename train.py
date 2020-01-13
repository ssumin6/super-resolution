import os
import tensorflow as tf

from model.edsr import edsr
from train_module import Trainer
from data_video import video_ds

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError

# super resolution factor 
scale = 4
# depth of residual blocks
depth = 20
# number of channel of conv blocks
channels = 32

def psnr(y_true, y_pred):
	return tf.image.psnr(y_true, y_pred, max_val=255)

def ssim(y_true, y_pred):
	return tf.image.ssim(y_true, y_pred, max_val=255)

def train(train_ds, valid_ds, ckpt_dir):
	model = Trainer(model = edsr(scale = scale, num_res_blocks= depth, num_filters= channels), learning_rate = 1e-04, checkpoint_dir=
	'./ckpt/')

	'''
	model = edsr(scale = scale, num_res_blocks= depth, num_filters= channels)
	model.compile(optimizer=Adam(learning_rate=1e-04),loss=MeanAbsoluteError(), metrics=[psnr, ssim])
	model.summary()

	
	if (not os.path.exists(ckpt_dir)): 
		os.makedirs(ckpt_dir)

	ckpt_path = os.path.join(ckpt_dir, "latest.ckpt")

	cp_callback = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                 save_weights_only=True,
                                                 verbose=1)

	#model.fit(lr_train, hr_train, batch_size = 64, epochs = ,validation_split= 0.2, callbacks = [cp_callback])
	'''
	model.train(train_ds,
              valid_ds,
              steps=200000, 
              evaluate_every=10000, 
              save_best_only=True)
              

	model.restore()

if __name__ == '__main__':
	#should change to different dataset.
	video_train = video_ds(subset='train')
	video_valid = video_ds(subset='valid')

	train_ds = video_train.dataset(batch_size=64, random_transform=True)
	valid_ds = video_valid.dataset(batch_size=1, random_transform=False, repeat_count=1)
	train(train_ds, valid_ds, ckpt_dir="./ckpt/")