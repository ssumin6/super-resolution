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

def train(train_ds, valid_ds, ckpt_dir):
	model = Trainer(model = edsr(scale = scale, num_res_blocks= depth, num_filters= channels), learning_rate = 1e-04, checkpoint_dir=
	'./ckpt/')

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