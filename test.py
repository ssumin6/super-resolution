import os
import tensorflow as tf

from model import evaluate
from model.edsr import edsr
from data_video import video_ds

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

test()