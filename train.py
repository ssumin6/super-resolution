import os
from model.edsr import edsr
from train_module import EdsrTrainer

# super resolution factor 
scale = 4
# depth of residual blocks
depth = 20
# number of channel of conv blocks
channels = 32


def train(train_ds, valid_ds, weights_dir):
	edsr = EdsrTrainer(model = edsr(scale = scale, num_res_blocks= depth, num_filters= channels), checkpoint_dir=f'.ckpt/edsr-{depth}-x{scale}', learning_rate = 1e-04)

	edsr.train(train_ds,
              valid_ds.take(10),
              steps=200000, 
              evaluate_every=10000, 
              save_best_only=True)
	edsr.restore()
	edsr.model.save_weights(weights_file)

if __name__ == '__main__':
	# Location of model weights (needed for demo)
	weights_dir = f'weights/edsr-{depth}-x{scale}'
	weights_file = os.path.join(weights_dir, 'weights.h5')

	os.makedirs(weights_dir, exist_ok=True)

	train_ds = div2k_train.dataset(batch_size=64, random_transform=True)
	valid_ds = div2k_valid.dataset(batch_size=1, random_transform=False, repeat_count=1)
	train(train_ds, valid_ds)