import os
from model.edsr import edsr
from train-module import EdsrTrainer

# super resolution factor 
scale = 4
# depth of residual blocks
depth = 20
# number of channel of conv blocks
channels = 32


def train():
	model = EdsrTrainer(model = edsr(scale = scale, num_res_blocks= depth), checkpoint_dir=f'.ckpt/edsr-{depth}-x{scale}')


if __name__ == '__main__':
	train()