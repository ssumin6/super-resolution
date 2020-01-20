![Travis CI](https://travis-ci.com/krasserm/super-resolution.svg?branch=master)

# Single Image Super-Resolution with EDSR, WDSR and SRGAN

A [Tensorflow 2.0](https://www.tensorflow.org/beta) based implementation of

- [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921) (EDSR), winner 
  of the [NTIRE 2017](http://www.vision.ee.ethz.ch/ntire17/) super-resolution challenge.

This is a complete re-write if the old Keras/Tensorflow 1.x based implementation available [here](https://github.com/krasserm/super-resolution/tree/previous).
Some parts are still work in progress but you can already train models as described in the papers via a high-level training 
API. [Training](#training) and [usage](#getting-started) examples are given in the notebooks

- [example-edsr.ipynb](example-edsr.ipynb)

A `DIV2K` [data provider](#div2k-dataset) automatically downloads [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) 
training and validation images of given scale (2, 3, 4 or 8) and downgrade operator ("bicubic", "unknown", "mild" or 
"difficult").

## Environment setup

Create a new [conda](https://conda.io) environment with

    conda env create -f environment.yml
    
and activate it with

    conda activate sisr

## Introduction

You can find an introduction to single-image super-resolution in [this article](https://krasserm.github.io/2019/09/04/super-resolution/). 
It also demonstrates how EDSR and WDSR models can be fine-tuned with SRGAN (see also [this section](#srgan-for-fine-tuning-edsr-and-wdsr-models)).

### EDSR

```python
from model import resolve_single
from model.edsr import edsr

from utils import load_image, plot_sample

model = edsr(scale=4, num_res_blocks=16)
model.load_weights('weights/edsr-16-x4/weights.h5')

lr = load_image('demo/0851x4-crop.png')
sr = resolve_single(model, lr)

plot_sample(lr, sr)
```

![result-edsr](docs/images/result-edsr.png)

## DIV2K dataset

For training and validation on [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) images, applications should use the 
provided `DIV2K` data loader. It automatically downloads DIV2K images to `.div2k` directory and converts them to a 
different format for faster loading.

### Training dataset

```python
from data import DIV2K

train_loader = DIV2K(scale=4,             # 2, 3, 4 or 8
                     downgrade='bicubic', # 'bicubic', 'unknown', 'mild' or 'difficult' 
                     subset='train')      # Training dataset are images 001 - 800
                     
# Create a tf.data.Dataset          
train_ds = train_loader.dataset(batch_size=16,         # batch size as described in the EDSR and WDSR papers
                                random_transform=True, # random crop, flip, rotate as described in the EDSR paper
                                repeat_count=None)     # repeat iterating over training images indefinitely

# Iterate over LR/HR image pairs                                
for lr, hr in train_ds:
    # .... 
```

Crop size in HR images is 96x96. 

### Validation dataset

```python
from data import DIV2K

valid_loader = DIV2K(scale=4,             # 2, 3, 4 or 8
                     downgrade='bicubic', # 'bicubic', 'unknown', 'mild' or 'difficult' 
                     subset='valid')      # Validation dataset are images 801 - 900
                     
# Create a tf.data.Dataset          
valid_ds = valid_loader.dataset(batch_size=1,           # use batch size of 1 as DIV2K images have different size
                                random_transform=False, # use DIV2K images in original size 
                                repeat_count=1)         # 1 epoch
                                
# Iterate over LR/HR image pairs                                
for lr, hr in valid_ds:
    # ....                                 
```

## Training 

The following training examples use the [training and validation datasets](#div2k-dataset) described earlier. The high-level 
training API is designed around *steps* (= minibatch updates) rather than *epochs* to better match the descriptions in the 
papers.

## EDSR

```python
from model.edsr import edsr
from train import EdsrTrainer

# Create a training context for an EDSR x4 model with 16 
# residual blocks.
trainer = EdsrTrainer(model=edsr(scale=4, num_res_blocks=16), 
                      checkpoint_dir=f'.ckpt/edsr-16-x4')
                      
# Train EDSR model for 300,000 steps and evaluate model
# every 1000 steps on the first 10 images of the DIV2K
# validation set. Save a checkpoint only if evaluation
# PSNR has improved.
trainer.train(train_ds,
              valid_ds.take(10),
              steps=300000, 
              evaluate_every=1000, 
              save_best_only=True)
              
# Restore from checkpoint with highest PSNR.
trainer.restore()

# Evaluate model on full validation set.
psnr = trainer.evaluate(valid_ds)
print(f'PSNR = {psnr.numpy():3f}')

# Save weights to separate location.
trainer.model.save_weights('weights/edsr-16-x4/weights.h5')                                    
```

Interrupting training and restarting it again resumes from the latest saved checkpoint. The trained Keras model can be
accessed with `trainer.model`.
