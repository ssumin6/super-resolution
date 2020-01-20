import os
import tensorflow as tf
from PIL import Image

def augment(lr_dir, hr_dir, scale= 4, patch_size = 64):
	lr_aug_dir = '../image_aug_240'
	hr_aug_dir = '../image_aug_960'

	if (not os.path.exists(lr_aug_dir)):
		os.makedirs(lr_aug_dir)

	if (not os.path.exists(hr_aug_dir)):
		os.makedirs(hr_aug_dir)

	nImages = [i*30 for i in range(60)]

	lr_crop_size = 64
	hr_crop_size = 256

	cnt = 0

	for number in nImages:
		img = Image.open(os.path.join(lr_dir, 'frame%04d.jpg' %number))
		print(img.size)

		tmp = cnt
		
		for i in range(0, img.size[0], lr_crop_size):
			for j in range(0, img.size[1], lr_crop_size):
				crop = img.crop((i, j, i+lr_crop_size, j+lr_crop_size))
				file_name = 'frame%05d.jpg' %cnt
				crop.save(os.path.join(lr_aug_dir, file_name))
				cnt += 1

		img = Image.open(os.path.join(hr_dir, 'frame%04d.jpg' %number))
		print(img.size)

		cnt = tmp

		for i in range(0, img.size[0], hr_crop_size):
			for j in range(0, img.size[1], hr_crop_size):
				#if (i+lr_crop_size > img.size[0]):
				crop = img.crop((i, j, i+hr_crop_size, j+hr_crop_size))
				file_name = 'frame%05d.jpg' %cnt
				crop.save(os.path.join(hr_aug_dir, file_name))
				cnt += 1

	total_images = cnt
	print('total_images', total_images)
 	
	#img = Image.open(os.path.join(lr_aug_dir, 'frame%05d.jpg' % 0))
	#print(img.size


def random(nImage):
	cnt = nImage
	lr_aug_dir = '../image_aug_240'
	hr_aug_dir = '../image_aug_960'

	for i in range(nImage):
		file_name = 'frame%05d.jpg' %i
		img = Image.open(os.path.join(lr_aug_dir, file_name))
		ro1 = img.rotate(90) ##90 rotate
		ro1.save(os.path.join(lr_aug_dir, 'frame%05d.jpg' %cnt))
		cnt += 1
		ro2 = ro1.rotate(90) ##180 rotate
		ro2.save(os.path.join(lr_aug_dir, 'frame%05d.jpg' %cnt))
		cnt += 1
		ro3 = ro2.rotate(90) ##270 rotate
		ro3.save(os.path.join(lr_aug_dir, 'frame%05d.jpg' %cnt))
		cnt += 1
		flip = img.transpose(Image.FLIP_LEFT_RIGHT)
		flip.save(os.path.join(lr_aug_dir, 'frame%05d.jpg' %cnt))
		cnt += 1
		flip = ro1.transpose(Image.FLIP_LEFT_RIGHT)
		flip.save(os.path.join(lr_aug_dir, 'frame%05d.jpg' %cnt))
		cnt += 1
		flip = ro2.transpose(Image.FLIP_LEFT_RIGHT)
		flip.save(os.path.join(lr_aug_dir, 'frame%05d.jpg' %cnt))
		cnt += 1
		flip = ro3.transpose(Image.FLIP_LEFT_RIGHT)
		flip.save(os.path.join(lr_aug_dir, 'frame%05d.jpg' %cnt))
		cnt += 1

	cnt = nImage
	for i in range(nImage):
		file_name = 'frame%05d.jpg' %i
		img = Image.open(os.path.join(hr_aug_dir, file_name))
		ro1 = img.rotate(90) ##90 rotate
		ro1.save(os.path.join(hr_aug_dir, 'frame%05d.jpg' %cnt))
		cnt += 1
		ro2 = ro1.rotate(90) ##180 rotate
		ro2.save(os.path.join(hr_aug_dir, 'frame%05d.jpg' %cnt))
		cnt += 1
		ro3 = ro2.rotate(90) ##270 rotate
		ro3.save(os.path.join(hr_aug_dir, 'frame%05d.jpg' %cnt))
		cnt += 1
		flip = img.transpose(Image.FLIP_LEFT_RIGHT)
		flip.save(os.path.join(hr_aug_dir, 'frame%05d.jpg' %cnt))
		cnt += 1
		flip = ro1.transpose(Image.FLIP_LEFT_RIGHT)
		flip.save(os.path.join(hr_aug_dir, 'frame%05d.jpg' %cnt))
		cnt += 1
		flip = ro2.transpose(Image.FLIP_LEFT_RIGHT)
		flip.save(os.path.join(hr_aug_dir, 'frame%05d.jpg' %cnt))
		cnt += 1
		flip = ro3.transpose(Image.FLIP_LEFT_RIGHT)
		flip.save(os.path.join(hr_aug_dir, 'frame%05d.jpg' %cnt))
		cnt += 1
	print('cnt', cnt)


if __name__ == '__main__':
	lr_dir = '../image_240'
	hr_dir = '../image_960'
	#augment(lr_dir, hr_dir, scale = 4)
	random(1680)