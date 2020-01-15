import os
import tensorflow as tf

from tensorflow.python.data.experimental import AUTOTUNE

class video_ds:
    def __init__(self,
                 subset='train',
                 lr_dir='../image_aug_240',
                 hr_dir='../image_aug_960',
                 caches_dir='../caches'):

        self.scale = 4

        if subset == 'train':
            self.image_ids = [i for i in range(0,6720)]
        elif subset == 'valid':
            self.image_ids = [i for i in range(6720,8400)]
        else:
            raise ValueError("subset must be 'train' or 'valid'")

        self.subset = subset
        self._lr_images_dir = lr_dir
        self._hr_images_dir = hr_dir
        self.caches_dir = caches_dir

        if (not os.path.exists(caches_dir)):
            os.makedirs(caches_dir)

    def __len__(self):
        return len(self.image_ids)

    def dataset(self, batch_size=64, repeat_count=None, random_transform=True):
        ds = tf.data.Dataset.zip((self.lr_dataset(), self.hr_dataset()))
        ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def hr_dataset(self):
        ds = self._images_dataset(self._hr_image_files())
        return ds

    def lr_dataset(self):
        ds = self._images_dataset(self._lr_image_files())
        return ds

    def _hr_image_files(self):
        images_dir = self._hr_images_dir
        return [os.path.join(images_dir, 'frame%05d.jpg' % image_id) for image_id in self.image_ids]

    def _lr_image_files(self):
        images_dir = self._lr_images_dir
        return [os.path.join(images_dir, 'frame%05d.jpg' %image_id) for image_id in self.image_ids]

    @staticmethod
    def _images_dataset(image_files):
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
        return ds

    @staticmethod
    def _populate_cache(ds, cache_file):
        print('Caching decoded images in %s ...' % cache_file)
        for _ in ds: pass
        print('Cached decoded images in %s.' % cache_file)
