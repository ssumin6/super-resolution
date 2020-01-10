import os
import tensorflow as tf

from tensorflow.python.data.experimental import AUTOTUNE

class video_ds:
    def __init__(self,
                 subset='train',
                 lr_dir='../image_240',
                 hr_dir='../image_960',
                 caches_dir='../caches'):

        self.scale = 4

        if subset == 'train':
            self.image_ids = [i*30 for i in range(0,50)]
        elif subset == 'valid':
            self.image_ids = [i*30 for i in range(50,60)]
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
        if random_transform:
            ds = ds.map(lambda lr, hr: random_crop(lr, hr, scale=self.scale), num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def hr_dataset(self):
        #ds = self._images_dataset(self._hr_image_files()).cache(self._hr_cache_file())
        ds = self._images_dataset(self._hr_image_files())

        #if not os.path.exists(self._hr_cache_index()):
        #    self._populate_cache(ds, self._hr_cache_file())

        return ds

    def lr_dataset(self):
        #ds = self._images_dataset(self._lr_image_files()).cache(self._lr_cache_file())
        ds = self._images_dataset(self._lr_image_files())

        #if not os.path.exists(self._lr_cache_index()):
        #    self._populate_cache(ds, self._lr_cache_file())

        return ds

    def _hr_image_files(self):
        images_dir = self._hr_images_dir
        return [os.path.join(images_dir, 'frame%04d.jpg' % image_id) for image_id in self.image_ids]

    def _lr_image_files(self):
        images_dir = self._lr_images_dir
        return [os.path.join(images_dir, 'frame%04d.jpg' %image_id) for image_id in self.image_ids]

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


# -----------------------------------------------------------
#  Transformations
# -----------------------------------------------------------


def random_crop(lr_img, hr_img, hr_crop_size=96, scale=2):
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))


def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)


'''
    def _hr_cache_file(self):
        return os.path.join(self.caches_dir, 'video_HR.cache')

    def _lr_cache_file(self):
        return os.path.join(self.caches_dir, 'video_LR.cache')

    def _hr_cache_index(self):
        path = self._hr_cache_file()
        return os.path.join(path, 'index')

    def _lr_cache_index(self):
        path = self._lr_cache_file()
        return os.path.join(path, 'index')
'''