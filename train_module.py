import timeit
import tensorflow as tf

from model import evaluate

from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam

### should modify train, fit

class Trainer:
    def __init__(self,
                 model,
                 learning_rate,
                 checkpoint_dir='./ckpt/'):

        self.now = None
        self.loss = MeanAbsoluteError()
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate),
                                              model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)
        self.restore()

    @property
    def model(self):
        return self.checkpoint.model

    def train(self, train_dataset, valid_dataset, steps, evaluate_every=1000, save_best_only=False):
        loss_mean = Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        self.now = timeit.default_timer()

        for lr, hr in train_dataset.take(steps - ckpt.step.numpy()):
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()    

            loss = self.train_step(lr, hr)
            loss_mean(loss)

            if step % evaluate_every == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_states()

                # Compute PSNR on validation dataset
                psnr_value, ssim_value = self.evaluate(valid_dataset)

                duration = timeit.default_timer() - self.now
                print('%d/%d: loss = %.3f, PSNR = %3f (%.2fs)' %(step, steps, loss_value.numpy(),psnr_value.numpy(), duration))

                if save_best_only and psnr_value <= ckpt.psnr:
                    self.now = timeit.timeit()
                    # skip saving checkpoint, no PSNR improvement
                    continue

                ckpt.psnr = psnr_value
                ckpt_mgr.save()

                self.now = timeit.timeit()

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint.model(lr, training=True)
            loss_value = self.loss(hr, sr)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_weights)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_weights))

        return loss_value

    def evaluate(self, dataset):
        return evaluate(self.checkpoint.model, dataset)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print('Model restored from checkpoint at step %d.' %self.checkpoint.step.numpy())

