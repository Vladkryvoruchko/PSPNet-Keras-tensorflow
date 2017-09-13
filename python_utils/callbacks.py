import os
import time
import keras.backend as K
from keras.callbacks import Callback, TensorBoard

class CheckPoints(Callback):
  def on_train_begin(self, logs={}):
    self.epoch_nmb = 0
    path = "weights_train"
    if not os.path.exists(path):
        os.makedirs(path)
    self.path_with_time = "{}/{}".format(path, int(round(time.time() * 1000)))
    if not os.path.exists(self.path_with_time):
      os.makedirs(self.path_with_time)
    return
 
  def on_train_end(self, logs={}):
    return

  def on_epoch_begin(self, epoch, logs={}):
    return

  def on_epoch_end(self, epoch, logs={}):
    if self.epoch_nmb % 25 == 0:
      self.model.save("{}/step_{}.h5".format(self.path_with_time, self.epoch_nmb))
    self.epoch_nmb += 1
    return

  def on_batch_begin(self, batch, logs={}):
    return

  def on_batch_end(self, batch, logs={}):
    return

class LrReducer(Callback):
  def __init__(self, max_iter=270000, epoch_size=50, power=0.9, verbose=1):
    super(Callback, self).__init__()
    self.max_iter = max_iter
    self.epoch_size = epoch_size
    self.power = power
    self.verbose = verbose

  def on_epoch_end(self, epoch, logs={}):
    lr = K.get_value(self.model.optimizer.lr)
    new_lr = lr * (1 - epoch * self.epoch_size / float(self.max_iter))**self.power
    K.set_value(self.model.optimizer.lr, new_lr)
    if self.verbose:
        print(" - learning rate: %10f" % (new_lr))

def callbacks(logdir):
  tensorboard_callback = TensorBoard(log_dir=logdir, write_graph=True, write_images=True)
  return [CheckPoints(), tensorboard_callback, LrReducer()]
