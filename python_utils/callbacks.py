import os
import time
import keras.backend as K
from keras.callbacks import Callback, TensorBoard, ReduceLROnPlateau, ModelCheckpoint

class LrReducer(Callback):
  def __init__(self, base_lr = 0.01, max_epoch = 150, power=0.9, verbose=1):
    super(Callback, self).__init__()
    self.max_epoch = max_epoch
    self.power = power
    self.verbose = verbose
    self.base_lr = base_lr

  def on_epoch_end(self, epoch, logs={}):
    lr_now = K.get_value(self.model.optimizer.lr)
    new_lr = max(0.00001, min(self.base_lr * (1 - epoch / float(self.max_epoch))**self.power, lr_now))
    K.set_value(self.model.optimizer.lr, new_lr)
    if self.verbose:
        print(" - learning rate: %10f" % (new_lr))

def callbacks(logdir):
  model_checkpoint = ModelCheckpoint("weights_train/weights.{epoch:02d}-{loss:.2f}.h5", monitor='loss', verbose=1, period=10) 
  tensorboard_callback = TensorBoard(log_dir=logdir, write_graph=True, write_images=True, histogram_freq=1)
  plateau_callback = ReduceLROnPlateau(monitor='loss', factor=0.99, verbose=1, patience=0, min_lr=0.00001) 
  #return [CheckPoints(), tensorboard_callback, LrReducer()]
  return [model_checkpoint, tensorboard_callback, plateau_callback, LrReducer()]
