from keras.callbacks import Callback
import tensorflow.python.keras.backend as K


class LrLogger(Callback):
    def on_train_begin(self, logs=None):
        self.lrs = []
        
    def on_epoch_end(self, epoch, logs=None):
        lr = float(K.get_value(self.model.optimizer.learning_rate))
        self.lrs.append(lr)