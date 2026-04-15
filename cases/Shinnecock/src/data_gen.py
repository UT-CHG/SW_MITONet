import tensorflow as tf
import numpy as np

class DONDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, b_data, t_data, target, batch_size, shuffle=True):
        """
        Initialization
        """
        self.batch_size = batch_size
        self.sample_size = b_data.shape[0]
        self.target_size = target.shape[0]
        self.b_data = b_data
        self.t_data = t_data
        self.target = target
        self.shuffle = shuffle
        self.indexes = np.arange(self.sample_size)

    def __len__(self):
        """
        Denotes the number of batches per epoch 
        """
        return int(np.ceil(self.sample_size / self.batch_size))

    def __getitem__(self, idx):
        """
        Returns a batch of data
        """
        batch_b_data = self.b_data[idx * self.batch_size : (idx + 1) * self.batch_size,:]
        batch_t_data = self.t_data
        batch_target = self.target[idx * self.batch_size : (idx + 1) * self.batch_size,:]

        ## You should return a tuple from generator/Sequence instance. The first element 
        # of the tuple is a list of input arrays (or just one array if your model has one 
        # input layer), and the second element is a list of output arrays (or just one 
        # array if your model has one output layer).
        return [batch_b_data, batch_t_data], batch_target

    def on_epoch_end(self):
        """
        Updates indexes after each epoch  
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)

class MIONDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, b_par_data, b_ic_data, b_bc_data, t_data, target, batch_size, shuffle=True):
        """
        Initialization
        """
        self.batch_size = batch_size
        self.sample_size = b_par_data.shape[0]
        self.target_size = target.shape[0]
        self.b_par_data = b_par_data
        self.b_ic_data = b_ic_data
        self.b_bc_data = b_bc_data
        self.t_data = t_data
        self.target = target
        self.shuffle = shuffle
        self.indexes = np.arange(self.sample_size)

    def __len__(self):
        """
        Denotes the number of batches per epoch 
        """
        return int(np.ceil(self.sample_size / self.batch_size))

    def __getitem__(self, idx):
        """
        Returns a batch of data
        """
        batch_b_par_data = self.b_par_data[idx * self.batch_size : (idx + 1) * self.batch_size,:]
        batch_b_ic_data = self.b_ic_data[idx * self.batch_size : (idx + 1) * self.batch_size,:]
        batch_b_bc_data = self.b_bc_data[idx * self.batch_size : (idx + 1) * self.batch_size,:]
        batch_t_data = self.t_data
        batch_target = self.target[idx * self.batch_size : (idx + 1) * self.batch_size,:]

        ## You should return a tuple from generator/Sequence instance. The first element 
        # of the tuple is a list of input arrays (or just one array if your model has one 
        # input layer), and the second element is a list of output arrays (or just one 
        # array if your model has one output layer).
        return [batch_b_par_data, batch_b_ic_data, batch_b_bc_data, batch_t_data], batch_target

    def on_epoch_end(self):
        """
        Updates indexes after each epoch  
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)