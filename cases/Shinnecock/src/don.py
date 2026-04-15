import tensorflow as tf
tf.keras.backend.set_floatx('float32') 
import numpy as np

class don_nn(tf.keras.models.Model):
    def __init__(self, branch_input_shape, b_number_layers, b_neurons_layer, branch_output_shape, b_actf, b_init, b_regularizer, trunk_input_shape, t_number_layers, t_neurons_layer, trunk_output_shape, t_actf, t_init, t_regularizer, dropout=False, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        
        if b_regularizer=='none':
            b_regularizer=eval('None')

        if t_regularizer=='none':
            t_regularizer=eval('None')    

        self.branch = self.MLP_Branch(branch_input_shape, branch_output_shape, b_number_layers, b_neurons_layer, b_actf, b_init, b_regularizer)

        
        self.trunk = self.MLP_Trunk(trunk_input_shape, trunk_output_shape, t_number_layers, t_neurons_layer, t_actf, t_init, t_regularizer)
        
        self.b0 = tf.Variable(tf.zeros(1,dtype=tf.float32), shape=tf.TensorShape(1), name='b0', dtype=tf.float32)   
            
    def MLP_Branch(self, branch_input_shape, branch_output_shape, number_layers, neurons_layer, actf, init, regularizer):
        branch_input_layer = tf.keras.layers.Input(branch_input_shape,name='branch_input')
        
        x = branch_input_layer
        for i in range(number_layers):
            x = tf.keras.layers.Dense(neurons_layer,activation=actf,kernel_initializer=init,kernel_regularizer=regularizer,name='branch_hidden'+str(i))(x)
            if self.dropout==True: 
                x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        output_layer = tf.keras.layers.Dense(branch_output_shape,kernel_initializer=init,kernel_regularizer=regularizer,name='output_branch')(x)
        model = tf.keras.Model(branch_input_layer,output_layer)
        return model

    def MLP_Trunk(self, trunk_input_shape, trunk_output_shape, number_layers, neurons_layer, actf, init, regularizer):
        trunk_input_layer = tf.keras.layers.Input((trunk_input_shape),name='trunk_input')
        x = trunk_input_layer
        for i in range(number_layers):
            x = tf.keras.layers.Dense(neurons_layer,activation=actf,kernel_initializer=init,kernel_regularizer=regularizer,name='branch_hidden'+str(i))(x)
            if self.dropout==True: 
                x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        output_layer = tf.keras.layers.Dense(trunk_output_shape, activation=actf, kernel_initializer=init,kernel_regularizer=regularizer,name='output_trunk')(x)

        model = tf.keras.Model(trunk_input_layer,output_layer)
        return model
        
    def call(self,data):
        b, t = data
        onet = tf.einsum('ip,jp->ij', self.branch(b), self.trunk(t)) + self.b0
#         onet = tf.reduce_sum(self.branch(data)*self.trunk(data), axis=1, keepdims=True) + self.b0
        return onet
    
class don_model(tf.keras.Model):
    def __init__(self, model):
        super(don_model, self).__init__()
        self.model = model

    def compile(self, optimizer, loss_fn):
        super(don_model, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def save(self,path,id_b=None):
        tf.keras.models.save_model(self.model,path)
        
        if id_b is not None:
            np.save(path+'/branch_id',id_b)
        
    def call(self, dataset):

        return self.model(dataset)
    
    def train_step(self, dataset):
        
        [self.b_input, self.t_input], self.target = dataset
        
        with tf.GradientTape() as tape:
            o_res = self.model([self.b_input, self.t_input], training=True)
            reg_loss = tf.add_n(self.losses) if self.losses else 0.0
            main_loss = self.loss_fn(o_res, self.target)
            loss = main_loss + reg_loss
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return {"loss": main_loss}

    def test_step(self, dataset):

        [self.b_input, self.t_input], self.target = dataset

        val = self.model([self.b_input, self.t_input], training=True)
        val_loss = self.loss_fn(val, self.target)

        return {"loss": val_loss}
