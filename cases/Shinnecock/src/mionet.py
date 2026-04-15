import tensorflow as tf
tf.keras.backend.set_floatx('float32') 
import numpy as np

class mio_nn(tf.keras.models.Model):
    def __init__(self, branch_par_input_shape, branch_ic_input_shape, branch_bc_input_shape, b_number_layers, branch_output_shape, b_neurons_layer, b_actf, b_init, b_regularizer, trunk_input_shape, t_number_layers, trunk_output_shape, t_neurons_layer, t_actf, t_init, t_regularizer, dropout=False, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        
        if b_regularizer=='none':
            b_regularizer=eval('None')

        if t_regularizer=='none':
            t_regularizer=eval('None')   

        self.branch_par = self.MLP_Branch(branch_par_input_shape, branch_output_shape, b_number_layers, b_neurons_layer, b_actf, b_init, b_regularizer)
        
        self.branch_ic = self.MLP_Branch(branch_ic_input_shape, branch_output_shape, b_number_layers, b_neurons_layer, b_actf, b_init, b_regularizer)

        self.branch_bc = self.MLP_Branch(branch_bc_input_shape, branch_output_shape, b_number_layers, b_neurons_layer, b_actf, b_init, b_regularizer)
        
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
        trunk_input_layer = tf.keras.layers.Input(trunk_input_shape,name='trunk_input')
        x = trunk_input_layer
        for i in range(number_layers):
            x = tf.keras.layers.Dense(neurons_layer,activation=actf,kernel_initializer=init,kernel_regularizer=regularizer,name='branch_hidden'+str(i))(x)
            if self.dropout==True: 
                x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        output_layer = tf.keras.layers.Dense(trunk_output_shape, activation=actf, kernel_initializer=init,kernel_regularizer=regularizer,name='output_trunk')(x)

        model = tf.keras.Model(trunk_input_layer,output_layer)
        return model
        
    def call(self,data):
        par, ic, bc, t = data
        br_par = self.branch_par(par)
        br_ic = self.branch_ic(ic)
        br_bc = self.branch_bc(bc)
        branch = br_par*br_ic*br_bc
        onet = tf.einsum('ip,jp->ij', branch, self.trunk(t)) + self.b0
        return onet
    
class mio_model(tf.keras.Model):
    def __init__(self, model):
        super(mio_model, self).__init__()
        self.model = model

    def compile(self, optimizer, loss_fn):
        super(mio_model, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def save(self,path,id_b=None):
        tf.keras.models.save_model(self.model,path)
        
        if id_b is not None:
            np.save(path+'/branch_id',id_b)
        
    def call(self, dataset):

        return self.model(dataset)
    
    def train_step(self, dataset):
        
        [self.b_par_input, self.b_ic_input, self.b_bc_input, self.t_input], self.target = dataset
        
        with tf.GradientTape() as tape:
            o_res = self.model([self.b_par_input, self.b_ic_input, self.b_bc_input, self.t_input], training=True)
            reg_loss = tf.add_n(self.losses) if self.losses else 0.0
            main_loss = self.loss_fn(o_res, self.target)
            loss = main_loss + reg_loss
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return {"loss": main_loss}

    def test_step(self, dataset):

        [self.b_par_input, self.b_ic_input, self.b_bc_input, self.t_input], self.target = dataset

        val = self.model([self.b_par_input, self.b_ic_input, self.b_bc_input, self.t_input], training=True)
        val_loss = self.loss_fn(val, self.target)

        return {"loss": val_loss}
