import tensorflow as tf
import numpy as np

class mito_nn(tf.keras.models.Model):
    def __init__(self, l_factor, latent_dim, branch_par_input_shape, branch_ic_input_shape, branch_bc_input_shape, b_number_layers, b_neurons_layer, b_actf, b_init, b_regularizer, b_encoder_layers, b_encoder_neurons, b_encoder_actf, b_encoder_init, b_encoder_regularizer, trunk_input_shape, t_number_layers, t_neurons_layer, t_actf, t_init, t_regularizer, t_encoder_layers, t_encoder_neurons, t_encoder_actf, t_encoder_init, t_encoder_regularizer, dropout=False, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.dropout = dropout
        self.droput_rate = dropout_rate
        
        if b_regularizer=='none':
            b_regularizer=eval('None')

        if t_regularizer=='none':
            t_regularizer=eval('None')   
            
        if b_encoder_regularizer=='none':
            b_encoder_regularizer=eval('None')  

        if t_encoder_regularizer=='none':
            t_encoder_regularizer=eval('None') 
            
        self.latent_dim = latent_dim
        self.l_factor = l_factor

        self.branch_encoder_par = self.Branch_Encoder(branch_par_input_shape, b_neurons_layer, b_encoder_layers, b_encoder_neurons, b_encoder_actf, b_encoder_init, b_encoder_regularizer)
        
        self.branch_encoder_ic = self.Branch_Encoder(branch_ic_input_shape, b_neurons_layer, b_encoder_layers, b_encoder_neurons, b_encoder_actf, b_encoder_init, b_encoder_regularizer)

        self.branch_encoder_bc = self.Branch_Encoder(branch_bc_input_shape, b_neurons_layer, b_encoder_layers, b_encoder_neurons, b_encoder_actf, b_encoder_init, b_encoder_regularizer)
        
        self.trunk_encoder = self.Trunk_Encoder(trunk_input_shape, t_neurons_layer, t_encoder_layers, t_encoder_neurons, t_encoder_actf, t_encoder_init, t_encoder_regularizer)

        self.branch_par = self.MLP_Branch_PAR(branch_par_input_shape, trunk_input_shape, l_factor*latent_dim, b_number_layers, b_neurons_layer, b_actf, b_init, b_regularizer)
        
        self.branch_ic = self.MLP_Branch_IC(branch_ic_input_shape, trunk_input_shape, l_factor*latent_dim, b_number_layers, b_neurons_layer, b_actf, b_init, b_regularizer)

        self.branch_bc = self.MLP_Branch_BC(branch_bc_input_shape, trunk_input_shape, l_factor*latent_dim, b_number_layers, b_neurons_layer, b_actf, b_init, b_regularizer)
        
        self.trunk = self.MLP_Trunk(branch_par_input_shape, branch_ic_input_shape, branch_bc_input_shape, trunk_input_shape, l_factor*latent_dim, t_number_layers, t_neurons_layer, t_actf, t_init, t_regularizer)
        
        self.b0 = tf.Variable(tf.zeros(latent_dim,dtype=tf.float32), shape=tf.TensorShape(latent_dim), name='b0', dtype=tf.float32)   


    def Branch_Encoder(self, input_shape, neurons_layer, encoder_layers, encoder_neurons, actf, init, regularizer):
        input_layer = tf.keras.layers.Input(input_shape)
        encoder = input_layer
        for e in range(encoder_layers):
            encoder = tf.keras.layers.Dense(encoder_neurons, activation=actf, kernel_initializer=init, kernel_regularizer=regularizer)(encoder)
            if self.dropout==True: 
                encoder = tf.keras.layers.Dropout(self.dropout_rate)(encoder)
        output_encoder = tf.keras.layers.Dense(neurons_layer, activation=actf, kernel_initializer=init, kernel_regularizer=regularizer)(encoder)
        model = tf.keras.Model(input_layer,output_encoder)
        return model

    def Trunk_Encoder(self, input_shape, neurons_layer, encoder_layers, encoder_neurons, actf, init, regularizer):
        input_layer = tf.keras.layers.Input(input_shape)
        encoder = input_layer
        for e in range(encoder_layers):
            encoder = tf.keras.layers.Dense(encoder_neurons, activation=actf, kernel_initializer=init, kernel_regularizer=regularizer)(encoder)
            if self.dropout==True: 
                encoder = tf.keras.layers.Dropout(self.dropout_rate)(encoder)            
        output_encoder = tf.keras.layers.Dense(neurons_layer, activation=actf, kernel_initializer=init,kernel_regularizer=regularizer)(encoder)
        model = tf.keras.Model(input_layer,output_encoder)
        return model
            
    def MLP_Branch_PAR(self, branch_par_input_shape, trunk_input_shape, output_shape, number_layers, neurons_layer, actf, init, regularizer):
        branch_par_input_layer = tf.keras.layers.Input(branch_par_input_shape,name='branch_par_input')
        
        trunk_input_layer = tf.keras.layers.Input(trunk_input_shape,name='trunk_input')
        x = branch_par_input_layer
        for i in range(number_layers):
            x = tf.keras.layers.Dense(neurons_layer,activation=actf,kernel_initializer=init,kernel_regularizer=regularizer,name='branch_hidden'+str(i))(x)
            ones = tf.ones_like(x)
            x1 = tf.keras.layers.Subtract(name='subtract_branch'+str(i))([ones,x])
            x2 = tf.keras.layers.Multiply(name='multiply_branch_encoder'+str(i))([x,self.branch_encoder_par(branch_par_input_layer)])
            x3 = tf.keras.layers.Multiply(name='multiply_trunk_encoder'+str(i))([x1,self.trunk_encoder(trunk_input_layer)])  
            x = tf.keras.layers.Add(name='add_product'+str(i))([x2,x3])
            if self.dropout==True: 
                x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        output_layer = tf.keras.layers.Dense(output_shape,kernel_initializer=init,kernel_regularizer=regularizer,name='output_branch')(x)
        model = tf.keras.Model([branch_par_input_layer, trunk_input_layer],output_layer)
        return model

    def MLP_Branch_IC(self, branch_ic_input_shape, trunk_input_shape, output_shape, number_layers, neurons_layer, actf, init, regularizer):
        
        branch_ic_input_layer = tf.keras.layers.Input(branch_ic_input_shape,name='branch_ic_input')
        
        trunk_input_layer = tf.keras.layers.Input(trunk_input_shape,name='trunk_input')
        x = branch_ic_input_layer
        for i in range(number_layers):
            x = tf.keras.layers.Dense(neurons_layer,activation=actf,kernel_initializer=init,kernel_regularizer=regularizer,name='branch_hidden'+str(i))(x)
            ones = tf.ones_like(x)
            x1 = tf.keras.layers.Subtract(name='subtract_branch'+str(i))([ones,x])
            x2 = tf.keras.layers.Multiply(name='multiply_branch_encoder'+str(i))([x,self.branch_encoder_ic(branch_ic_input_layer)])
            x3 = tf.keras.layers.Multiply(name='multiply_trunk_encoder'+str(i))([x1,self.trunk_encoder(trunk_input_layer)])
            x = tf.keras.layers.Add(name='add_product'+str(i))([x2,x3])
            if self.dropout==True:
                x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        output_layer = tf.keras.layers.Dense(output_shape,kernel_initializer=init,kernel_regularizer=regularizer,name='output_branch')(x)
        model = tf.keras.Model([branch_ic_input_layer, trunk_input_layer],output_layer)
        return model

    def MLP_Branch_BC(self, branch_bc_input_shape, trunk_input_shape, output_shape, number_layers, neurons_layer, actf, init, regularizer):

        branch_bc_input_layer = tf.keras.layers.Input(branch_bc_input_shape,name='branch_bc_input')
        trunk_input_layer = tf.keras.layers.Input(trunk_input_shape,name='trunk_input')
        x = branch_bc_input_layer
        for i in range(number_layers):
            x = tf.keras.layers.Dense(neurons_layer,activation=actf,kernel_initializer=init,kernel_regularizer=regularizer,name='branch_hidden'+str(i))(x)
            ones = tf.ones_like(x)
            x1 = tf.keras.layers.Subtract(name='subtract_branch'+str(i))([ones,x])
            x2 = tf.keras.layers.Multiply(name='multiply_branch_encoder'+str(i))([x,self.branch_encoder_bc(branch_bc_input_layer)])
            x3 = tf.keras.layers.Multiply(name='multiply_trunk_encoder'+str(i))([x1,self.trunk_encoder(trunk_input_layer)])
            x = tf.keras.layers.Add(name='add_product'+str(i))([x2,x3])
            if self.dropout==True:
                x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        output_layer = tf.keras.layers.Dense(output_shape,kernel_initializer=init,kernel_regularizer=regularizer,name='output_branch')(x)
        model = tf.keras.Model([branch_bc_input_layer, trunk_input_layer],output_layer)
        return model    

    def MLP_Trunk(self, branch_par_input_shape, branch_ic_input_shape, branch_bc_input_shape, trunk_input_shape, output_shape, number_layers, neurons_layer, actf, init, regularizer):
        branch_par_input_layer = tf.keras.layers.Input(branch_par_input_shape,name='branch_par_input')
        branch_ic_input_layer = tf.keras.layers.Input(branch_ic_input_shape,name='branch_ic_input')
        branch_bc_input_layer = tf.keras.layers.Input(branch_bc_input_shape,name='branch_bc_input')
        trunk_input_layer = tf.keras.layers.Input(trunk_input_shape,name='trunk_input')
        x = trunk_input_layer
        for i in range(number_layers):
            x = tf.keras.layers.Dense(neurons_layer,activation=actf,kernel_initializer=init,kernel_regularizer=regularizer,name='branch_hidden'+str(i))(x)
            ones = tf.ones_like(x)
            x1 = tf.keras.layers.Subtract(name='subtract_branch'+str(i))([ones,x])
            x2 = tf.keras.layers.Multiply(name='multiply_branch_encoder'+str(i))([x,self.branch_encoder_par(branch_par_input_layer),self.branch_encoder_ic(branch_ic_input_layer),self.branch_encoder_bc(branch_bc_input_layer)])
            x3 = tf.keras.layers.Multiply(name='multiply_trunk_encoder'+str(i))([x1,self.trunk_encoder(trunk_input_layer)])  
            x = tf.keras.layers.Add(name='add_product'+str(i))([x2,x3])
            if self.dropout==True: 
                x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        output_layer = tf.keras.layers.Dense(output_shape, activation=actf, kernel_initializer=init,kernel_regularizer=regularizer,name='output_trunk')(x)

        model = tf.keras.Model([branch_par_input_layer, branch_ic_input_layer,branch_bc_input_layer, trunk_input_layer],output_layer)
        return model
        
    def call(self,data):
        par, ic, bc, t = data
        br_par = self.branch_par([par,t])
        br_ic = self.branch_ic([ic,t])
        br_bc = self.branch_bc([bc,t])
        branch = br_par*br_ic*br_bc
        br = tf.reshape(branch, [-1, self.latent_dim, self.l_factor])# automate the m and p shapes
        tr = tf.reshape(self.trunk(data), [-1, self.latent_dim, self.l_factor])
        onet = tf.einsum('ijk,ijk->ij', br, tr) + self.b0
#         onet = tf.reduce_sum(self.branch(data)*self.trunk(data), axis=1, keepdims=True) + self.b0
        return onet
    
class mito_model(tf.keras.Model):
    def __init__(self, model):
        super(mito_model, self).__init__()
        self.model = model

    def compile(self, optimizer, loss_fn):
        super(mito_model, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def save(self,path,id_b=None):
        tf.keras.models.save_model(self.model,path)
        
        if id_b is not None:
            np.save(path+'/branch_id',id_b)
        
    def call(self, dataset):

        return self.model(dataset)
    
    def train_step(self, dataset):
        
        self.b_par_input, self.b_ic_input, self.b_bc_input, self.t_input, self.target = dataset
        
        with tf.GradientTape() as tape:
            o_res = self.model([self.b_par_input, self.b_ic_input, self.b_bc_input, self.t_input], training=True)
            reg_loss = tf.add_n(self.losses) if self.losses else 0.0
            main_loss = self.loss_fn(o_res, self.target)
            loss = main_loss + reg_loss
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return {"loss": main_loss}

    def test_step(self, dataset):

        self.b_par_input, self.b_ic_input, self.b_bc_input, self.t_input, self.target = dataset

        val = self.model([self.b_par_input, self.b_ic_input, self.b_bc_input, self.t_input], training=True)
        val_loss = self.loss_fn(val, self.target)

        return {"loss": val_loss}
