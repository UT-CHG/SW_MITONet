#Load module
import os
import sys
import time
from datetime import datetime
import netCDF4 as nc
from tqdm import tqdm
from importlib import reload as reload
import numpy as np
import joblib

import tensorflow as tf
tf.keras.backend.set_floatx('float32') 

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

from pathlib import Path
try:
    work_dir.exists()
except NameError:
    curr_dir = Path().resolve()
    work_dir = curr_dir.parent  

scripts_dir = work_dir / "src"
settings_dir = work_dir / "settings"
sys.path.append(str(scripts_dir.absolute()))
sys.path.append(str(settings_dir.absolute()))

ae_model_dir = "saved_models_ldon"
ldon_model_dir = "saved_models_ldon"
scalers_dir = "scalers_ldon"

#Load custom modules
import ldon
import autoencoder as ae
import data_loader as dl 
import settings_ldon as sett
import processing_utils as pu
from data_gen import DONDataGenerator

data_dir = Path(sett.data_dir)

#Epochs and trials
ldon_epochs = sett.ldon_epochs
ae_epochs = sett.ae_epochs

#Data
key = sett.key
train_day0 = sett.day0
train_day1 = sett.day1
train_days = [train_day0, train_day1]
val_day0 = sett.val_day0
val_day1 = sett.val_day1
val_days = [val_day0, val_day1]
test_day0 = sett.test_day0
test_day1 = sett.test_day1
test_days = [test_day0, test_day1]
param_list = sett.param_list
param_train = sett.param_train
param_val = sett.param_val
param_test = sett.param_test

scaling = sett.scaling

Np_train = len(param_train)
Np_val = len(param_val)
Np_test = len(param_test)

mesh = dl.load_mesh(data_dir/"cf0025")
train_snaps = [np.where(mesh['t']/60/60/24==train_days[0])[0][0], np.where(mesh['t']/60/60/24==train_days[1])[0][0]]
Nt_snaps_train = mesh['t'][train_snaps[0]:train_snaps[1]].shape[0]
val_snaps = [np.where(mesh['t']/60/60/24==val_days[0])[0][0], np.where(mesh['t']/60/60/24==val_days[1])[0][0]]
Nt_snaps_val = mesh['t'][val_snaps[0]:val_snaps[1]].shape[0]
test_snaps = [np.where(mesh['t']/60/60/24==test_days[0])[0][0], np.where(mesh['t']/60/60/24==test_days[1])[0][0]]
Nt_snaps_test = mesh['t'][test_snaps[0]:test_snaps[1]].shape[0]

data_dict = {}
for inx, param in enumerate(param_list):
    dirname = 'cf'+str(param).split('.')[1];
    data_dict[inx] = dl.load_variables(data_dir/dirname)

Nn = mesh['nodes'].shape[0]
Ne = mesh['triangles'].shape[0]
Nt = mesh['t'].shape[0]

print(f"Loaded u,v,h simulation output for {len(param_list)} parameter values at {Nn} nodes and {Nt} time steps")

### Prepare Autoencoder input data 
train_data = np.empty((0, Nn ),)  ## Augment snapshots with parameter value, if needed
val_data = np.empty((0, Nn ),)
test_data = np.empty((0, Nn ),)

for inx,param in enumerate(param_train):
    indx = param_list.index(param)
    snap = data_dict[indx][key].data.T 
    train_data = np.vstack((train_data, snap))
    
validation_data = True
if validation_data:
    
    for inx,param in enumerate(param_val):
        indx = param_list.index(param)
        val_snap = data_dict[indx][key].data.T
        val_data = np.vstack((val_data, val_snap))

    for inx,param in enumerate(param_test):
        indx = param_list.index(param)
        test_snap = data_dict[indx][key].data.T
        test_data = np.vstack((test_data, test_snap))

print("Split data into training, validation, and testing for:",key)

### CROP
train_data_resh = np.reshape(train_data,(Np_train,Nt,Nn))
val_data_resh = np.reshape(val_data,(Np_val,Nt,Nn))
test_data_resh = np.reshape(test_data,(Np_test,Nt,Nn))

train_data_crop = train_data_resh[:,train_snaps[0]:train_snaps[1],:]
train_data = np.reshape(train_data_crop,(Np_train*Nt_snaps_train,Nn))
train_data_resh = np.reshape(train_data,(Np_train,Nt_snaps_train,Nn))

val_data_crop = val_data_resh[:,val_snaps[0]:val_snaps[1],:]
val_data = np.reshape(val_data_crop,(Np_val*Nt_snaps_val,Nn))
val_data_resh = np.reshape(val_data,(Np_val,Nt_snaps_val,Nn))

test_data_crop = test_data_resh[:,test_snaps[0]:test_snaps[1],:]
test_data = np.reshape(test_data_crop,(Np_test*Nt_snaps_test,Nn))
test_data_resh = np.reshape(test_data,(Np_test,Nt_snaps_test,Nn))

Nt_train = train_data_resh.shape[1]
Nt_val = val_data_resh.shape[1]
Nt_test = test_data_resh.shape[1]

### SCALE
if scaling is True:
    scalers_dir = Path(scalers_dir)
    scalers_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Fit the scaler using the adjusted min and max
    scaler.fit(train_data)

    joblib.dump(scaler, str(scalers_dir)+'/ae_scaler_'+str(key)+'.save')

    # Transform the train, validation, and test data
    train_data_scaled = scaler.transform(train_data)
    val_data_scaled = scaler.transform(val_data)
    test_data_scaled = scaler.transform(test_data)

    # Reshape scaled data back to the original reshaped form
    train_data_scaled_resh = np.reshape(train_data_scaled, (Np_train, Nt_train, Nn))
    val_data_scaled_resh = np.reshape(val_data_scaled, (Np_val, Nt_val, Nn))
    test_data_scaled_resh = np.reshape(test_data_scaled, (Np_test, Nt_test, Nn))

### TRAIN

def NN():
    # Define search space
    verbosity_mode = 1

    number_layers = sett.ae_number_layers
    latent_dim = sett.latent_dim
    init_lr = sett.ae_init_lr
    enc_act = sett.enc_act
    dec_act = sett.dec_act

    set_opt = ae.Optimizer(lr=init_lr)
    optimizerr = "Adam"

    size = np.zeros(number_layers,dtype=int)
    for i in range(number_layers):
        if i==0:
            size[i] = int(Nn)
        else:
            size[i] = int(size[i-1]/2)
    
    model = ae.Autoencoder(latent_dim, enc_act, dec_act, size, )

    model.compile(optimizer = set_opt.get_opt(opt=optimizerr), 
              loss_fn = tf.keras.losses.MeanSquaredError(),
             )
    
    return model


## Define minibatch generators for training and validation using Tensorflow Dataset API
size_buffer = Nt_train 
    
train_ds, val_ds = ae.gen_batch_ae(train_data_scaled, val_data_scaled,  
                                 batch_size=sett.ae_batch_size, shuffle_buffer=size_buffer)

ae_model = NN()

set_opt = ae.Optimizer(lr=sett.ae_init_lr)
optimizerr = "Adam"
# from keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=5, patience=4)

ae_model.compile(optimizer = set_opt.get_opt(opt=optimizerr), 
              loss_fn = tf.keras.losses.MeanSquaredError(), #ae.MyNMSELoss(), #
              # metrics=additional_metrics)
             )


save_logs = False
save_model = False

model_dir_train = model_dir if save_model else None
log_dir_train = log_dir if save_logs else None

init_time = time.time()

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                    patience=sett.reduce_patience, min_lr=1e-8, min_delta=0, verbose=1)


history = ae_model.fit(train_ds, #train_data_scaled,
#                     train_data_scaled,
                    validation_data = (val_ds,), #(val_data_scaled, val_data_scaled),
                    epochs = ae_epochs, callbacks=[reduce_lr],#, model_check], 
                    verbose = 1,)


end_time = time.time()
train_time = end_time - init_time
hrs = int(train_time//3600); rem_time = train_time - hrs*3600
mins = int(rem_time//60); secs = int(rem_time%60)
print('Training time: %d H %d M, %d S'%(hrs,mins,secs))

ae_model.build(train_data.shape)
ae_model.summary()

encoded = ae_model.encoder(train_data_scaled).numpy()
decoded = ae_model.decoder(encoded).numpy()


print('\n*********AE inverse decoder reconstruction error*********\n')
print('u  Reconstruction MSE: ' + str(np.mean(np.square(scaler.inverse_transform(decoded)))))

train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = history.epoch
lr = history.history['lr']

if not os.path.exists('Figures'):
    os.mkdir('Figures')

fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(16,8),constrained_layout=True)

ax[0].plot(epochs,train_loss,label=key+':train_loss',marker='v',markevery=128)
ax[0].plot(epochs,val_loss,label=key+'::val_loss',marker='s',markevery=128)
ax[0].set_yscale('log')
ax[0].legend()


ax[1].plot(epochs, lr,label=key+':LR',marker='p',markevery=128)
ax[1].set_yscale('log'); ax[1].legend()

plt.suptitle('Training/Validation losses and Learning rate decay in semi-log scale')
plt.savefig('Figures/'+'ae_loss')

## Save the trained AE model
reload(ae)
save_model = True
if save_model:
    

    if not os.path.exists(ae_model_dir):
        os.mkdir(ae_model_dir)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = os.path.join(ae_model_dir, "AE_"+key)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

#     if flag == 'sigma':
#         savedir = model_dir / "saved_model_AA1"
# #         savedir = model_dir / "saved_model_AA2"

#     savedir.mkdir(parents=True, exist_ok=True)
#     mnum = str(savedir).split('saved_model_')[1]
    
#     if flag == 'sigma':
    msg = f'Train_list = {param_train}, Val_list = {param_val}, Test_list = {param_test}'\
        +'\nTrains for %dh %dm %ds,'%(hrs,mins,secs)\
        +'\nStep decay LR scheduler starting from %.2e, Batch Size = %d,'%(lr[0],sett.ae_batch_size)\
        +'\nDecay factor = %.3f every %d epochs.'%(0.9,500)+' Trained for %d epochs,'%(len(epochs))\
        +'\nEncoder input is not augmented by parameter value'
    print("\n===========")
    print(msg)
    
    
    model_results = {'loss': train_loss, 'valloss': val_loss,
                     'epochs': epochs, 'msg': msg,
                     'umax': 1, 'umin': 0,
                     'savedir': str(out_dir), 'timestamp': timestamp}
    
    ae.save_model(ae_model, train_data.shape, model_results)


train_ls = ae_model.encoder(train_data_scaled)
val_ls = ae_model.encoder(val_data_scaled)
test_ls = ae_model.encoder(test_data_scaled)

latent_dim = train_ls.shape[1]

train_ls = np.reshape(train_ls,(Np_train,Nt_train,latent_dim))
val_ls = np.reshape(val_ls,(Np_val,Nt_val,latent_dim))
test_ls = np.reshape(test_ls,(Np_test,Nt_test,latent_dim))

b_train, t_train, target_train = pu.latent_stacked_param_hard_windows(param_train,train_ls,train_snaps[0],train_snaps[1],data_dir)
b_val, t_val, target_val = pu.latent_stacked_param_hard_windows(param_val,val_ls,val_snaps[0],val_snaps[1],data_dir)

###SCALING
if scaling is True:
    t_scaler = np.max(t_train)
    b_scaler = MinMaxScaler(feature_range=(-1, 1))
    ls_scaler = MinMaxScaler(feature_range=(-1, 1))

    b_scaler.fit(b_train)
    ls_scaler.fit(target_train)

    # Reshape and scale the datasets
    t_train = t_train/t_scaler
    t_val = t_val/t_scaler

    b_train = b_scaler.transform(b_train)
    b_val = b_scaler.transform(b_val)

    target_train = ls_scaler.transform(target_train)
    target_val = ls_scaler.transform(target_val)

    joblib.dump(t_scaler, str(scalers_dir)+'/t_scaler_'+str(key)+'.save')
    joblib.dump(b_scaler, str(scalers_dir)+'/b_scaler_'+str(key)+'.save')
    joblib.dump(ls_scaler, str(scalers_dir)+'/ls_scaler_'+str(key)+'.save')


def NN():
    # Define search space
    verbosity_mode = 1

    branch_input_shape = b_train.shape[1]
    t_input_shape = t_train.shape[1]
    l_factor = sett.l_factor 

    b_number_layers = sett.b_number_layers 
    b_neurons_layer = sett.b_neurons
    b_actf = sett.b_actf  
    b_regularizer = sett.b_regularizer  
    b_initializer = sett.b_initializer 

    t_number_layers = sett.t_number_layers     
    t_neurons_layer = sett.t_neurons
    t_actf = sett.t_actf  
    t_regularizer = sett.t_regularizer  
    t_initializer = sett.t_initializer 

    init_lr = sett.init_lr

    optimizer = tf.keras.optimizers.Adam(init_lr)  

    loss_obj = tf.keras.losses.MeanSquaredError()
    
    nn = ldon.ldon_nn(l_factor, 
                    latent_dim, 
                    branch_input_shape,
                    b_number_layers, 
                    b_neurons_layer, 
                    b_actf, 
                    b_initializer, 
                    b_regularizer, 
                    t_input_shape, 
                    t_number_layers, 
                    t_neurons_layer, 
                    t_actf, 
                    t_initializer, 
                    t_regularizer, 
                   )

    model = ldon.ldon_model(nn)

    optimizer = tf.keras.optimizers.Adam(init_lr)  

    loss_obj = tf.keras.losses.MeanSquaredError()

    model.compile(
        optimizer = optimizer,
        loss_fn = loss_obj)
    
    return model

model = NN()

batch_size = sett.batch_size

train_generator = DONDataGenerator(b_train,t_train,target_train,batch_size, shuffle=True)
val_generator = DONDataGenerator(b_val,t_val,target_val,batch_size, shuffle=True)  ## Should shuffle be True??

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                    patience=sett.reduce_patience, min_lr=1e-8, min_delta=0, verbose=1) 

ldon_checkpoint_filepath = './tmp/checkpoint_ldon'+str(key)
model_check = tf.keras.callbacks.ModelCheckpoint(
    filepath=ldon_checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

i=0

model.fit(train_generator, validation_data=val_generator, epochs=ldon_epochs,
           callbacks=[reduce_lr, model_check])

## Save the trained LDON model
save_model = True
if save_model:

    if not os.path.exists(ldon_model_dir):
        os.mkdir(ldon_model_dir)
    
    out_dir = os.path.join(ldon_model_dir, "LDON_"+key)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    model.save(out_dir) 
    np.savez(os.path.join(ldon_model_dir,'history'), history=model.history.history, allow_pickle=True,)

if not os.path.exists('Figures'):
    os.mkdir('Figures')

fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(16,8),constrained_layout=True)

ax[0].plot(model.history.epoch,model.history.history['loss'],label=key+':train_loss',marker='v',markevery=128)
ax[0].plot(model.history.epoch,model.history.history['val_loss'],label=key+'::val_loss',marker='s',markevery=128)
ax[0].set_yscale('log'); #ax[0,2].set_title('Validation and Training losses in semi-log scale')
ax[0].legend()


ax[1].plot(model.history.epoch,model.history.history['lr'],label=key+':LR',marker='p',markevery=128)
ax[1].set_yscale('log'); ax[1].legend()

plt.suptitle('Training/Validation losses and Learning rate decay in semi-log scale')
plt.savefig('Figures/'+'ldon_loss')

out_dir = os.path.join(ldon_model_dir, "LDON_checkpoint"+key)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

model.load_weights(ldon_checkpoint_filepath)
model.save(out_dir)

