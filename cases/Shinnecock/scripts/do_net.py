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

don_model_dir = "saved_models_don"
scalers_dir = "scalers_don"

#Load custom modules
import don
import data_loader as dl 
import settings_don as sett
import processing_utils as pu
from data_gen import DONDataGenerator

data_dir = Path(sett.data_dir)

#Epochs and trials
don_epochs = sett.don_epochs

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

b_train, t_train, target_train = pu.full_stacked_param_hard_windows(param_train,key,train_snaps[0],train_snaps[1],data_dir)
b_val, t_val, target_val = pu.full_stacked_param_hard_windows(param_val,key,val_snaps[0],val_snaps[1],data_dir)

###SCALING
if scaling is True:
    scalers_dir = Path(scalers_dir)
    scalers_dir.mkdir(parents=True, exist_ok=True)
    tx_scaler = MinMaxScaler(feature_range=(-1, 1))
    ty_scaler = MinMaxScaler(feature_range=(-1, 1))
    tt_scaler = np.max(t_train[:,2])
    b_scaler = MinMaxScaler(feature_range=(-1, 1))
    target_scaler = MinMaxScaler(feature_range=(-1, 1))

    tx_scaler.fit(t_train[:,0].reshape(-1, 1))
    ty_scaler.fit(t_train[:,1].reshape(-1, 1))
    b_scaler.fit(b_train[:,1:])
    target_scaler.fit(target_train)

    # Reshape and scale the datasets
    t_train[:,0] = tx_scaler.transform(t_train[:,0].reshape(-1, 1))[:,0]
    t_train[:,1] = ty_scaler.transform(t_train[:,1].reshape(-1, 1))[:,0]
    t_train[:,2] = t_train[:,2]/tt_scaler

    t_val[:,0] = tx_scaler.transform(t_val[:,0].reshape(-1, 1))[:,0]
    t_val[:,1] = ty_scaler.transform(t_val[:,1].reshape(-1, 1))[:,0]
    t_val[:,2] = t_val[:,2]/tt_scaler
    
    b_train[:,1:] = b_scaler.transform(b_train[:,1:])
    b_val[:,1:] = b_scaler.transform(b_val[:,1:])

    target_train = target_scaler.transform(target_train)
    target_val = target_scaler.transform(target_val)

    joblib.dump(tx_scaler, str(scalers_dir)+'/tx_scaler_'+str(key)+'.save')
    joblib.dump(ty_scaler, str(scalers_dir)+'/ty_scaler_'+str(key)+'.save')
    joblib.dump(tt_scaler, str(scalers_dir)+'/tt_scaler_'+str(key)+'.save')
    joblib.dump(b_scaler, str(scalers_dir)+'/b_scaler_'+str(key)+'.save')
    joblib.dump(target_scaler, str(scalers_dir)+'/target_scaler_'+str(key)+'.save')

def NN():
    # Define search space
    verbosity_mode = 1

    b_input_shape = b_train.shape[1]
    t_input_shape = t_train.shape[1]
    output_shape = sett.output_shape

    b_neurons_layer = sett.b_neurons
    b_number_layers = sett.b_number_layers 
    b_actf = sett.b_actf  
    b_regularizer = sett.b_regularizer  
    b_initializer = sett.b_initializer 

    t_neurons_layer = sett.t_neurons
    t_number_layers = sett.t_number_layers     
    t_actf = sett.t_actf  
    t_regularizer = sett.t_regularizer  
    t_initializer = sett.t_initializer 

    init_lr = sett.init_lr

    optimizer = tf.keras.optimizers.Adam(init_lr)  

    loss_obj = tf.keras.losses.MeanSquaredError()
    
    nn = don.don_nn(b_input_shape,
                    b_number_layers, 
                    b_neurons_layer,
                    output_shape, 
                    b_actf, 
                    b_initializer, 
                    b_regularizer, 
                    t_input_shape, 
                    t_number_layers,
                    t_neurons_layer,
                    output_shape, 
                    t_actf, 
                    t_initializer, 
                    t_regularizer, 
                   )

    model = don.don_model(nn)

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

don_checkpoint_filepath = './tmp/checkpoint_don'+str(key)
model_check = tf.keras.callbacks.ModelCheckpoint(
    filepath=don_checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

i=0

model.fit(train_generator, validation_data=val_generator, epochs=don_epochs,
           callbacks=[reduce_lr, model_check])

## Save the trained DON model
save_model = True
if save_model:

    if not os.path.exists(don_model_dir):
        os.mkdir(don_model_dir)
    
    out_dir = os.path.join(don_model_dir, "DON_"+key)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    model.save(out_dir) 
    np.savez(os.path.join(don_model_dir,'history'), history=model.history.history, allow_pickle=True,)

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
plt.savefig('Figures/'+'don_loss')

out_dir = os.path.join(don_model_dir, "DON_checkpoint"+key)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

model.load_weights(don_checkpoint_filepath)
model.save(out_dir)

