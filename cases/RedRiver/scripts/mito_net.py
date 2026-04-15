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

import optuna
from optuna.integration import TFKerasPruningCallback
from tensorflow.keras.backend import clear_session

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

ae_model_dir = work_dir / "saved_models"
mito_model_dir = work_dir / "saved_models"
scalers_dir = work_dir / "scalers"

#Load custom modules
import mitonet as don
import autoencoder as ae
import data_loader as dl 
import settings as sett
from processing_utils import multiple_param_windows

data_dir = Path(sett.data_dir)

#Epochs and trials
ae_epochs = sett.ae_epochs
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

in_nodes = sett.discharge_nodes
out_nodes = sett.tailwater_nodes

## Skipping every alternate snapshots
t_skip = sett.t_skip

scaling = sett.scaling
scale_min = sett.scale_min
scale_max = sett.scale_max

window_size = sett.lookforward_window

Np_train = len(param_train)
Np_val = len(param_val)
Np_test = len(param_test)

data_dict = {}
for inx, param in enumerate(param_list):
    fname = f'Inset_T5.198e+06_mn{param:.6f}.npz'
    data_dict[inx] = dl.load_variables_adh(data_dir/fname, comp_names=[key], t_start = 480) ## Ignoring the first five days of simulation data

time_load = data_dict[0]['time'][::t_skip].copy()

if isinstance(train_day0, list):
    train_snaps = [[np.searchsorted(time_load/60/60/24, train_day0[i]), np.searchsorted(time_load/60/60/24, train_day1[i])] for i in range(len(train_day0))]
    Nt_snaps_train = 0
    for i in range(len(train_day0)):
        Nt_snaps_train += train_snaps[i][1] - train_snaps[i][0]
    
elif isinstance(train_day0, int):
    train_snaps = [np.searchsorted(time_load/60/60/24, train_days[0]), np.searchsorted(time_load/60/60/24,train_days[1])]
    Nt_snaps_train = time_load[train_snaps[0]:train_snaps[1]].shape[0]
else:
    print("'day0' and 'day1' can only be 'int' or 'list' objects.")
    
val_snaps = [np.searchsorted(time_load/60/60/24,val_days[0]), np.searchsorted(time_load/60/60/24,val_days[1])]
Nt_snaps_val = time_load[val_snaps[0]:val_snaps[1]].shape[0]
test_snaps = [np.searchsorted(time_load/60/60/24,test_days[0]), np.searchsorted(time_load/60/60/24,test_days[1])]
Nt_snaps_test = time_load[test_snaps[0]:test_snaps[1]].shape[0]

print(f"Train steps - {train_snaps}, Val steps - {val_snaps}, Test steps - {test_snaps}")
mesh_fname = "red_river_mesh.npz"
mesh = dl.load_mesh_adh(data_dir / mesh_fname)

Nn = mesh['nodes'].shape[0]
Ne = mesh['triangles'].shape[0]
Nt = time_load.shape[0]

print(f"Loaded {key} simulation output for {len(param_list)} parameter values at {Nn} nodes and {Nt} time steps")

###  Prepare Autoencoder input data =

train_data = np.empty((0, Nn ),)  ## Augment snapshots with parameter value, if needed
val_data = np.empty((0, Nn ),)
test_data = np.empty((0, Nn ),)

for inx,param in enumerate(param_train):
    indx = param_list.index(param)
    snap = data_dict[indx][key][:,::t_skip].T 
    train_data = np.vstack((train_data, snap))
    
validation_data = True
if validation_data:
    
    for inx,param in enumerate(param_val):
        indx = param_list.index(param)
        val_snap = data_dict[indx][key][:,::t_skip].T
        val_data = np.vstack((val_data, val_snap))

    for inx,param in enumerate(param_test):
        indx = param_list.index(param)
        test_snap = data_dict[indx][key][:,::t_skip].T
        test_data = np.vstack((test_data, test_snap))

print("Split data into training, validation, and testing for:",key)

### CROP
train_data_resh = np.reshape(train_data,(Np_train,Nt,Nn))
val_data_resh = np.reshape(val_data,(Np_val,Nt,Nn))
test_data_resh = np.reshape(test_data,(Np_test,Nt,Nn))

if isinstance(train_day0, list):
    train_data_crop = train_data_resh[:,train_snaps[0][0]:train_snaps[0][1],:]
    for i in range(1,len(train_day0)):
        train_data_crop = np.concatenate((train_data_crop, train_data_resh[:,train_snaps[i][0]:train_snaps[i][1],:]), axis=1)
elif isinstance(train_day0, int):
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

if not os.path.exists(scalers_dir):
    os.mkdir(scalers_dir)

### SCALE
if scaling is True:
    # Initialize the scaler
    scaler = MinMaxScaler(feature_range=(scale_min, scale_max))

    # Fit the scaler using the adjusted min and max
    min_vec = np.min(train_data, axis=0) - np.abs(np.min(train_data, axis=0)*0.20) 
    max_vec = np.max(train_data, axis=0) + np.abs(np.max(train_data, axis=0)*0.20) 
    scaling_vec = np.vstack((max_vec,min_vec))
    scaler.fit(scaling_vec)

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
        elif i==1:
            size[i] = int(size[i-1]/32) 
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


if isinstance(train_day0, int):

    b_par_train, b_ic_train, b_bc1_train, b_bc2_train, t_train, target_train = multiple_param_windows(param_train,train_ls,time_load,
                                                                            train_snaps[0],train_snaps[1],
                                                                            t_skip, data_dir, in_nodes, out_nodes,
                                                                            window_s=sett.lookforward_window)

elif isinstance(train_day0, list):
    train_start = np.array([train_snaps[i][0] for i in range(len(train_day0))])
    train_end = np.array([train_snaps[i][1] for i in range(len(train_day0))])
    cumul_snaps_per_window = np.cumsum(train_end - train_start)

    for idx in range(len(train_day0)):
        if idx == 0:
            
            b_par_train, b_ic_train, b_bc1_train, b_bc2_train, t_train, target_train = multiple_param_windows(param_train,
                                                                                    train_ls[:,:cumul_snaps_per_window[idx],:], 
                                                                                    time_load,
                                                                                    train_start[0], train_end[0],
                                                                                    t_skip, data_dir, in_nodes, out_nodes, 
                                                                                    window_s=sett.lookforward_window)
            
        else:
            
            tmp_par_train, tmp_ic_train, tmp_bc1_train, tmp_bc2_train, tmp_t_train, tmp_target_train = multiple_param_windows(param_train,
                train_ls[:,cumul_snaps_per_window[idx-1]:cumul_snaps_per_window[idx],:],
                time_load,
                train_start[idx], train_end[idx],
                t_skip, data_dir, in_nodes, out_nodes,  
                window_s=sett.lookforward_window)
            
            b_par_train = np.vstack([b_par_train, tmp_par_train])
            b_ic_train = np.vstack([b_ic_train, tmp_ic_train])
            b_bc1_train = np.vstack([b_bc1_train, tmp_bc1_train])
            b_bc2_train = np.vstack([b_bc2_train, tmp_bc2_train])
            t_train = np.vstack([t_train, tmp_t_train])
            target_train = np.vstack([target_train, tmp_target_train])


            
b_par_val, b_ic_val, b_bc1_val, b_bc2_val, t_val, target_val = multiple_param_windows(param_val,val_ls,time_load, 
                                                                                val_snaps[0],val_snaps[1],
                                                                                t_skip, data_dir, in_nodes, out_nodes, 
                                                                                window_s=sett.lookforward_window)
    
###SCALING
if scaling is True:
    t_scaler = np.max(t_train)
    b_bc1_scaler = MinMaxScaler(feature_range=(-1, 1))
    b_bc2_scaler = MinMaxScaler(feature_range=(-1, 1))
    ls_scaler = MinMaxScaler(feature_range=(-1, 1))


    # Fit the scalers using the adjusted min and max values
    bc1_min_vec = np.min(b_bc1_train, axis=0) - np.abs(np.min(b_bc1_train, axis=0)*0.20) 
    bc1_max_vec = np.max(b_bc1_train, axis=0) + np.abs(np.max(b_bc1_train, axis=0)*0.20) 
    bc1_scaling_vec = np.vstack((bc1_max_vec,bc1_min_vec))
    b_bc1_scaler.fit(bc1_scaling_vec)
    bc2_min_vec = np.min(b_bc2_train, axis=0) - np.abs(np.min(b_bc2_train, axis=0)*0.20) 
    bc2_max_vec = np.max(b_bc2_train, axis=0) + np.abs(np.max(b_bc2_train, axis=0)*0.20) 
    bc2_scaling_vec = np.vstack((bc2_max_vec,bc2_min_vec))
    b_bc2_scaler.fit(bc2_scaling_vec)    
    ls_min_vec = np.min(target_train, axis=0) - np.abs(np.min(target_train, axis=0)*0.20) 
    ls_max_vec = np.max(target_train, axis=0) + np.abs(np.max(target_train, axis=0)*0.20)
    ls_scaling_vec = np.vstack((ls_max_vec,ls_min_vec))
    ls_scaler.fit(ls_scaling_vec)

    # Reshape and scale the datasets
    t_train = t_train/t_scaler
    t_val = t_val/t_scaler

    b_bc1_train = b_bc1_scaler.transform(b_bc1_train)
    b_bc1_val = b_bc1_scaler.transform(b_bc1_val)


    b_bc2_train = b_bc2_scaler.transform(b_bc2_train)
    b_bc2_val = b_bc2_scaler.transform(b_bc2_val)
    
    b_ic_train = ls_scaler.transform(b_ic_train)
    b_ic_val = ls_scaler.transform(b_ic_val)

    target_train = ls_scaler.transform(target_train)
    target_val = ls_scaler.transform(target_val)

    joblib.dump(t_scaler, str(scalers_dir)+'/t_scaler_'+str(key)+'.save')
    joblib.dump(b_bc1_scaler, str(scalers_dir)+'/b_bc1_scaler_'+str(key)+'.save')
    joblib.dump(b_bc2_scaler, str(scalers_dir)+'/b_bc2_scaler_'+str(key)+'.save')
    joblib.dump(ls_scaler, str(scalers_dir)+'/ls_scaler_'+str(key)+'.save')

def NN():
    # Define search space
    verbosity_mode = 1

    branch_par_sensors = b_par_train.shape[1]
    branch_ic_sensors = b_ic_train.shape[1]
    branch_bc1_sensors = b_bc1_train.shape[1]
    branch_bc2_sensors = b_bc2_train.shape[1]
    l_factor = sett.l_factor 
    l_factor_encoder = sett.l_factor_encoder 
    b_number_layers = sett.b_number_layers 
    b_actf = sett.b_actf  
    b_regularizer = sett.b_regularizer  
    b_initializer = sett.b_initializer 
    b_encoder_layers = sett.b_number_layers_encoder 
    b_encoder_actf = sett.b_encoder_actf  
    b_encoder_regularizer = sett.b_encoder_regularizer  
    b_encoder_init = sett.b_encoder_init 
    
    t_number_layers = sett.t_number_layers     
    t_actf = sett.t_actf  
    t_regularizer = sett.t_regularizer  
    t_initializer = sett.t_initializer 
    
    
    t_encoder_layers =sett.t_number_layers_encoder 
    
    t_encoder_actf = sett.t_encoder_actf  
    t_encoder_regularizer = sett.t_encoder_regularizer  
    t_encoder_init = sett.t_encoder_init 

    init_lr = sett.init_lr

    optimizer = tf.keras.optimizers.Adam(init_lr)  

    loss_obj = tf.keras.losses.MeanSquaredError()
    
    nn = don.don_nn(l_factor, 
                    latent_dim, 
                    branch_par_sensors,
                    branch_ic_sensors,
                    branch_bc1_sensors,
                    branch_bc2_sensors,
                    b_number_layers, 
                    l_factor*latent_dim, 
                    b_actf, 
                    b_initializer, 
                    b_regularizer, 
                    b_encoder_layers, 
                    l_factor_encoder*latent_dim, 
                    b_encoder_actf, 
                    b_encoder_init, 
                    b_encoder_regularizer, 
                    1, 
                    t_number_layers, 
                    l_factor*latent_dim, 
                    t_actf, 
                    t_initializer, 
                    t_regularizer, 
                    t_encoder_layers, 
                    l_factor_encoder*latent_dim, 
                    t_encoder_actf, 
                    t_encoder_init, 
                    t_encoder_regularizer
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

dataset = tf.data.Dataset.from_tensor_slices((b_par_train,b_ic_train,b_bc1_train,b_bc2_train,t_train, target_train))
dataset = dataset.shuffle(buffer_size=int(t_train.shape[0])).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((b_par_val,b_ic_val,b_bc1_val,b_bc2_val,t_val, target_val))
val_dataset = val_dataset.batch(batch_size)

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

model.fit(dataset, validation_data=val_dataset, epochs=don_epochs,
           callbacks=[reduce_lr, model_check])

## Save the trained DON model
save_model = True
if save_model:

    if not os.path.exists(mito_model_dir):
        os.mkdir(mito_model_dir)
    
    out_dir = os.path.join(mito_model_dir, "MITO_"+key)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    model.save(out_dir)  
    np.savez(os.path.join(mito_model_dir,'history'), history=model.history.history, allow_pickle=True,)


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

out_dir = os.path.join(mito_model_dir, "MITO_checkpoint"+key)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

model.load_weights(don_checkpoint_filepath)
model.save(out_dir)
