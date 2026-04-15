from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parents[3]
RED_RIVER_DATA_DIR = (
    _ROOT_DIR
    / 'data'
    / 'PRJ-6207'
    / 'Simulation--2d-adh-simulation-of-riverine-hydrodynamics-in-a-section-of-the-red-river-in-louisiana-usa-parameterized-by-the-bottom-friction-coefficient'
    / 'data'
    / 'Model--adh'
    / 'Input--2d-adh-input-files'
    / 'Output--processed-2d-adh-output'
    / 'data'
)
data_dir = str(RED_RIVER_DATA_DIR)


###GENERAL
ae_epochs = 20000 
ae_tuner_epochs = 3000 
ae_trials = 50
don_epochs = 20000
don_tuner_epochs = 3000
don_trials = 100
loss = 'mse'
optimizer_str = 'adam'
scaling = True
scale_min = -1
scale_max = 1

key = 'S_vy' #S_vx, S_vy, S_dep
day0 = [15,30,45] 
day1 = [20,35,50] 
val_day0 = 20 
val_day1 = 30 
test_day0 = 40 
test_day1 = 55 
t_skip = 4
lookforward_window = 10

param_list = [
    0.02375,
    0.023875,
    0.024125,
    0.024375,
    0.024625,
    0.024875,
    0.025,
    0.025125,
    0.025375,
    0.025625,
    0.025875,
    0.026125,
    0.02625,
]
param_train = [0.023875, 0.024375,  0.024875,  0.025125, 0.025625, 0.026125]
param_val = [0.024125, 0.024625, 0.025375, 0.025875]
param_test = [0.02375, 0.025, 0.02625]

###BDRY NODES
discharge_nodes = [7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
tailwater_nodes = [12248, 12249, 12250, 12251, 12252, 12253, 12254, 12270, 12271, 12272, 12273, 12274,
 12275, 12276, 12284, 12285, 12286, 12290]

###DON OPTUNA
b_number_layers_lower = 2
b_number_layers_upper = 5
b_number_layers_step = 1


b_number_layers_encoder_lower = 2
b_number_layers_encoder_upper = 4
b_number_layers_encoder_step = 1

t_number_layers_encoder_lower = 2
t_number_layers_encoder_upper = 4
t_number_layers_encoder_step = 1

l_factor_lower = 2
l_factor_upper = 4
l_factor_step = 1

l_factor_encoder_lower = 1
l_factor_encoder_upper = 3
l_factor_encoder_step = 1

b_actf = ["relu", "elu", "tanh", "swish"]

b_regularizer = ["none", "l1", "l2"]

b_initializer = ["glorot_normal", "glorot_uniform", "he_normal", "he_uniform"]

t_number_layers_lower = 2
t_number_layers_upper = 5
t_number_layers_step = 1

t_actf = ["relu", "elu", "tanh", "swish"]

t_regularizer = ["none", "l1", "l2"]

t_initializer = ["glorot_normal", "glorot_uniform", "he_normal", "he_uniform"]


b_encoder_actf = ["relu", "elu", "tanh", "swish"]

b_encoder_regularizer = ["none", "l1", "l2"]

b_encoder_init = ["glorot_normal", "glorot_uniform", "he_normal", "he_uniform"]

t_encoder_actf = ["relu", "elu", "tanh", "swish"]

t_encoder_regularizer = ["none", "l1", "l2"]

t_encoder_init = ["glorot_normal", "glorot_uniform", "he_normal", "he_uniform"]

batch_size = [512,1024,2048]


init_lr = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

###AE OPTUNA
ae_steps = 100 #1000
ae_factor = 0.9
ae_learning_rate_decay = True

ae_batch_size = [64,128,256,512,1024]
ae_init_lr = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

ae_number_layers_lower = 2
ae_number_layers_upper = 3
ae_number_layers_step = 1

latent_dim_lower = 64 
latent_dim_upper = 128
latent_dim_step = 32


enc_act = ["relu", "elu", "tanh", "swish"]
dec_act = ["relu", "elu", "tanh", "swish"]

ae_optimizer = "Adam"

###CALLBACKS
reduce_patience = 100
