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
ae_epochs = 10000
don_epochs = 10000
loss = 'mse'
optimizer_str = 'adam'
scaling = True
scale_min = 0#-1
scale_max = 1

key = 'S_dep' #S_vx, S_vy, S_dep
day0 = [15,30,45] 
day1 = [20,35,50] 
val_day0 = 20 
val_day1 = 30 
test_day0 = 10 
test_day1 = 55 
t_skip = 4
lookforward_window = 5

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
###DON

l_factor = 2
l_factor_encoder = 1

b_number_layers = 3
b_actf = 'swish'
b_regularizer = 'none'
b_initializer = 'glorot_normal'

b_number_layers_encoder = 2
b_encoder_actf = 'tanh'
b_encoder_regularizer = 'l2'
b_encoder_init = 'he_normal'

t_number_layers = 5
t_actf = 'tanh'
t_regularizer = 'l2'
t_initializer = 'he_normal'

t_number_layers_encoder =  2
t_encoder_actf = 'elu'
t_encoder_regularizer = 'l2'
t_encoder_init = 'glorot_normal'

batch_size = 512
init_lr = 1e-5

###AE OPTUNA
ae_steps = 100
ae_factor = 0.9
ae_learning_rate_decay = True

ae_batch_size = 1024
ae_init_lr = 5e-05

ae_number_layers = 2

latent_dim = 64


enc_act = "swish"
dec_act = "relu"

ae_optimizer = "Adam"

###CALLBACKS
reduce_patience = 100
