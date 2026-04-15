from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parents[3]
SHINNECOCK_DATA_DIR = (
    _ROOT_DIR
    / 'data'
    / 'PRJ-5716'
    / 'Simulation--2d-adcirc-simulation-of-tidal-flow-in-shinnecock-inlet-ny-parameterized-by-bottom-friction-coefficient'
    / 'data'
    / 'Model--adcirc-model'
    / 'data'
)
data_dir = str(SHINNECOCK_DATA_DIR)


###GENERAL
don_epochs = 20000
don_tuner_epochs = 3000
don_trials = 50
loss = 'mse'
optimizer_str = 'adam'
scaling = True

key = 'h' #u,v,h
day0 = 15
day1 = 30
val_day0 = 5
val_day1 = 15
test_day0 = 5
test_day1 = 60
param_list = [0.0025, 0.00275, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1]
param_train = [0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
param_val = [0.00275, 0.0075, 0.025, 0.075]
param_test = [.0025, 0.015, 0.1]


###DON OPTUNA
output_shape_lower = 64
output_shape_upper = 512
output_shape_step = 64

b_number_layers_lower = 2
b_number_layers_upper = 5
b_number_layers_step = 1

b_neurons_layer_lower = 64
b_neurons_layer_upper = 512
b_neurons_layer_step = 64

b_actf = ["relu", "elu", "tanh", "swish"]

b_regularizer = ["none", "l1", "l2"]

b_initializer = ["glorot_normal", "glorot_uniform", "he_normal", "he_uniform"]

t_number_layers_lower = 2
t_number_layers_upper = 5
t_number_layers_step = 1

t_neurons_layer_lower = 64
t_neurons_layer_upper = 512
t_neurons_layer_step = 64

t_actf = ["relu", "elu", "tanh", "swish"]

t_regularizer = ["none", "l1", "l2"]

t_initializer = ["glorot_normal", "glorot_uniform", "he_normal", "he_uniform"]

batch_size = [64,128,256,512]

init_lr = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

###CALLBACKS
reduce_patience = 100
