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
mio_epochs = 20000
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

###MIO
output_shape = 256
b_neurons = 256
b_number_layers = 3
b_actf = 'elu'
b_regularizer = 'none'
b_initializer = 'he_uniform'

t_neurons = 256
t_number_layers = 4
t_actf = 'elu'
t_regularizer = 'none'
t_initializer = 'he_normal'

batch_size = 128
init_lr = 0.0005

###CALLBACKS
reduce_patience = 100

