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
ae_epochs = 20000
mito_epochs = 20000
loss = 'mse'
optimizer_str = 'adam'
scaling = True

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
window_size = 5

#H
###MITONet
key = 'h' #u,v,h
l_factor = 4
l_factor_encoder = 3

b_number_layers = 3
b_actf = 'elu'
b_regularizer = 'none'
b_initializer = 'he_uniform'

b_number_layers_encoder = 4
b_encoder_actf = 'relu'
b_encoder_regularizer = 'l1'
b_encoder_init = 'he_uniform'

t_number_layers = 4
t_actf = 'elu'
t_regularizer = 'none'
t_initializer = 'he_normal'

t_number_layers_encoder = 3
t_encoder_actf = 'relu'
t_encoder_regularizer = 'none'
t_encoder_init = 'glorot_uniform'

batch_size = 512              # MITONet training (from table)
init_lr = 5.00e-04            # MITONet training (from table)

###AE OPTUNA (from table)
ae_steps = 100
ae_factor = 0.9
ae_learning_rate_decay = True

ae_batch_size = 64
ae_init_lr = 5.00e-05

ae_number_layers = 3
latent_dim = 60
enc_act = "tanh"
dec_act = "tanh"
ae_optimizer = "Adam"

###CALLBACKS
reduce_patience = 100

##########################

# #U
# ###MITONet
# key = 'u' #u,v,h
# l_factor = 5
# l_factor_encoder = 5

# b_number_layers = 5
# b_actf = 'elu'
# b_regularizer = 'none'
# b_initializer = 'he_uniform'

# b_number_layers_encoder = 3
# b_encoder_actf = 'tanh'
# b_encoder_regularizer = 'none'
# b_encoder_init = 'glorot_uniform'

# t_number_layers = 4
# t_actf = 'swish'
# t_regularizer = 'l1'
# t_initializer = 'he_normal'

# t_number_layers_encoder = 2
# t_encoder_actf = 'tanh'
# t_encoder_regularizer = 'none'
# t_encoder_init = 'glorot_uniform'

# batch_size = 1024             # MITONet training (from table)
# init_lr = 5.00e-04            # MITONet training (from table)

# ###AE OPTUNA (from table)
# ae_steps = 100
# ae_factor = 0.9
# ae_learning_rate_decay = True

# ae_batch_size = 64
# ae_init_lr = 5.00e-05

# ae_number_layers = 2
# latent_dim = 60
# enc_act = "swish"
# dec_act = "swish"
# ae_optimizer = "Adam"

# ###CALLBACKS
# reduce_patience = 100

##########################

# #V
# ###MITONet
# key = 'v' #u,v,h
# l_factor = 6
# l_factor_encoder = 5

# b_number_layers = 5
# b_actf = 'relu'
# b_regularizer = 'none'
# b_initializer = 'glorot_normal'

# b_number_layers_encoder = 3
# b_encoder_actf = 'elu'
# b_encoder_regularizer = 'none'
# b_encoder_init = 'glorot_normal'

# t_number_layers = 2
# t_actf = 'tanh'
# t_regularizer = 'l2'
# t_initializer = 'glorot_normal'

# t_number_layers_encoder = 3
# t_encoder_actf = 'swish'
# t_encoder_regularizer = 'none'
# t_encoder_init = 'he_uniform'

# batch_size = 512              # MITONet training (from table)
# init_lr = 5.00e-05            # MITONet training (from table)

# ###AE OPTUNA (from table)
# ae_steps = 100
# ae_factor = 0.9
# ae_learning_rate_decay = True

# ae_batch_size = 64
# ae_init_lr = 5.00e-05

# ae_number_layers = 2
# latent_dim = 60
# enc_act = "swish"
# dec_act = "tanh"
# ae_optimizer = "Adam"

# ###CALLBACKS
# reduce_patience = 100
