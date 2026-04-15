import numpy as np
import data_loader as dl 
from tqdm import tqdm

def latent_multiple_param(a_list,latent_data,start,end,fpath):

    meshdir = 'cf'+str(a_list[0]).split('.')[1]
    mesh = dl.load_mesh(fpath/meshdir)
    full_data = dl.load_variables(fpath/meshdir)
    coord_n = mesh['nodes'][:,0].shape[0]
    t_t = mesh['t'][start:end]

    b_par_input = np.empty((len(a_list)*t_t.shape[0], 1))
    b_ic_input = np.empty((len(a_list)*t_t.shape[0], latent_data.shape[2]))
    b_bc_input = np.empty((len(a_list)*t_t.shape[0], 75))
    t_input = np.empty((len(a_list)*t_t.shape[0],1))
    target = np.empty((len(a_list)*t_t.shape[0],latent_data.shape[2]))

    count = 0
    for inx,a in tqdm(enumerate(a_list)):

        data = latent_data[inx,:,:]
        ic = data[0,:]
        
        for id_t,it in enumerate(t_t):
            
            id_t_temp_1 = id_t 
            id_t_temp_2 = id_t + start 
            solution_h = full_data['h'][:,id_t_temp_2]

            h_bc = solution_h[:75]
               
            b_par_input[count,:] = a
            b_ic_input[count,:] = ic
            b_bc_input[count,:] = h_bc
            t_input[count,:] = it
            target[count,:] = data[id_t_temp_1,:]
            
            count = count+1
            
    return b_par_input, b_ic_input, b_bc_input, t_input, target

def latent_multiple_param_windows(a_list,latent_data,start,end,fpath,window_s=5):

    meshdir = 'cf'+str(a_list[0]).split('.')[1]
    mesh = dl.load_mesh(fpath/meshdir)
    full_data = dl.load_variables(fpath/meshdir)
    coord_n = mesh['nodes'][:,0].shape[0]
    t_t = mesh['t'][start:end]

    b_par_input = np.empty((len(a_list)*window_s*t_t.shape[0], 1))
    b_ic_input = np.empty((len(a_list)*window_s*t_t.shape[0], latent_data.shape[2]))
    b_bc_input = np.empty((len(a_list)*window_s*t_t.shape[0], 75))
    t_input = np.empty((len(a_list)*window_s*t_t.shape[0],1))
    target = np.empty((len(a_list)*window_s*t_t.shape[0],latent_data.shape[2]))

    t_t = mesh['t'][start:end-window_s]

    count = 0
    for inx,a in tqdm(enumerate(a_list)):

        data = latent_data[inx,:,:]

        
        for id_t,it in enumerate(t_t):
            ic = data[id_t,:]

            window = 1
            it = 1800.

            for j in range(window_s):
                id_t_temp_1 = id_t + window
                id_t_temp_2 = id_t + start + window

                solution_h = full_data['h'][:,id_t_temp_2]
                h_bc = solution_h[:75]

                b_par_input[count,:] = a
                b_ic_input[count,:] = ic
                b_bc_input[count,:] = h_bc
                t_input[count,:] = it
                target[count,:] = data[id_t_temp_1,:]

                count = count+1
                window = window + 1
                it = it + 1800.

    return b_par_input, b_ic_input, b_bc_input, t_input, target

# def latent_stacked_param(a_list,latent_data,start,end,fpath):

#     meshdir = 'cf'+str(a_list[0]).split('.')[1]
#     mesh = dl.load_mesh(fpath/meshdir)
#     full_data = dl.load_variables(fpath/meshdir)
#     coord_n = mesh['nodes'][:,0].shape[0]
#     x = mesh['nodes'][:,0]
#     y = mesh['nodes'][:,1]
#     t_t = mesh['t'][start:end]
#     latent_dim = latent_data.shape[-1]

#     b_input = np.empty((len(a_list)*int(t_t.shape[0]), 1+latent_dim+75*window_s))
#     target = np.empty((len(a_list)*int(t_t.shape[0]),latent_dim*window_s))

#     t_t = mesh['t'][start:end-window_s]

#     count = 0
#     for inx,a in tqdm(enumerate(a_list)):
#         data = latent_data[inx,:,:]

#         for id_t in range(0,len(t_t)):
#             id_t_temp_1 = id_t 
#             id_t_temp_2 = id_t + start 
            
#             ic = data[id_t_temp_1,:]

#             solution_h = full_data['h'][:,id_t_temp_2+1:id_t_temp_2+1+window_s]
#             h_bc = solution_h[:75,:]

#             b_input[count,:] = np.hstack([a,ic,h_bc.flatten(order='F')])
#             target[count,:] = data[id_t_temp_1+1:id_t_temp_1+1+window_s,:].flatten(order='F')

#             count = count + 1

#     t_input = np.empty((int(window_s),1))
#     dt = t_t[1]-t_t[0]

#     t_input[0] = dt
    
#     for i in range(1,window_s):
#         t_input[i] = dt*(i+1)

#     return b_input, t_input, target

def latent_stacked_param_hard_windows(a_list,latent_data,start,end,fpath,window_s=5):

    meshdir = 'cf'+str(a_list[0]).split('.')[1]
    mesh = dl.load_mesh(fpath/meshdir)
    full_data = dl.load_variables(fpath/meshdir)
    coord_n = mesh['nodes'][:,0].shape[0]
    x = mesh['nodes'][:,0]
    y = mesh['nodes'][:,1]
    t_t = mesh['t'][start:end-window_s]
    latent_dim = latent_data.shape[-1]

    b_input = np.empty((len(a_list)*int(t_t.shape[0]/window_s), 1+latent_dim+75*window_s))
    target = np.empty((len(a_list)*int(t_t.shape[0]/window_s),latent_dim*window_s))

    # t_t = mesh['t'][start:end-window_s]

    count = 0
    for inx,a in tqdm(enumerate(a_list)):
        data = latent_data[inx,:,:]

        for id_t in range(0,len(t_t),window_s):
            id_t_temp_1 = id_t 
            id_t_temp_2 = id_t + start 
            
            ic = data[id_t_temp_1,:]

            solution_h = full_data['h'][:,id_t_temp_2+1:id_t_temp_2+1+window_s]
            h_bc = solution_h[:75,:]

            b_input[count,:] = np.hstack([a,ic,h_bc.flatten(order='F')])
            target[count,:] = data[id_t_temp_1+1:id_t_temp_1+1+window_s,:].T.flatten(order='F')

            count = count + 1

    t_input = np.empty((int(window_s),1))
    dt = t_t[1]-t_t[0]

    t_input[0] = dt
    
    for i in range(1,window_s):
        t_input[i] = dt*(i+1)

    return b_input, t_input, target

# def full_stacked_param(a_list,key,start,end,fpath,window_s=5):

#     meshdir = 'cf'+str(a_list[0]).split('.')[1]
#     mesh = dl.load_mesh(fpath/meshdir)
#     coord_n = mesh['nodes'][:,0].shape[0]
#     x = mesh['nodes'][:,0]
#     y = mesh['nodes'][:,1]
#     t_t = mesh['t'][start:end]

#     b_input = np.empty((len(a_list)*int(t_t.shape[0]), 1+coord_n+75*window_s))
#     target = np.empty((len(a_list)*int(t_t.shape[0]),coord_n*window_s))
    
#     t_t = mesh['t'][start:end-window_s]
    
#     count = 0
#     for inx,a in tqdm(enumerate(a_list)):
#         meshdir = 'cf'+str(a).split('.')[1]
#         full_data = dl.load_variables(fpath/meshdir)
#         data = full_data[key]

#         for id_t in range(0,len(t_t)):
#             id_t_temp = id_t + start
#             ic = data[:,id_t_temp]

#             solution_h = full_data['h'][:,id_t_temp+1:id_t_temp+1+window_s]
#             h_bc = solution_h[:75,:]

#             b_input[count,:] = np.hstack([a,ic,h_bc.flatten(order='F')])
#             target[count,:] = data[:,id_t_temp+1:id_t_temp+1+window_s].flatten(order='F')

#             count = count + 1

#     t_input = np.empty((int(coord_n*window_s),3))
#     dt = t_t[1]-t_t[0]

#     t_input[0:coord_n,0] = x
#     t_input[0:coord_n,1] = y
#     t_input[0:coord_n,2] = np.tile(dt,coord_n)
    
#     for i in range(1,window_s):
#         t_input[i*coord_n:(i+1)*coord_n,0] = x
#         t_input[i*coord_n:(i+1)*coord_n,1] = y
#         t_input[i*coord_n:(i+1)*coord_n,2] = np.tile(dt*(i+1),coord_n)

#     return b_input, t_input, target

def full_stacked_param_hard_windows(a_list,key,start,end,fpath,window_s=5):

    meshdir = 'cf'+str(a_list[0]).split('.')[1]
    mesh = dl.load_mesh(fpath/meshdir)
    coord_n = mesh['nodes'][:,0].shape[0]
    x = mesh['nodes'][:,0]
    y = mesh['nodes'][:,1]
    t_t = mesh['t'][start:end]

    b_input = np.empty((len(a_list)*int(t_t.shape[0]/window_s), 1+coord_n+75*window_s))
    target = np.empty((len(a_list)*int(t_t.shape[0]/window_s),coord_n*window_s))
    
    count = 0
    for inx,a in tqdm(enumerate(a_list)):
        meshdir = 'cf'+str(a).split('.')[1]
        full_data = dl.load_variables(fpath/meshdir)
        data = full_data[key]

        for id_t in range(0,len(t_t),window_s):
            id_t_temp = id_t + start
            ic = data[:,id_t_temp]

            solution_h = full_data['h'][:,id_t_temp+1:id_t_temp+1+window_s]
            h_bc = solution_h[:75,:]

            b_input[count] = np.hstack([a,ic,h_bc.flatten(order='F')])
            target[count,:] = data[:,id_t_temp+1:id_t_temp+1+window_s].flatten(order='F')

            count = count + 1

    t_input = np.empty((int(coord_n*window_s),3))
    dt = t_t[1]-t_t[0]

    t_input[0:coord_n,0] = x
    t_input[0:coord_n,1] = y
    t_input[0:coord_n,2] = np.tile(dt,coord_n)
    
    for i in range(1,window_s):
        t_input[i*coord_n:(i+1)*coord_n,0] = x
        t_input[i*coord_n:(i+1)*coord_n,1] = y
        t_input[i*coord_n:(i+1)*coord_n,2] = np.tile(dt*(i+1),coord_n)

    return b_input, t_input, target

def full_stacked_param_hard_windows_td(a_list,key,start,end,fpath,window_s=5):

    meshdir = 'cf'+str(a_list[0]).split('.')[1]
    mesh = dl.load_mesh(fpath/meshdir)
    coord_n = mesh['nodes'][:,0].shape[0]
    x = mesh['nodes'][:,0]
    y = mesh['nodes'][:,1]
    t_t = mesh['t'][start:end]

    b_input = np.empty((len(a_list)*int(t_t.shape[0]/window_s), window_s, 1+coord_n+75))
    target = np.empty((len(a_list)*int(t_t.shape[0]/window_s), window_s, coord_n))
    
    count = 0
    for inx,a in tqdm(enumerate(a_list)):
        meshdir = 'cf'+str(a).split('.')[1]
        full_data = dl.load_variables(fpath/meshdir)
        data = full_data[key]

        for id_t in range(0,len(t_t),window_s):
            id_t_temp = id_t + start
            ic = data[:,id_t_temp]

            solution_h = full_data['h'][:,id_t_temp+1:id_t_temp+1+window_s]
            h_bc = solution_h[:75,:].T

            a_window = np.full((window_s, 1), a)
            ic_window = np.tile(ic, (window_s, 1))

            b_input[count] = np.hstack([a_window,ic_window,h_bc])
            print(b_input.shape)
            target[count,:] = data[:,id_t_temp+1:id_t_temp+1+window_s].T

            count = count + 1

    dt = t_t[1] - t_t[0]
    
    t_input = np.empty((window_s, coord_n, 3))
    t_input[..., 0] = x[None, :]              
    t_input[..., 1] = y[None, :]             
    t_input[..., 2] = (np.arange(1, window_s+1) * dt)[:, None] 

    return b_input, t_input, target
    
# def full_multiple_param(a_list,key,start,end,fpath,window_s=5):

#     meshdir = 'cf'+str(a_list[0]).split('.')[1]
#     mesh = dl.load_mesh(fpath/meshdir)
#     coord_n = mesh['nodes'][:,0].shape[0]
#     x = mesh['nodes'][:,0]
#     y = mesh['nodes'][:,1]
#     t_t = mesh['t'][start:end]

#     b_par_input = np.empty((len(a_list)*int(t_t.shape[0]), 1))
#     b_bc_input = np.empty((len(a_list)*int(t_t.shape[0]), int(75*window_s)))
#     b_ic_input = np.empty((len(a_list)*int(t_t.shape[0]), coord_n))
#     target = np.empty((len(a_list)*int(t_t.shape[0]),coord_n*window_s))

#     t_t = mesh['t'][start:end-window_s]

#     count = 0
#     for inx,a in tqdm(enumerate(a_list)):
#         meshdir = 'cf'+str(a).split('.')[1]
#         full_data = dl.load_variables(fpath/meshdir)
#         data = full_data[key]

#         for id_t in range(0,len(t_t)):
#             id_t_temp = id_t + start
#             ic = data[:,id_t_temp]

#             solution_h = full_data['h'][:,id_t_temp+1:id_t_temp+1+window_s]
#             h_bc = solution_h[:75,:]

#             b_par_input[count] = a
#             b_ic_input[count] = ic
#             b_bc_input[count] = h_bc.flatten(order='F')
#             target[count,:] = data[:,id_t_temp+1:id_t_temp+1+window_s].flatten(order='F')

#             count = count + 1

#     t_input = np.empty((int(coord_n*window_s),3))
#     dt = t_t[1]-t_t[0]

#     t_input[0:coord_n,0] = x
#     t_input[0:coord_n,1] = y
#     t_input[0:coord_n,2] = np.tile(dt,coord_n)
    
#     for i in range(1,window_s):
#         t_input[i*coord_n:(i+1)*coord_n,0] = x
#         t_input[i*coord_n:(i+1)*coord_n,1] = y
#         t_input[i*coord_n:(i+1)*coord_n,2] = np.tile(dt*(i+1),coord_n)

#     return b_par_input, b_ic_input, b_bc_input, t_input, target
    
def full_multiple_param_hard_windows(a_list,key,start,end,fpath,window_s=5):

    meshdir = 'cf'+str(a_list[0]).split('.')[1]
    mesh = dl.load_mesh(fpath/meshdir)
    coord_n = mesh['nodes'][:,0].shape[0]
    x = mesh['nodes'][:,0]
    y = mesh['nodes'][:,1]
    t_t = mesh['t'][start:end]

    b_par_input = np.empty((len(a_list)*int(t_t.shape[0]/window_s), 1))
    b_bc_input = np.empty((len(a_list)*int(t_t.shape[0]/window_s), int(75*window_s)))
    b_ic_input = np.empty((len(a_list)*int(t_t.shape[0]/window_s), coord_n))
    target = np.empty((len(a_list)*int(t_t.shape[0]/window_s),coord_n*window_s))

    count = 0
    for inx,a in tqdm(enumerate(a_list)):
        meshdir = 'cf'+str(a).split('.')[1]
        full_data = dl.load_variables(fpath/meshdir)
        data = full_data[key]

        for id_t in range(0,len(t_t),window_s):
            id_t_temp = id_t + start
            ic = data[:,id_t_temp]

            solution_h = full_data['h'][:,id_t_temp+1:id_t_temp+1+window_s]
            h_bc = solution_h[:75,:]

            b_par_input[count] = a
            b_ic_input[count] = ic
            b_bc_input[count] = h_bc.flatten(order='F')
            target[count,:] = data[:,id_t_temp+1:id_t_temp+1+window_s].flatten(order='F')

            count = count + 1

    t_input = np.empty((int(coord_n*window_s),3))
    dt = t_t[1]-t_t[0]

    t_input[0:coord_n,0] = x
    t_input[0:coord_n,1] = y
    t_input[0:coord_n,2] = np.tile(dt,coord_n)
    
    for i in range(1,window_s):
        t_input[i*coord_n:(i+1)*coord_n,0] = x
        t_input[i*coord_n:(i+1)*coord_n,1] = y
        t_input[i*coord_n:(i+1)*coord_n,2] = np.tile(dt*(i+1),coord_n)

    return b_par_input, b_ic_input, b_bc_input, t_input, target
