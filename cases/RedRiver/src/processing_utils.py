import numpy as np
import data_loader as dl 
from tqdm import tqdm

def multiple_param(a_list, latent_data, time_snaps, start, end, t_skip,fpath, in_nodes, out_nodes):

    t_t = time_snaps[start:end]

    for inx,a in enumerate(a_list):
        fname = f'Inset_T5.198e+06_mn{a:.6f}.npz'
        full_data = dl.load_variables_adh(fpath/fname, comp_names=['S_vx', 'S_vy', 'S_dep'], t_start = 480) ## Ignoring the first five days of simulation data
        data = latent_data[inx,:,:]
        ic = data[0,:]
        
        for id_t,it in tqdm(enumerate(t_t)):
            
            id_t_temp_1 = id_t 
            id_t_temp_2 = id_t + start

            solution_u = full_data['S_vx'][:,::t_skip][:,id_t_temp_2]
            solution_v = full_data['S_vy'][:,::t_skip][:,id_t_temp_2]
            solution_h = full_data['S_dep'][:,::t_skip][:,id_t_temp_2]
            dis_bc = np.sqrt(solution_u[in_nodes]**2 + solution_v[in_nodes]**2) * solution_h[in_nodes]
            
               
            if inx==0 and id_t_temp_2==start:
                b_par_input = a
                b_ic_input = ic
                b_bc1_input = dis_bc
                b_bc2_input = solution_h[out_nodes]
                t_input = it
                target = data[id_t_temp_1,:]
                
            else:
                b_par_input = np.vstack([b_par_input, a])
                b_ic_input = np.vstack([b_ic_input, ic])
                b_bc1_input = np.vstack([b_bc1_input, dis_bc])
                b_bc2_input = np.vstack([b_bc2_input, solution_h[out_nodes]])
                t_input = np.vstack([t_input, it])
                target = np.vstack([target, data[id_t_temp_1,:]])  
            
    return b_par_input, b_ic_input, b_bc1_input, b_bc2_input, t_input, target


### Modifying this function for 2D AdH
def multiple_param_windows(a_list,latent_data,time_snaps,start,end,t_skip,fpath,
                           in_nodes, out_nodes, window_s=5):
 
    for inx,a in enumerate(a_list):
        fname = f'Inset_T5.198e+06_mn{a:.6f}.npz'
        full_data = dl.load_variables_adh(fpath/fname, comp_names=['S_vx', 'S_vy', 'S_dep'], t_start = 480) ## Ignoring the first five days of simulation data
        
        data = latent_data[inx,:,:]
        
        # for id_t,it in tqdm(enumerate(t_t)):
        for id_t in tqdm(range(end - start - window_s)):
            ic = data[id_t,:]

            window = 1

            for j in range(window_s):
                id_t_temp_1 = id_t + window
                id_t_temp_2 = id_t + start + window

                solution_u = full_data['S_vx'][:,::t_skip][:,id_t_temp_2]
                solution_v = full_data['S_vy'][:,::t_skip][:,id_t_temp_2]
                solution_h = full_data['S_dep'][:,::t_skip][:,id_t_temp_2]
                dis_bc = np.sqrt(solution_u[in_nodes]**2 + solution_v[in_nodes]**2) * solution_h[in_nodes]

                if inx==0 and id_t_temp_2==start+1:
                    ## Should be only called once
                    b_par_input = a
                    b_ic_input = ic
                    b_bc1_input = dis_bc
                    b_bc2_input = solution_h[out_nodes]
                    t_input = time_snaps[start + id_t + window] - time_snaps[start + id_t]
                    target = data[id_t_temp_1,:]
                    
                else:
                    b_par_input = np.vstack([b_par_input, a])
                    b_ic_input = np.vstack([b_ic_input, ic])
                    b_bc1_input = np.vstack([b_bc1_input, dis_bc])
                    b_bc2_input = np.vstack([b_bc2_input, solution_h[out_nodes]])
                    t_input = np.vstack([t_input, time_snaps[start + id_t + window] - time_snaps[start + id_t]])
                    target = np.vstack([target, data[id_t_temp_1,:]])  
                
                window = window + 1
            
    return b_par_input, b_ic_input, b_bc1_input, b_bc2_input, t_input, target


