import numpy as np
import scipy
import netCDF4 as nc

def load_mesh(fpath):
    """
    Script to load ADCIRC variables from  
    fort.63 and fort.64 files stored in 
    NetCDF4 format.
    
    Extracts x, y nodal coordinates, 
    elements (triangles), time (t), 
    and bathymetry (bathy). Also returns
    a "nodes" array with x and y 
    coordinates stacked vertically.
    
    Returns a python dictionary.
    """
    data ={}
    
    ## Load depth and mesh
    with nc.Dataset(fpath / "fort.63.nc") as depthfile:
        data['triangles'] = depthfile.variables['element'][:]
        data['t'] = depthfile.variables['time'][:]
        nodes_x = depthfile['x'][:]; 
        nodes_y = depthfile.variables['y'][:]
        data['nodes'] = np.vstack((nodes_x, nodes_y)).T 
        
        ## Load velocities and bathymetry
    with nc.Dataset(fpath / "fort.64.nc") as velfile:
        data['bathy'] = velfile.variables['depth'][:]
 
        
    return data

def load_variables(fpath, scaling = False):
    """
    Script to load ADCIRC variables from  
    fort.63 and fort.64 files stored in 
    NetCDF4 format.
    
    Extracts time, water elevation (h),
    x (u) and y (v) velocities, and 
    bathymetry (bathy).
    
    Returns a python dictionary. By default,
    ADCIRC stores water elevation (h) w.r.t. 
    geoid. If scaling = True, returns
    h = h + bathy.
    """
    data ={}
    
    ## Load depth and mesh
    with nc.Dataset(fpath / "fort.63.nc") as depthfile:
        data['t'] = depthfile.variables['time'][:]
        data['h'] = depthfile.variables['zeta'][:].T
    
    ## Load velocities and bathymetry
    with nc.Dataset(fpath / "fort.64.nc") as velfile:
        data['u'] = velfile.variables['u-vel'][:].T
        data['v'] = velfile.variables['v-vel'][:].T
        data['bathy'] = velfile.variables['depth'][:]
        if scaling:
            data['h'] = data['h'] + np.outer(data['bathy'], np.ones(data['t'].shape))
        
    return data
