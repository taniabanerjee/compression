import numpy as np
from numpy import linalg as LA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import math

import os
from PIL import Image

from Functions import *
from tqdm import tqdm

import adios2 as ad2
import xgc4py
import nanopq
import os


# In[2]:


def draw(frame):
    x = np.linspace(0, 38, 39)
    y = np.linspace(0, 38, 39)
    X, Y = np.meshgrid(x, y)
    plt.imshow(frame, origin='lower')
    #plt.imshow(frame, origin='upper')
    plt.colorbar()
    plt.contour(X, Y, frame, 5, origin='image', colors='white', alpha=0.5)


# In[3]:


with ad2.open('./d3d_coarse_v2/restart_dir/xgc.f0.00420.bp', 'r') as f:
    i_f = f.read('i_f')


# In[4]:


data_load = np.moveaxis(i_f, 1, 2)
print(data_load.shape)

all_planes = np.copy(data_load)

# changed for separate tucker

mu = np.mean(all_planes, axis=(2,3))
sig = np.std(all_planes, axis=(2,3))
all_planes = (all_planes - mu[:,:,np.newaxis, np.newaxis])/sig[:,:,np.newaxis, np.newaxis]

first_plane = all_planes[0].copy()
first_plane = first_plane.reshape(1, first_plane.shape[0], first_plane.shape[1], first_plane.shape[2])
with ad2.open('./mgard_inputs/normalimage/xgcmgard.f0.00420.bp', 'w') as fw:
    shape = first_plane.shape
    start = [0,]*len(first_plane.shape)
    count = first_plane.shape
    fw.write('i_f', first_plane.copy(), shape, start, count)

#os.system("./MGARD-XGC/build/test_xgc_5d asb 1e-03 /home/tania/ornl/mgard_inputs/normalimage/ xgcmgard.f0.00420.bp 1")
def rmse_error_err(x):

    mse = np.mean(x**2)

    return np.sqrt(mse)

with ad2.open('./MGARD-XGC/build/xgcmgard.f0.00420.bp.mgard', 'r') as fw:
    f_g = fw.read('i_f_5d')
err = abs(all_planes[0] - f_g[0][0])
print (np.histogram(err))
print ('L2:', np.linalg.norm(err))
print ('RMSE:', rmse_error_err(err))
print ('Linf:', np.max(err))
# In[5]:


xgcexp = xgc4py.XGC('./d3d_coarse_v2')


# In[6]:


def rmse_error(x, y):

    assert(x.shape == y.shape)
    mse = np.mean((x-y)**2)

    return np.sqrt(mse)


# In[8]:


def physics_loss_rmse(data, data_recon):
    batch_size = data.shape[0]

    f0_f = np.copy(data)
    den, u_para, T_perp, T_para, _, _ = xgcexp.f0_diag(0, batch_size, 1, f0_f)
        
    f0_f = np.copy(data_recon)
    den_recon, u_para_recon, T_perp_recon, T_para_recon, _, _ = xgcexp.f0_diag(0, batch_size, 1, f0_f)
    
    den_err = rmse_error(den_recon, den)
    u_para_err = rmse_error(u_para_recon, u_para)
    T_perp_err = rmse_error(T_perp_recon, T_perp)
    T_para_err = rmse_error(T_para_recon, T_para)

    return (den_err, u_para_err, T_perp_err, T_para_err)


# In[9]:


def compute_diff(x, x_):
    assert(x.shape == x_.shape)
    max_diff = np.max(np.abs(x - x_))
    max_v = np.max(np.abs(x))
    
    return max_diff/max_v


# In[10]:


def compute_diff_mine(x, x_):
    assert(x.shape == x_.shape)
    max_diff = np.max(np.abs(x - x_))
    max_v = np.max(np.abs(x_))
    
    return max_diff/max_v


# In[11]:


def physics_loss_Linf(data, data_recon):
    batch_size = data.shape[0]
    
    den = np.zeros((batch_size))
    u_para = np.zeros((batch_size))
    T_perp = np.zeros((batch_size))
    T_para = np.zeros((batch_size))
    
    den_recon = np.zeros((batch_size))
    u_para_recon = np.zeros((batch_size))
    T_perp_recon = np.zeros((batch_size))
    T_para_recon = np.zeros((batch_size))

    f0_f0 = np.copy(data)
    den, u_para, T_perp, T_para = xgc.f0_diag(isp=1, f0_f=f0_f0)
    f0_f0 = np.copy(data_recon)
    den_recon, u_para_recon, T_perp_recon, T_para_recon = xgc.f0_diag(isp=1, f0_f=f0_f0)
    
    den_err = compute_diff(den_recon, den)
    u_para_err = compute_diff(u_para_recon, u_para)
    T_perp_err = compute_diff(T_perp_recon, T_perp)
    T_para_err = compute_diff(T_para_recon, T_para)

    return (den_err, u_para_err, T_perp_err, T_para_err)


# In[12]:


def physics_loss_Linf_mine(data, data_recon):
    batch_size = data.shape[0]
    
    den = np.zeros((batch_size))
    u_para = np.zeros((batch_size))
    T_perp = np.zeros((batch_size))
    T_para = np.zeros((batch_size))
    
    den_recon = np.zeros((batch_size))
    u_para_recon = np.zeros((batch_size))
    T_perp_recon = np.zeros((batch_size))
    T_para_recon = np.zeros((batch_size))

    f0_f0 = np.copy(data)
    den, u_para, T_perp, T_para = xgc.f0_diag(isp=1, f0_f=f0_f0)
    f0_f0 = np.copy(data_recon)
    den_recon, u_para_recon, T_perp_recon, T_para_recon = xgc.f0_diag(isp=1, f0_f=f0_f0)
    
    den_err = compute_diff_mine(den_recon, den)
    u_para_err = compute_diff_mine(u_para_recon, u_para)
    T_perp_err = compute_diff_mine(T_perp_recon, T_perp)
    T_para_err = compute_diff_mine(T_para_recon, T_para)

    return (den_err, u_para_err, T_perp_err, T_para_err)


# In[13]:


def physics_loss_Linf_rmse(data, data_recon):
    batch_size = data.shape[0]
    
    den = np.zeros((batch_size))
    u_para = np.zeros((batch_size))
    T_perp = np.zeros((batch_size))
    T_para = np.zeros((batch_size))
    
    den_recon = np.zeros((batch_size))
    u_para_recon = np.zeros((batch_size))
    T_perp_recon = np.zeros((batch_size))
    T_para_recon = np.zeros((batch_size))

    f0_f0 = np.copy(data)
    den, u_para, T_perp, T_para = xgc.f0_diag(isp=1, f0_f=f0_f0)
    f0_f0 = np.copy(data_recon)
    den_recon, u_para_recon, T_perp_recon, T_para_recon = xgc.f0_diag(isp=1, f0_f=f0_f0)
    
    den_err = rmse_error(den_recon, den)
    u_para_err = rmse_error(u_para_recon, u_para)
    T_perp_err = rmse_error(T_perp_recon, T_perp)
    T_para_err = rmse_error(T_para_recon, T_para)

    return (den_err, u_para_err, T_perp_err, T_para_err)


# In[14]:


def f0f_err_max(data, data_recon):
    fmax = np.max(data, axis=(2,3))
    abserr = np.max(np.abs(data-data_recon), axis=(2,3))
    relabserr = abserr/fmax
    relabserr_max = np.max(relabserr, axis=(1))
    relabserr_max_avg = np.average(relabserr_max)
    
    return relabserr_max_avg


# In[15]:


def f0f_err_max_plane(data, data_recon):
    fmax = np.max(data, axis=(1,2))
    abserr = np.max(np.abs(data-data_recon), axis=(1,2))
    relabserr = abserr/fmax
    relabserr_max = np.max(relabserr)
    
    return relabserr_max

latent_dims = [32, 64, 128, 256]
threshold = [0, 10, 100, 1000, 10000, 100000, 10000000]
threshold = [1.0e-08, 1.0e-07, 1.0e-06, 1.0e-05, 1.0e-04, 1.0e-03, 1.0e-02, 1.0e-01, 1.0, 10, 100]
TuckerAE_Spectral_per_plane_error_no_clustering = np.load('./results/npy_files/TuckerAE_Spectral_per_plane_error_no_clustering.npy')
latent_dims
for k in range(len(latent_dims)):
    print ("Latent dimensions", k)
    err = abs(TuckerAE_Spectral_per_plane_error_no_clustering[0][k])
    #err = TuckerAE_Spectral_per_plane_error_no_clustering[0][k]
    #err = np.divide(err, all_planes[0])
    #err = abs(err * 100)
    print (np.histogram(err))
    print ('L2:', np.linalg.norm(err))
    print ('RMSE:', rmse_error_err(err))
    print ('Linf:', np.max(err))
    pixels_tbe = [len(err[err > t]) for t in threshold]
    pixels_tben = np.asarray([len(err[err < t]) for t in threshold])
    pixels_tben[1:] -= pixels_tben[:-1].copy()
    print (pixels_tbe)
    print (pixels_tben)
#    for t in threshold:
#        indices = np.unique(np.where(err > t)[0])
#        print ('Thresold:', t, len(indices))

k = 0
TuckerAE_Spectral_per_plane_residual = TuckerAE_Spectral_per_plane_error_no_clustering[0][0]
shape = TuckerAE_Spectral_per_plane_residual.shape
TuckerAE_Spectral_per_plane_residual = TuckerAE_Spectral_per_plane_residual.reshape(1, shape[0], shape[1], shape[2])
residual = TuckerAE_Spectral_per_plane_residual
with ad2.open('./mgard_inputs/residualnc/xgcmgardnc.f0.00420.bp', 'w') as fw:
    shape = residual.shape
    start = [0,]*len(residual.shape)
    count = residual.shape
    fw.write('i_f', residual.copy(), shape, start, count)

TuckerAE_Spectral_per_plane_error = np.load('./results/npy_files/TuckerAE_Spectral_per_plane_error_with_clustering.npy')
latent_dims
for k in range(len(latent_dims)):
    print ("Latent dimensions", k)
    err = abs(TuckerAE_Spectral_per_plane_error[0][k])
    #err = TuckerAE_Spectral_per_plane_error_no_clustering[0][k]
    #err = np.divide(err, all_planes[0])
    #err = abs(err * 100)
    print (np.histogram(err))
    print ('L2:', np.linalg.norm(err))
    print ('RMSE:', rmse_error_err(err))
    print ('Linf:', np.max(err))
    pixels_tbe = [len(err[err > t]) for t in threshold]
    pixels_tben = np.asarray([len(err[err < t]) for t in threshold])
    pixels_tben[1:] -= pixels_tben[:-1].copy()
    print (pixels_tbe)
    print (pixels_tben)
#    for t in threshold:
#        indices = np.unique(np.where(err > t)[0])
#        print ('Thresold:', t, len(indices))
k = 0
TuckerAE_Spectral_per_plane_residual = TuckerAE_Spectral_per_plane_error[0][0]
shape = TuckerAE_Spectral_per_plane_residual.shape
TuckerAE_Spectral_per_plane_residual = TuckerAE_Spectral_per_plane_residual.reshape(1, shape[0], shape[1], shape[2])
residual = TuckerAE_Spectral_per_plane_residual
with ad2.open('./mgard_inputs/residual/xgcmgard.f0.00420.bp', 'w') as fw:
    shape = residual.shape
    start = [0,]*len(residual.shape)
    count = residual.shape
    fw.write('i_f', residual.copy(), shape, start, count)

