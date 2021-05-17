import pandas as pd
import numpy as np
import scipy
import sys
#from numpy import linalg as LA
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import adios2 as ad2
import time
import csv
import os

# In[3]:


# function to plot 39 by 39 velocity histogram

def draw(frame):
    x = np.linspace(0, 38, 39)
    y = np.linspace(0, 38, 39)
    X, Y = np.meshgrid(x, y)
    plt.imshow(frame, origin='lower')
    #plt.imshow(frame, origin='upper')
    plt.colorbar()
    plt.contour(X, Y, frame, 5, origin='image', colors='white', alpha=0.5)


# In[4]:


bt = time.time()
# Read geometry information

with ad2.open('./d3d_coarse_v2/xgc.mesh.bp', 'r') as f:
    nnodes = int(f.read('n_n', ))
    ncells = int(f.read('n_t', ))
    rz = f.read('rz')
    conn = f.read('nd_connect_list')
    psi = f.read('psi')
    nextnode = f.read('nextnode')
    epsilon = f.read('epsilon')
    node_vol = f.read('node_vol')
    node_vol_nearest = f.read('node_vol_nearest')
    psi_surf = f.read('psi_surf')
    surf_idx = f.read('surf_idx')
    surf_len = f.read('surf_len')

# r, z are the coordinate of each mesh node
r = rz[:,0]
z = rz[:,1]
print (nnodes)


# In[5]:


# Load velocity histogram data

with ad2.open('./d3d_coarse_v2/restart_dir/xgc.f0.00420.bp', 'r') as f:
    i_f = f.read('i_f')


# In[6]:


data_load = np.moveaxis(i_f, 1, 2)
print(data_load.shape)


# In[ ]:


# Z-normalization of dataset

all_planes = np.copy(data_load)

# changed for separate tucker
#scaled_data = StandardScaler().fit_transform(all_planes.reshape(8*16395*39, 39))
#all_planes = scaled_data.reshape(8, 16395, 39, 39)
mu = np.mean(all_planes, axis=(2,3))
sig = np.std(all_planes, axis=(2,3))
all_planes = (all_planes - mu[:,:,np.newaxis, np.newaxis])/sig[:,:,np.newaxis, np.newaxis]
#print(all_planes.shape, mu.shape, sig.shape)


labels = np.loadtxt(sys.argv[1]).astype(int)

(unique, counts) = np.unique(labels, return_counts=True)
frequencies = np.asarray((unique, counts)).T
nround = frequencies[np.argsort(-frequencies[:, 1])][:, 0]
n_comp = 1
labelset = []
num_clusters = unique.size
threshold = 16395 * 1/16
for i in range(num_clusters):
    freq = frequencies[nround[i]][1]
    if (freq > threshold):
        n_comp = n_comp+1
        labelset.append(frequencies[nround[i]][0])
    else:
        break

print(labels.shape)
for i in range(16395):
    if (labels[i] in labelset):
        labels[i] = labelset.index(labels[i])
    else:
        labels[i] = n_comp-1

for i in range(n_comp):
    m_indices = np.where(labels == i)
    images = all_planes[0][m_indices]
    centroid = np.mean(images, axis=0)
    draw(centroid)
    plt.show()
    residuals = np.subtract(images, centroid)
    print (np.sum(residuals))

