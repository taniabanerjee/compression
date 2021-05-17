#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy
#from numpy import linalg as LA
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.utils.data import DataLoader
#import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.tri as tri
#import pymesh
#import math

#import os
#from PIL import Image

#from Functions import *
#from tqdm import tqdm

import adios2 as ad2
import time
import csv
import os
import sys
#import nanopq


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


# In[7]:


# Read meta data and other inherent funciton

#xgcexp = xgc4py.XGC('./d3d_coarse_v2')

# In[8]:
def similarity(w, sigma):
    return np.exp(-w/(sigma*sigma))

def distance(xi, xj):
    return np.sum(np.square(np.subtract(xi, xj)))

# Plot mesh graph and mesh point

plt.figure(figsize=[8,16])

trimesh = tri.Triangulation(r, z, conn)
plt.triplot(trimesh, alpha=0.2)
plt.plot(r[8052], z[8052], 'x')
#plt.show()

draw(all_planes[0][0])
#plt.show()
draw(all_planes[0][8052])
plt.title('distance: {}'.format(distance(all_planes[0][0], all_planes[0][8052])))
#plt.show()

data = all_planes[0]
data_reshape = data.reshape(data.shape[0], 39*39)
data_mins = data_reshape.min(axis = 1)
print(data_mins.shape)
data_maxs = data_reshape.max(axis = 1)
print(data_maxs.shape)
# shape is (16395,)
plt.hist(data_mins, bins=100)
plt.xlabel('min values')
plt.ylabel('pixels')
plt.title('Minimum value for each 39x39 image')
#plt.show()

plt.hist(data_maxs, bins=100)
plt.xlabel('max values')
plt.ylabel('pixels')
plt.title('Maximum value for each 39x39 image')
#plt.show()

plane = 0
knn = 1
sigmalist = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.7, 2.0]
sigma = sigmalist[0]
edgelist = []
num_points = rz.shape[0]
orig_points = num_points
#num_points = 100
ptlist = [i for i in range(num_points)]
mask = np.ones(num_points, dtype=bool)
print ('Find the edge lists')
mypath = './edgelist.csv'
for i in range(0, num_points):
    trianglelist = []
    verticeslist = []
    edgelisti = []
    for c in range (0, 3):
        trianglelist = trianglelist + np.where(conn[:, c]==i)[0].tolist()
    verticeslist.extend([conn[x, :].tolist() for x in trianglelist])
    for el in verticeslist:
        edgelisti.extend(el)
    edgelisti = list(dict.fromkeys(edgelisti))
    edgelist.append(sorted(edgelisti))

iter_start = 0
if (len(sys.argv) == 2):
    y_labels = np.loadtxt(sys.argv[1]).astype(int)
    (unique, counts) = np.unique(y_labels, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    nround = frequencies[np.argsort(-frequencies[:, 1])][:, 0]
    mask = y_labels != nround[0]
    num_points = np.count_nonzero(mask)
    ptlist = np.where(mask)[0]
    iter_start = 1
for iteration in range (iter_start, 8):
    W = np.zeros([num_points,num_points])
    WNN = np.zeros([num_points,num_points])
    print ('Compute the weight matrix')
    for i in range(0, num_points):
        W[i,i] = 1
        edgelisti = edgelist[ptlist[i]]
        for j in range (i+1, num_points):
            if (ptlist[j] in edgelisti):
                w = distance(all_planes[plane][ptlist[i]], all_planes[plane][ptlist[j]])
                W[i, j] = similarity(w, sigma)
                W[j, i] = W[i, j]
    print ('Compute the KNN neighbors')
    WNN = W.copy()
    for i in range(1, knn):
        WNN = scipy.linalg.blas.dgemm(1.0, WNN, W)
    
    print ('Compute the diagonal matrix degree matrix')
    D = np.zeros([num_points,num_points])
    for i in range(0, num_points):
        D[i,i] = np.sum(WNN[i,:])
    Dinv = np.linalg.inv(D)
    
    print ('Compute the Laplacian matrix L')
    L = D-WNN
    print ('Compute Dinverse times L')
    #A = np.einsum('ij,jk->ik', Dinv, L)
    A = scipy.linalg.blas.dgemm(1.0, Dinv, L)
    
    print ('Compute the eigenvalues and eigenvectors')
    #vals, vecs = scipy.linalg.eig(A)
    vals, vecs = scipy.linalg.eig(A)
    vecs = vecs[:,np.argsort(vals)].real
    vals = vals[np.argsort(vals)].real
    
    with open('./eigenvalues_knn{}_sigma{}_plane{}.csv'.format(knn, sigma, plane), mode='w') as edgefile:
        csv_writer = csv.writer(edgefile, delimiter=',')
        csv_writer.writerow(vals)
    
    print ('Compute Kmeans clustering')
    num_clusters=len([e for e in vals if e < 1.0e-14])
    print (num_clusters)
    #num_clusters=16
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(vecs[:,1:num_clusters+1])
    y_labels = kmeans.labels_
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    pallette=['r', 'b', 'g', 'y']
    for i in range(num_clusters):
        ax.scatter3D(vecs[y_labels==i,1], vecs[y_labels==i, 2], vecs[y_labels==i, 3], c=pallette[i%4])
    #ax.set_zlim(-2, 2)
    ax.set_xlabel('1st dimension')
    ax.set_ylabel('2nd dimension')
    ax.set_zlabel('3rd dimension')
    plt.savefig('embedded_knn{}_sigma{}_plane{}_clus{}_iter{}.png'.format(knn, sigma, plane, num_clusters, iteration), format='png')
    #plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot()
    pallette=['r', 'b', 'g', 'y']
    for i in range(num_clusters):
        plt.scatter(vecs[y_labels==i,1], vecs[y_labels==i, 2], s=5, c=pallette[i%4], marker='o')
    #ax.set_zlim(-2, 2)
    ax.set_xlabel('1st dimension')
    ax.set_ylabel('2nd dimension')
    plt.savefig('embedded_knn{}_sigma{}_plane{}__clus{}_iter{}_2D.png'.format(knn, sigma, plane, num_clusters, iteration), format='png')
    #plt.show()
    
    # In[11]:


    plt.figure(figsize=[8,16])
    
    trimesh = tri.Triangulation(r, z, conn)
    plt.triplot(trimesh, alpha=0.2)
    
    colormap = plt.cm.Dark2
    colors = plt.cm.rainbow(np.linspace(0, 1, num_clusters))
    
    for i in range(num_points):
        plt.plot(r[i], z[i], 'x', c=colors[y_labels[i]])
        '''
        if (y_labels[i] == 0):
            plt.plot(r[i], z[i], 'x', c='r')
        elif (y_labels[i]%4 == 1):
            plt.plot(r[i], z[i], 'x', c='b')
        elif (y_labels[i]%4 == 2):
            plt.plot(r[i], z[i], 'x', c='g')
        elif (y_labels[i]%4 == 3):
            plt.plot(r[i], z[i], 'x', c='y')
       ''' 
    #plt.plot(r[k[0]], z[k[0]], 's', c=colormap(i%colormap.N))
    plt.savefig('domain_knn{}_sigma{}_plane{}_clus{}_iter{}.png'.format(knn, sigma, plane, num_clusters, iteration), format='png')
    #plt.show()
    print (time.time()-bt)
    
    (unique, counts) = np.unique(y_labels, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    nround = frequencies[np.argsort(-frequencies[:, 1])][:, 0]
    top_labels = y_labels
    if (num_points < orig_points):
        top_labels = np.full([orig_points], -1, dtype=int)
        for i in range(num_points):
            top_labels[ptlist[i]] = y_labels[i]
    mask = top_labels != nround[0]
    
    print(frequencies)
    
    with open('labels_knn{}_sigma{}_plane{}_clus{}_iter{}.txt'.format(knn, sigma, plane, num_clusters, iteration), 'w') as filehandle:
        for label in top_labels:
            filehandle.write('%s\n' % label)
        for u, c in zip(unique, counts):
            filehandle.write('{}: {}\n'.format(u, c))
    sigma = sigmalist[iteration]
'''
# In[13]:


y_labels


# In[14]:


cluster = []
index_cluster = []

temp = np.moveaxis(all_planes, 0,1)
print(temp.shape)

for i in range(n_comp):
    clus = []
    idx = []
    for j in range(temp.shape[0]):
        if (y_labels[j] == i):
            temp1 = temp[j]
            clus.append(temp1)
            idx.append(j)
    
    clus = np.array(clus)
    clus = np.moveaxis(clus, 0,1)
    print(clus.shape)
    cluster.append(clus)
    
    idx = np.array(idx)
    index_cluster.append(idx)
    
del temp, temp1


# In[15]:


draw(cluster[2][0][2100])
plt.show()


# In[16]:


draw(all_planes[0][index_cluster[2][2100]])
plt.show()


# In[17]:


len(cluster), len(cluster[0]), len(cluster[1]), len(cluster[2]), len(cluster[3])


# In[18]:


len(index_cluster[0])


# In[ ]:



'''
