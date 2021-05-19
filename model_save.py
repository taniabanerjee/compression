#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

r = rz[:,0]
z = rz[:,1]
print (nnodes)


# In[4]:


with ad2.open('./d3d_coarse_v2/restart_dir/xgc.f0.00420.bp', 'r') as f:
    i_f = f.read('i_f')


# In[5]:


data_load = np.moveaxis(i_f, 1, 2)
print(data_load.shape)


# In[6]:


xgcexp = xgc4py.XGC('./d3d_coarse_v2')


# In[7]:


import f0_diag_test
xgc = f0_diag_test.XGC_f0_diag(xgcexp)


# In[8]:


all_planes = np.copy(data_load)

# changed for separate tucker

#mu = np.mean(all_planes, axis=(2,3))
#sig = np.std(all_planes, axis=(2,3))
#all_planes = (all_planes - mu[:,:,np.newaxis, np.newaxis])/sig[:,:,np.newaxis, np.newaxis]
#print(all_planes.shape, mu.shape, sig.shape)
all_planes = all_planes/1.0e+15


# In[133]:


Epochs = 100
Lr_Rate = 1e-3
Batch_Size = 128


# In[134]:


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent):
        super(Autoencoder, self).__init__()
        
        self.enc1 = nn.Linear(in_features=input_dim, out_features=latent)
    
    def encode(self, x):
        x = self.enc1(x)        
        return x
    
    def decode(self, x):        
        x = F.linear(x, weight=self.enc1.weight.transpose(0,1))        
        return x

    def forward(self, x):
        
        latent = self.encode(x)
        recon = self.decode(latent)

        return recon


# In[135]:


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


# In[150]:


def training(model, train_loader, Epochs):
    train_loss = []
    best_loss = 100
    for epoch in range(Epochs):
        running_loss = 0.0
        for data in train_loader:
            data = data[0].to(device)            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(train_loader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.6f}'.format(
            epoch+1, Epochs, loss))
        if (loss < best_loss):
            torch.save(model.state_dict(), './model/best-model-parameters.pt')
            best_loss = loss
            print('best loss: ', best_loss)

    return train_loss


# In[137]:


def test_reconstruct(model, test_loader, num_instance, input_dim):
    #reconstructions = np.zeros((78*16,num_coeffs))
    reconstructions = []
    
    temp = 0
    for batch in test_loader:
        data = batch[0]
        data = data.to(device)
        outputs = model(data)
        if (temp == 0):
            print('Tensor type:', outputs[0][0].dtype)
            temp += 1
        reconstructions.append(outputs.cpu().data.numpy())
        #reconstructions[count] = outputs.cpu().numpy()
        #count = count + 1
        
    decoded_vectors = np.zeros((num_instance,input_dim), dtype=np.float32)
    count = 0
    for i in range(len(reconstructions)):
        for j in range(len(reconstructions[i])):
            decoded_vectors[count] = np.float32(reconstructions[i][j])
            count = count + 1
    
    return decoded_vectors


# In[138]:


def encoding(model, test_loader, num_instance, latent_dim):
    #reconstructions = np.zeros((78*16,num_coeffs))
    latent_vectors = []
    
    #count = 0
    for batch in test_loader:
        data = batch[0]
        data = data.to(device)
        outputs = model.encode(data)
        latent_vectors.append(outputs.cpu().data.numpy())
        #reconstructions[count] = outputs.cpu().numpy()
        #count = count + 1
    
    encoded_vectors = np.zeros((num_instance,latent_dim), dtype=np.float32)
    count = 0
    for i in range(len(latent_vectors)):
        for j in range(len(latent_vectors[i])):
            encoded_vectors[count] = np.float32(latent_vectors[i][j])
            count = count + 1  
    
    return encoded_vectors


# In[139]:


def decoding(model, test_loader, num_instance, input_dim):
    #reconstructions = np.zeros((78*16,num_coeffs))
    reconstructions = []    

    for batch in test_loader:
        data = batch[0]
        data = data.to(device)
        outputs = model.decode(data)
        reconstructions.append(outputs.cpu().data.numpy())
        #reconstructions[count] = outputs.cpu().numpy()
        #count = count + 1
    
    decoded_vectors = np.zeros((num_instance,input_dim), dtype=np.float32)
    count = 0
    for i in range(len(reconstructions)):
        for j in range(len(reconstructions[i])):
            decoded_vectors[count] = np.float32(reconstructions[i][j])
            count = count + 1
    
    return decoded_vectors


# In[140]:


def rmse_error(x, y):

    assert(x.shape == y.shape)
    mse = np.mean((x-y)**2)

    return np.sqrt(mse)


# In[141]:


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


# In[142]:


def compute_diff(x, x_):
    assert(x.shape == x_.shape)
    max_diff = np.max(np.abs(x - x_))
    max_v = np.max(np.abs(x))
    
    return max_diff/max_v


# In[143]:


def compute_diff_mine(x, x_):
    assert(x.shape == x_.shape)
    max_diff = np.max(np.abs(x - x_))
    max_v = np.max(np.abs(x_))
    
    return max_diff/max_v


# In[144]:


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


# In[145]:


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


# In[146]:


def f0f_err_max_plane(data, data_recon):
    fmax = np.max(data, axis=(1,2))
    abserr = np.max(np.abs(data-data_recon), axis=(1,2))
    relabserr = abserr/fmax
    relabserr_max = np.max(relabserr)
    
    return relabserr_max


# In[23]:


labels_tania3 = np.loadtxt('labels_sig0.3.txt')
labels_tania3


# In[24]:


labels = np.zeros((16395), dtype = int)
print(labels.shape)

for i in range(16395):
    if (labels_tania3[i] == 0):
        labels[i] = 0
    #elif ((labels_tania[i] != 0) and (labels_tania[i] != 775)):
    #elif ((labels_tania2[i] != 0) and (labels_tania2[i] != 59)):
    elif (labels_tania3[i] != 0):
        labels[i] = 1


# In[25]:


np.count_nonzero(labels == 0), np.count_nonzero(labels == 1)


# In[31]:


X_train_tucker = np.zeros((16395,39,39))


# In[32]:


n_comp = 2
cluster_pixel = []

temp = all_planes[0]

for i in range(n_comp):
    clus = []
    for j in range(temp.shape[0]):
        if (labels[j] == i):
            clus.append(temp[j])
    
    clus = np.array(clus)
    print(clus.shape)
    cluster_pixel.append(clus)


# In[211]:


# Tucker AE - Per plane, Full Tucker coeffs

latent_dims = [32, 64, 128, 256]
#latent_dims = [32]

TuckerAE_all = []

#for p in range(all_planes.shape[0]):
for p in range(1):
    
    X_train = all_planes[p]
    
    train_num = X_train.shape[0]
    
    results = []

    tucker_data = np.array(X_train).reshape([X_train.shape[0],39,39])
    
    tqdm.write('converting data to tucker basis...')
    coeff_tucker, factors = tucker(tucker_data)

    experiments = [
        {
            "num_coeffs": 1521
        }
    ]    

    for experiment in experiments:
        num_coeffs = experiment['num_coeffs']
        
        tqdm.write('tucker_coeffs: '+str(num_coeffs))

        top_coeffs = tucker_top_coeffs(coeff_tucker, num_coeffs)

        X_train_tucker = to_tucker_basis(np.array(X_train).reshape([X_train.shape[0],39,39]), factors, top_coeffs)
        
        print(X_train.shape)
        print(X_train_tucker.shape)
        
        n_comp = 2
        cluster = []
        index_cluster = []
        
        for i in range(n_comp):
            clus = []
            idx = []
            for j in range(X_train_tucker.shape[0]):
                if (labels[j] == i):
                    clus.append(X_train_tucker[j])
                    idx.append(j)
    
            clus = np.array(clus)
            print(clus.shape)
            cluster.append(clus)
    
            idx = np.array(idx)
            index_cluster.append(idx)
        
        #X_train_tucker = X_train_tucker.reshape([700,16,num_coeffs])
        #X_test_tucker = X_test_tucker.reshape([78,16,num_coeffs])
        
        training_data_tucker = torch.utils.data.TensorDataset(torch.Tensor(X_train_tucker))
        
        training_loader_tucker = DataLoader(training_data_tucker,
                                            batch_size=Batch_Size,
                                            shuffle=False,
                                            pin_memory=True)
        
        #train_physics_err_all = []
        #train_f0f_err_all = []
        TuckerAE_cluster1_all = []
        TuckerAE_cluster2_all = []
        
        for j in range(len(latent_dims)):
            TuckerAE_cluster1 = []
            TuckerAE_cluster2 = []
            for c in range(n_comp):                
                
                latent_dim = latent_dims[j]
                model = Autoencoder(X_train_tucker.shape[1], latent_dim)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=Lr_Rate)
                device = get_device()
                model.to(device)
                print(model)
                
                train_loss = training(model, training_loader_tucker, Epochs)            
            
                                
                training_data_tucker_cluster = torch.utils.data.TensorDataset(torch.Tensor(cluster[c]))
                
                training_loader_tucker_cluster = DataLoader(training_data_tucker_cluster,
                                                            batch_size=Batch_Size,
                                                            shuffle=False,
                                                            pin_memory=True)
                if (c==1):
                    train_loss = training(model, training_loader_tucker_cluster, Epochs)
                
                #model = Autoencoder(*args, **kwargs)
                model.load_state_dict(torch.load('./model/best-model-parameters.pt'))
                model.eval()
                
                latent_train = encoding(model, training_loader_tucker_cluster, cluster[c].shape[0], latent_dim)
                
                print('latent train shape: ', latent_train.shape)
                
                def train_quantizer():
                    seed = np.random.randint(100)
                    pq = nanopq.PQ(M=latent_dim, Ks=256, verbose=True)
                    pq.fit(vecs=latent_train, iter=20, seed=seed)
                    return pq

                pq = train_quantizer()
                #pq_dic.append(pq.codewords)
                
                latent_train_quan = pq.encode(vecs=latent_train)
                
                latent_train_dequan = pq.decode(codes=latent_train_quan)
                
                training_latent = torch.utils.data.TensorDataset(torch.Tensor(latent_train_dequan))
                
                training_loader_latent = DataLoader(training_latent,
                                                    batch_size=Batch_Size,
                                                    shuffle=False,
                                                    pin_memory=True)
        
                recon_train_tucker = decoding(model, training_loader_latent, cluster[c].shape[0], cluster[c].shape[1])
                print('recon train shape: ', recon_train_tucker.shape)            
        
                #train_recon = from_tucker_basis(recon_train_tucker, factors, top_coeffs, data_shape=[cluster[c].shape[0],39,39])
                if (c == 0):
                    #TuckerAE_cluster1.append(train_recon)
                    TuckerAE_cluster1.append(recon_train_tucker)
                elif (c == 1):
                    #TuckerAE_cluster2.append(train_recon)
                    TuckerAE_cluster2.append(recon_train_tucker)
            
            TuckerAE_cluster1_all.append(TuckerAE_cluster1)
            TuckerAE_cluster2_all.append(TuckerAE_cluster2)
            #TuckerAE_all.append(TuckerAE_cluster)
                


# In[212]:


TuckerAE_cluster1_all = np.array(TuckerAE_cluster1_all)
TuckerAE_cluster1_all.shape


# In[213]:


TuckerAE_cluster2_all = np.array(TuckerAE_cluster2_all)
TuckerAE_cluster2_all.shape


# In[214]:


TuckerAE_cluster_combine_Tucker = np.zeros((4,1,16395,1521))

for i in range(n_comp):
    for j in range(1):
        for k in range(len(latent_dims)):
            if (i == 0):
                temp = np.array(TuckerAE_cluster1_all[k][j])
                print(temp.shape)
            elif (i == 1):
                temp = np.array(TuckerAE_cluster2_all[k][j])
                print(temp.shape)
            for l in range(temp.shape[0]):
                TuckerAE_cluster_combine_Tucker[k][j][index_cluster[i][l]] = temp[l]


# In[215]:


TuckerAE_cluster_combine_Tucker.shape


# In[216]:


TuckerAE_cluster_combine = np.zeros((4,1,16395,39,39))

for i in range(4):
    for j in range(1):
        train_recon = from_tucker_basis(TuckerAE_cluster_combine_Tucker[i][j], factors, top_coeffs, data_shape=[16395,39,39])
        TuckerAE_cluster_combine[i][j] = train_recon


# In[158]:


np.save('./results/npy_files/TuckerAE_cluster_combine.npy', TuckerAE_cluster_combine)


# In[159]:


draw(all_planes[0][8050])


# In[163]:


draw(TuckerAE_cluster_combine[3][0][8050])


# In[217]:


del TuckerAE_cluster1_all, TuckerAE_cluster2_all


# In[218]:


n_comp = 2
TuckerAE_cluster1_all = []
TuckerAE_cluster2_all = []

temp = np.moveaxis(TuckerAE_cluster_combine, 1,2)
print(temp.shape)
temp = np.moveaxis(temp, 0,1)
print(temp.shape)

for i in range(n_comp):
    clus = []
    for j in range(temp.shape[0]):
        if (labels[j] == i):
            clus.append(temp[j])
    
    clus = np.array(clus)
    print(clus.shape)
    clus = np.moveaxis(clus, 0,1)
    print(clus.shape)
    clus = np.moveaxis(clus, 1,2)
    print(clus.shape)
    if (i == 0):
        TuckerAE_cluster1_all.append(clus)
    elif (i == 1):
        TuckerAE_cluster2_all.append(clus)
        
TuckerAE_cluster1_all = np.array(TuckerAE_cluster1_all[0])
TuckerAE_cluster2_all = np.array(TuckerAE_cluster2_all[0])


# In[219]:


TuckerAE_cluster1_all.shape, TuckerAE_cluster2_all.shape


# In[57]:


np.save('./results/npy_files/TuckerAE_cluster1_all_plane1.npy', TuckerAE_cluster1_all)
np.save('./results/npy_files/TuckerAE_cluster2_all_plane1.npy', TuckerAE_cluster2_all)


# In[98]:


TuckerAE_cluster1_all = np.load('./results/npy_files/TuckerAE_cluster1_all_plane1.npy')
TuckerAE_cluster2_all = np.load('./results/npy_files/TuckerAE_cluster2_all_plane1.npy')


# In[99]:


TuckerAE_cluster1_all.shape, TuckerAE_cluster2_all.shape


# In[220]:


compression_ratio = []
for i in range(len(latent_dims)):
    compression_ratio.append(64.*39*39./(8.*latent_dims[i]))

print(compression_ratio)


# In[76]:


cluster[0].shape


# In[168]:


draw(cluster_pixel[0][10000])


# In[169]:


draw(TuckerAE_cluster1_all[3][0][10000])


# In[40]:


all_planes.shape, TuckerAE_cluster1_all[1,0].shape, len(cluster_pixel), cluster_pixel[0].shape


# In[221]:


f0f_TuckerAE_cluster1 = []
f0f_rmse_TuckerAE_cluster1 = []
f0f_TuckerAE_cluster2 = []
f0f_rmse_TuckerAE_cluster2 = []

for i in range(len(latent_dims)):
    f0f_TuckerAE_cluster1.append(f0f_err_max_plane(cluster_pixel[0], TuckerAE_cluster1_all[i,0]))
    f0f_rmse_TuckerAE_cluster1.append(rmse_error(cluster_pixel[0], TuckerAE_cluster1_all[i,0]))
    f0f_TuckerAE_cluster2.append(f0f_err_max_plane(cluster_pixel[1], TuckerAE_cluster2_all[i,0]))
    f0f_rmse_TuckerAE_cluster2.append(rmse_error(cluster_pixel[1], TuckerAE_cluster2_all[i,0]))

f0f_TuckerAE_cluster1 = np.array(f0f_TuckerAE_cluster1)
f0f_rmse_TuckerAE_cluster1 = np.array(f0f_rmse_TuckerAE_cluster1)
f0f_TuckerAE_cluster2 = np.array(f0f_TuckerAE_cluster2)
f0f_rmse_TuckerAE_cluster2 = np.array(f0f_rmse_TuckerAE_cluster2)


# In[222]:


f0f_TuckerAE_cluster1, f0f_TuckerAE_cluster2


# In[223]:


f0f_rmse_TuckerAE_cluster1, f0f_rmse_TuckerAE_cluster2


# In[175]:


TuckerAE_cluster1_all.shape, TuckerAE_cluster2_all.shape


# In[103]:


TuckerAE_cluster_combine = np.zeros((4,1,16395,39,39))

for i in range(n_comp):
    for j in range(1):
        for k in range(len(latent_dims)):
            if (i == 0):
                temp = np.array(TuckerAE_cluster1_all[k][j])
                print(temp.shape)
            elif (i == 1):
                temp = np.array(TuckerAE_cluster2_all[k][j])
                print(temp.shape)
            for l in range(temp.shape[0]):
                TuckerAE_cluster_combine[k][j][index_cluster[i][l]] = temp[l]


# In[115]:


draw(all_planes[0][8050])


# In[152]:


draw(TuckerAE_cluster_combine[3][0][8050])


# In[224]:


f0f_TuckerAE_combine = []
f0f_rmse_TuckerAE_combine = []

for i in range(len(latent_dims)):
    f0f_TuckerAE_combine.append(f0f_err_max_plane(all_planes[0], TuckerAE_cluster_combine[i,0]))
    f0f_rmse_TuckerAE_combine.append(rmse_error(all_planes[0], TuckerAE_cluster_combine[i,0]))

f0f_TuckerAE_combine = np.array(f0f_TuckerAE_combine)
f0f_rmse_TuckerAE_combine = np.array(f0f_rmse_TuckerAE_combine)


# In[225]:


f0f_TuckerAE_combine, f0f_rmse_TuckerAE_combine


# In[111]:


np.min(TuckerAE_cluster_combine), np.max(TuckerAE_cluster_combine), TuckerAE_cluster_combine.shape


# In[112]:


TuckerAE_plane1 = np.load('./results/npy_files/TuckerAE_plane1.npy')
TuckerAE_plane1.shape


# In[179]:


np.min(TuckerAE_plane1), np.max(TuckerAE_plane1)


# In[113]:


f0f_TuckerAE_plane1 = []
f0f_rmse_TuckerAE_plane1 = []

for i in range(len(latent_dims)):
    f0f_TuckerAE_plane1.append(f0f_err_max_plane(all_planes[0], TuckerAE_plane1[i,0]))
    f0f_rmse_TuckerAE_plane1.append(rmse_error(all_planes[0], TuckerAE_plane1[i,0]))

f0f_TuckerAE_plane1 = np.array(f0f_TuckerAE_plane1)
f0f_rmse_TuckerAE_plane1 = np.array(f0f_rmse_TuckerAE_plane1)


# In[114]:


f0f_TuckerAE_plane1, f0f_rmse_TuckerAE_plane1


# In[130]:


temp = np.moveaxis(TuckerAE_plane1, 1,2)
print(temp.shape)
temp = np.moveaxis(temp, 0,1)
print(temp.shape)


# In[115]:


n_comp = 2
TuckerAE_plane1_cluster1 = []
TuckerAE_plane1_cluster2 = []

temp = np.moveaxis(TuckerAE_plane1, 1,2)
print(temp.shape)
temp = np.moveaxis(temp, 0,1)
print(temp.shape)

for i in range(n_comp):
    clus = []
    for j in range(temp.shape[0]):
        if (labels[j] == i):
            clus.append(temp[j])
    
    clus = np.array(clus)
    print(clus.shape)
    if (i == 0):
        TuckerAE_plane1_cluster1.append(clus)
    elif (i == 1):
        TuckerAE_plane1_cluster2.append(clus)

TuckerAE_plane1_cluster1 = np.array(TuckerAE_plane1_cluster1[0])
TuckerAE_plane1_cluster2 = np.array(TuckerAE_plane1_cluster2[0])


# In[116]:


TuckerAE_plane1_cluster1.shape, TuckerAE_plane1_cluster2.shape


# In[117]:


TuckerAE_plane1_cluster1 = np.moveaxis(TuckerAE_plane1_cluster1, 0,1)
TuckerAE_plane1_cluster1 = np.moveaxis(TuckerAE_plane1_cluster1, 1,2)
print(TuckerAE_plane1_cluster1.shape)
TuckerAE_plane1_cluster2 = np.moveaxis(TuckerAE_plane1_cluster2, 0,1)
TuckerAE_plane1_cluster2 = np.moveaxis(TuckerAE_plane1_cluster2, 1,2)
print(TuckerAE_plane1_cluster2.shape)


# In[161]:


print(np.min(TuckerAE_cluster1_all), np.max(TuckerAE_cluster1_all))
print(np.min(TuckerAE_cluster2_all), np.max(TuckerAE_cluster2_all))
print(np.min(TuckerAE_plane1_cluster1), np.max(TuckerAE_plane1_cluster1))
print(np.min(TuckerAE_plane1_cluster2), np.max(TuckerAE_plane1_cluster2))


# In[118]:


f0f_TuckerAE_withoutcluster_cluster1 = []
f0f_rmse_TuckerAE_withoutcluster_cluster1 = []
f0f_TuckerAE_withoutcluster_cluster2 = []
f0f_rmse_TuckerAE_withoutcluster_cluster2 = []

for i in range(len(latent_dims)):
    f0f_TuckerAE_withoutcluster_cluster1.append(f0f_err_max_plane(cluster_pixel[0], TuckerAE_plane1_cluster1[i,0]))
    f0f_rmse_TuckerAE_withoutcluster_cluster1.append(rmse_error(cluster_pixel[0], TuckerAE_plane1_cluster1[i,0]))
    f0f_TuckerAE_withoutcluster_cluster2.append(f0f_err_max_plane(cluster_pixel[1], TuckerAE_plane1_cluster2[i,0]))
    f0f_rmse_TuckerAE_withoutcluster_cluster2.append(rmse_error(cluster_pixel[1], TuckerAE_plane1_cluster2[i,0]))

f0f_TuckerAE_withoutcluster_cluster1 = np.array(f0f_TuckerAE_withoutcluster_cluster1)
f0f_rmse_TuckerAE_withoutcluster_cluster1 = np.array(f0f_rmse_TuckerAE_withoutcluster_cluster1)
f0f_TuckerAE_withoutcluster_cluster2 = np.array(f0f_TuckerAE_withoutcluster_cluster2)
f0f_rmse_TuckerAE_withoutcluster_cluster2 = np.array(f0f_rmse_TuckerAE_withoutcluster_cluster2)


# In[119]:


f0f_TuckerAE_withoutcluster_cluster1, f0f_TuckerAE_withoutcluster_cluster2


# In[120]:


f0f_rmse_TuckerAE_withoutcluster_cluster1, f0f_rmse_TuckerAE_withoutcluster_cluster2


# In[121]:


TuckerAE_cluster_combine.shape


# In[226]:


# Denormalizing

TuckerAE_cluster_combine_denorm = np.zeros((4,1,16395,39,39))

for i in range(4):
    for p in range(1):
        f0_f_raw = np.copy(TuckerAE_cluster_combine[i][p])
        #f0_f_raw = f0_f_raw*sig[p,:,np.newaxis, np.newaxis] + mu[p,:,np.newaxis, np.newaxis]
        f0_f_raw = f0_f_raw*1.0e+15
        TuckerAE_cluster_combine_denorm[i][p] = f0_f_raw


# In[121]:


np.save('./results/npy_files/TuckerAE_cluster_combine_denorm.npy', TuckerAE_cluster_combine_denorm)


# In[123]:


draw(data_load[0][13050])


# In[176]:


draw(TuckerAE_cluster_combine_denorm[3][0][13050])


# In[227]:


f0f_TuckerAE_combine_denorm = []
f0f_rmse_TuckerAE_combine_denorm = []

for i in range(len(latent_dims)):
    f0f_TuckerAE_combine_denorm.append(f0f_err_max_plane(data_load[0], TuckerAE_cluster_combine_denorm[i,0]))
    f0f_rmse_TuckerAE_combine_denorm.append(rmse_error(data_load[0], TuckerAE_cluster_combine_denorm[i,0]))

f0f_TuckerAE_combine_denorm = np.array(f0f_TuckerAE_combine_denorm)
f0f_rmse_TuckerAE_combine_denorm = np.array(f0f_rmse_TuckerAE_combine_denorm)


# In[228]:


f0f_TuckerAE_combine_denorm, f0f_rmse_TuckerAE_combine_denorm


# In[191]:


f0f_TuckerAE_plane1_denorm = np.load('./physics_err/f0f_TuckerAE_plane1.npy')
f0f_rmse_TuckerAE_plane1_denorm = np.load('./physics_err/f0f_rmse_TuckerAE_plane1.npy')


# In[192]:


f0f_TuckerAE_plane1_denorm, f0f_rmse_TuckerAE_plane1_denorm


# In[232]:


plt.plot(compression_ratio,f0f_TuckerAE_withoutcluster_cluster1, '-ro', label='Tucker AE Default (Cluster 1)')
plt.plot(compression_ratio,f0f_TuckerAE_cluster1, '-bo', label='Tucker AE + Clustering (Cluster 1)')

plt.legend(loc="lower right", bbox_to_anchor=(1.7, 0.3))
plt.xlabel('Compression ratio')
plt.ylabel('Relative abs error')
plt.title('f0f error - normalized space')


# In[233]:


plt.plot(compression_ratio,f0f_TuckerAE_withoutcluster_cluster2, '-ro', label='Tucker AE Default (Cluster 2)')
plt.plot(compression_ratio,f0f_TuckerAE_cluster2, '-bo', label='Tucker AE + Clustering (Cluster 2)')

plt.legend(loc="lower right", bbox_to_anchor=(1.7, 0.3))
plt.xlabel('Compression ratio')
plt.ylabel('Relative abs error')
plt.title('f0f error - normalized space')


# In[234]:


plt.plot(compression_ratio,f0f_rmse_TuckerAE_withoutcluster_cluster1, '-ro', label='Tucker AE Default (Cluster 1)')
plt.plot(compression_ratio,f0f_rmse_TuckerAE_cluster1, '-bo', label='Tucker AE + Clustering (Cluster 1)')

plt.legend(loc="lower right", bbox_to_anchor=(1.7, 0.3))
plt.xlabel('Compression ratio')
plt.ylabel('RMSE error')
plt.title('f0f error - normalized space')


# In[235]:


plt.plot(compression_ratio,f0f_rmse_TuckerAE_withoutcluster_cluster2, '-ro', label='Tucker AE Default (Cluster 2)')
plt.plot(compression_ratio,f0f_rmse_TuckerAE_cluster2, '-bo', label='Tucker AE + Clustering (Cluster 2)')

plt.legend(loc="lower right", bbox_to_anchor=(1.7, 0.3))
plt.xlabel('Compression ratio')
plt.ylabel('RMSE error')
plt.title('f0f error - normalized space')


# In[238]:


plt.plot(compression_ratio,f0f_TuckerAE_plane1, '-ro', label='Tucker AE Default (Combined)')
plt.plot(compression_ratio,f0f_TuckerAE_combine, '-bo', label='Tucker AE + Clustering (Combined)')

plt.legend(loc="lower right", bbox_to_anchor=(1.7, 0.3))
plt.xlabel('Compression ratio')
plt.ylabel('Relative abs error')
plt.title('f0f error - normalized space')


# In[239]:


plt.plot(compression_ratio,f0f_rmse_TuckerAE_plane1, '-ro', label='Tucker AE Default (Combined)')
plt.plot(compression_ratio,f0f_rmse_TuckerAE_combine, '-bo', label='Tucker AE + Clustering (Combined)')

plt.legend(loc="lower right", bbox_to_anchor=(1.7, 0.3))
plt.xlabel('Compression ratio')
plt.ylabel('RMSE error')
plt.title('f0f error - normalized space')


# In[197]:


f0f_mgard_plane1 = np.load('./physics_err/f0f_mgard_plane1.npy')
f0f_rmse_mgard_plane1 = np.load('./physics_err/f0f_rmse_mgard_plane1.npy')


# In[196]:


compression_ratio_mgard = [38.84513554609529, 95.44419194608854, 147.9154344666052, 434.2270010309102]


# In[242]:


plt.plot(compression_ratio,f0f_TuckerAE_plane1_denorm, '-ro', label='Tucker AE Default (Combined)')
plt.plot(compression_ratio,f0f_TuckerAE_combine_denorm, '-bo', label='Tucker AE + Clustering (Combined)')

plt.legend(loc="lower right", bbox_to_anchor=(1.7, 0.3))
plt.xlabel('Compression ratio')
plt.ylabel('Relative abs error')
plt.title('f0f error - original space')


# In[243]:


plt.plot(compression_ratio,f0f_rmse_TuckerAE_plane1_denorm, '-ro', label='Tucker AE Default (Combined)')
plt.plot(compression_ratio,f0f_rmse_TuckerAE_combine_denorm, '-bo', label='Tucker AE + Clustering (Combined)')

plt.legend(loc="lower right", bbox_to_anchor=(1.7, 0.3))
plt.xlabel('Compression ratio')
plt.ylabel('RMSE error')
plt.title('f0f error - original space')


# In[195]:


TuckerAE_cluster_combine_denorm.shape


# In[196]:


Linf_rmse_TuckerAE_combine_denorm = []
Linf_relative_TuckerAE_combine_denorm = []

for i in range(4):
    temp1 = []
    temp2 = []
    
    for j in range(1):
        temp1.append(physics_loss_Linf_rmse(data_load[j], TuckerAE_cluster_combine_denorm[i][j]))        
        temp2.append(physics_loss_Linf_mine(data_load[j], TuckerAE_cluster_combine_denorm[i][j]))
        
    Linf_rmse_TuckerAE_combine_denorm.append(temp1)
    Linf_relative_TuckerAE_combine_denorm.append(temp2)

Linf_rmse_TuckerAE_combine_denorm = np.array(Linf_rmse_TuckerAE_combine_denorm)
Linf_relative_TuckerAE_combine_denorm = np.array(Linf_relative_TuckerAE_combine_denorm)


# In[197]:


Linf_rmse_TuckerAE_combine_denorm = Linf_rmse_TuckerAE_combine_denorm[:,0,:]
Linf_relative_TuckerAE_combine_denorm = Linf_relative_TuckerAE_combine_denorm[:,0,:]


# In[198]:


Linf_rmse_TuckerAE_plane1 = np.load('./physics_err/Linf_rmse_TuckerAE_plane1.npy')
Linf_relative_TuckerAE_plane1 = np.load('./physics_err/Linf_relative_TuckerAE_plane1.npy')
Linf_rmse_TuckerAE_plane1 = Linf_rmse_TuckerAE_plane1[:,0,:]
Linf_relative_TuckerAE_plane1 = Linf_relative_TuckerAE_plane1[:,0,:]


# In[199]:


plt.plot(compression_ratio,Linf_relative_TuckerAE_plane1[:,0], '-ro', label='Tucker AE Default')
plt.plot(compression_ratio,Linf_relative_TuckerAE_combine_denorm[:,0], '-bo', label='Tucker AE + Clustering')

plt.legend(loc="lower right", bbox_to_anchor=(1.5, 0.3))
plt.xlabel('Compression ratio')
plt.ylabel('Node-wise L inf')
plt.title('density')


# In[200]:


plt.plot(compression_ratio,Linf_rmse_TuckerAE_plane1[:,0], '-ro', label='Tucker AE Default')
plt.plot(compression_ratio,Linf_rmse_TuckerAE_combine_denorm[:,0], '-bo', label='Tucker AE + Clustering')

plt.legend(loc="lower right", bbox_to_anchor=(1.5, 0.3))
plt.xlabel('Compression ratio')
plt.ylabel('Node-wise RMSE')
plt.title('density')


# In[201]:


plt.plot(compression_ratio,Linf_relative_TuckerAE_plane1[:,1], '-ro', label='Tucker AE Default')
plt.plot(compression_ratio,Linf_relative_TuckerAE_combine_denorm[:,1], '-bo', label='Tucker AE + Clustering')

plt.legend(loc="lower right", bbox_to_anchor=(1.5, 0.3))
plt.xlabel('Compression ratio')
plt.ylabel('Node-wise L inf')
plt.title('U_para')


# In[202]:


plt.plot(compression_ratio,Linf_rmse_TuckerAE_plane1[:,1], '-ro', label='Tucker AE Default')
plt.plot(compression_ratio,Linf_rmse_TuckerAE_combine_denorm[:,1], '-bo', label='Tucker AE + Clustering')

plt.legend(loc="lower right", bbox_to_anchor=(1.5, 0.3))
plt.xlabel('Compression ratio')
plt.ylabel('Node-wise RMSE')
plt.title('U_para')


# In[207]:


plt.plot(compression_ratio,Linf_relative_TuckerAE_plane1[:,2], '-ro', label='Tucker AE Default')
plt.plot(compression_ratio,Linf_relative_TuckerAE_combine_denorm[:,2], '-bo', label='Tucker AE + Clustering')

plt.legend(loc="lower right", bbox_to_anchor=(1.5, 0.3))
plt.xlabel('Compression ratio')
plt.ylabel('Node-wise L inf')
plt.title('T_perp')


# In[208]:


plt.plot(compression_ratio,Linf_rmse_TuckerAE_plane1[:,2], '-ro', label='Tucker AE Default')
plt.plot(compression_ratio,Linf_rmse_TuckerAE_combine_denorm[:,2], '-bo', label='Tucker AE + Clustering')

plt.legend(loc="lower right", bbox_to_anchor=(1.5, 0.3))
plt.xlabel('Compression ratio')
plt.ylabel('Node-wise RMSE')
plt.title('T_perp')


# In[209]:


plt.plot(compression_ratio,Linf_relative_TuckerAE_plane1[:,3], '-ro', label='Tucker AE Default')
plt.plot(compression_ratio,Linf_relative_TuckerAE_combine_denorm[:,3], '-bo', label='Tucker AE + Clustering')

plt.legend(loc="lower right", bbox_to_anchor=(1.5, 0.3))
plt.xlabel('Compression ratio')
plt.ylabel('Node-wise L inf')
plt.title('T_para')


# In[210]:


plt.plot(compression_ratio,Linf_rmse_TuckerAE_plane1[:,3], '-ro', label='Tucker AE Default')
plt.plot(compression_ratio,Linf_rmse_TuckerAE_combine_denorm[:,3], '-bo', label='Tucker AE + Clustering')

plt.legend(loc="lower right", bbox_to_anchor=(1.5, 0.3))
plt.xlabel('Compression ratio')
plt.ylabel('Node-wise RMSE')
plt.title('T_para')


# In[ ]:




