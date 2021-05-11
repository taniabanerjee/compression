import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wdf = pd.read_csv('weights-sim.csv')
print ('Total weight:', wdf['distance'].sum())
filenames = ['labels_knn100_eps5_plane0_clus2.txt', 'labels_knn100_eps5_plane0_clus3.txt', 'labels_knn100_eps5_plane0_clus4.txt']
for filename in filenames:
    file1 = open(filename, 'r')
    Lines = file1.readlines()
    lines = [l.strip() for l in Lines]
    col_in = wdf['in'].astype(int)
    col_out = wdf['out'].astype(int)
    labels_in = [lines[i] for i in col_in]
    labels_out = [lines[i] for i in col_out]
    wdf['labels_in'] = labels_in
    wdf['labels_out'] = labels_out
    cdf = wdf[wdf['labels_in'] == wdf['labels_out']]
    weight = cdf['distance'].sum()
    print (filename, weight)
