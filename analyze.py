import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#edf = pd.read_csv('eigenvalues_knn1_sigma0.3_plane0.csv')
#x = [i for i in range(0,edf.shape[1])]
#plt.plot(x, edf.columns.tolist())
#plt.show()

sigma = [3.0, 4.0, 5.0, 10]
wdf = pd.read_csv('weights-sim.csv')
print (wdf.describe())
w = wdf['distance']
for s in sigma:
    sim = np.exp(-w*w/(2*s*s))
    #wdf['similarity-{}'.format(s)] = sim
    plt.hist(sim)
    plt.xlabel('Similarity, Sigma: {}'.format(s))
    plt.ylabel('Pairs')
    plt.show()

#print (wdf.describe())
#wdf.describe().to_csv('sim.csv')

