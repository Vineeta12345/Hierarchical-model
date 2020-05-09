# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:45:00 2020

@author: Vineeta
"""

import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage 
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from pylab import rcParams
import seaborn as sb    
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm
np.set_printoptions(precision=4, suppress = True)
plt.figure(figsize=(10,3))
%matplotlib inline
plt.style.use('seaborn-whitegrid')
hier = pd.read_csv('Hierachical.csv')
x = hier.iloc[:,[0,2]].values
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)
plt.scatter(x[y_hc == 0,0], x[y_hc == 0,1], s=100, c='red', label = 'cluster 1')
plt.scatter(x[y_hc == 1,0], x[y_hc == 1,1], s=100, c='orange', label = 'cluster 2')
plt.scatter(x[y_hc == 2,0], x[y_hc == 2,1], s=100, c='pink', label = 'cluster 3')
plt.title('Hierarchical Clustering')
plt.xlabel('Budget and Income')
plt.ylabel('Genre')
plt.legend()
plt.show()