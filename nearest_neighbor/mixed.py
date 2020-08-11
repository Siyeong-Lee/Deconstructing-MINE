#Copyright Weihao Gao, UIUC

import scipy.spatial as ss
from scipy.special import digamma
from math import log
import numpy.random as nr
import numpy as np

#Main Function
def Mixed_KSG(x,y,k=5):
    '''
    Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
    Using *Mixed-KSG* mutual information estimator

        Input: x: 2D array of size N*d_x (or 1D list of size N if d_x = 1)
        y: 2D array of size N*d_y (or 1D list of size N if d_y = 1)
        k: k-nearest neighbor parameter

        Output: one number of I(X;Y)
        '''

    assert len(x)==len(y), "Lists should have same length"
    assert k <= len(x)-1, "Set k smaller than num. samples - 1"
    N = len(x)
    if x.ndim == 1:
        x = x.reshape((N,1))
    dx = len(x[0])
    if y.ndim == 1:
        y = y.reshape((N,1))
    dy = len(y[0])
    data = np.concatenate((x,y),axis=1)

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_dis = [tree_xy.query(point,k+1,p=float('inf'))[0][k] for point in data]
    ans = 0

    for i in range(N):
        kp, nx, ny = k, k, k
        if knn_dis[i] == 0:
            kp = len(tree_xy.query_ball_point(data[i],1e-15,p=float('inf')))
            nx = len(tree_x.query_ball_point(x[i],1e-15,p=float('inf')))
            ny = len(tree_y.query_ball_point(y[i],1e-15,p=float('inf')))
        else:
            nx = len(tree_x.query_ball_point(x[i],knn_dis[i]-1e-15,p=float('inf')))
            ny = len(tree_y.query_ball_point(y[i],knn_dis[i]-1e-15,p=float('inf')))
        ans += (digamma(kp) + log(N) - digamma(nx) - digamma(ny))/N
    return ans

'''
Below are other estimators used in the paper for comparison
'''

#Partitioning Algorithm (Red Line)
def Partitioning(x,y,numb=8):
    assert len(x)==len(y), "Lists should have same length"
    N = len(x)
    if x.ndim == 1:
        x = x.reshape((N,1))
    dx = len(x[0])
    if y.ndim == 1:
        y = y.reshape((N,1))
    dy = len(y[0])

    minx = np.zeros(dx)
    miny = np.zeros(dy)
    maxx = np.zeros(dx)
    maxy = np.zeros(dy)
    for d in range(dx):
        minx[d], maxx[d] = x[:,d].min()-1e-15, x[:,d].max()+1e-15
        for d in range(dy):
            miny[d], maxy[d] = y[:,d].min()-1e-15, y[:,d].max()+1e-15

    freq = np.zeros((numb**dx+1,numb**dy+1))
    for i in range(N):
        index_x = 0
        for d in range(dx):
            index_x *= dx
            index_x += int((x[i][d]-minx[d])*numb/(maxx[d]-minx[d]))
        index_y = 0
        for d in range(dy):
            index_y *= dy
            index_y += int((y[i][d]-miny[d])*numb/(maxy[d]-miny[d]))
        freq[index_x][index_y] += 1.0/N
    freqx = [sum(t) for t in freq]
    freqy = [sum(t) for t in freq.transpose()]

    ans = 0
    for i in range(numb**dx):
        for j in range(numb**dy):
            if freq[i][j] > 0:
                ans += freq[i][j]*log(freq[i][j]/(freqx[i]*freqy[j]))
    return ans

#Noisy KSG Algorithm (Green Line)
def Noisy_KSG(x,y,k=5,noise=0.01):
    assert len(x)==len(y), "Lists should have same length"
    assert k <= len(x)-1, "Set k smaller than num. samples - 1"
    N = len(x)
    if x.ndim == 1:
        x = x.reshape((N,1))
    dx = len(x[0])
    if y.ndim == 1:
        y = y.reshape((N,1))
    dy = len(y[0])
    data = np.concatenate((x,y),axis=1)

    if noise > 0:
        data += nr.normal(0,noise,(N,dx+dy))

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_dis = [tree_xy.query(point,k+1,p=float('inf'))[0][k] for point in data]
    ans = 0

    for i in range(N):
        nx = len(tree_x.query_ball_point(x[i],knn_dis[i]-1e-15,p=float('inf')))
        ny = len(tree_y.query_ball_point(y[i],knn_dis[i]-1e-15,p=float('inf')))
        ans += (digamma(k) + log(N) - digamma(nx) - digamma(ny))/N
    return ans

#Original KSG estimator (Blue line)
def KSG(x,y,k=5):
    assert len(x)==len(y), "Lists should have same length"
    assert k <= len(x)-1, "Set k smaller than num. samples - 1"
    N = len(x)
    if x.ndim == 1:
        x = x.reshape((N,1))
    dx = len(x[0])
    if y.ndim == 1:
        y = y.reshape((N,1))
    dy = len(y[0])
    data = np.concatenate((x,y),axis=1)

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_dis = [tree_xy.query(point,k+1,p=float('inf'))[0][k] for point in data]
    ans = 0

    for i in range(N):
        nx = len(tree_x.query_ball_point(x[i],knn_dis[i]+1e-15,p=float('inf')))-1
        ny = len(tree_y.query_ball_point(y[i],knn_dis[i]+1e-15,p=float('inf')))-1
        ans += (digamma(k) + log(N) - digamma(nx) - digamma(ny))/N
    return ans



import numpy as np
import tqdm

DIM = 20
BATCHSIZE = 64
ITERATIONS = 20000

SAMPLESIZE = BATCHSIZE * ITERATIONS // 5

def mi_schedule(n_iter):
    mis = np.round(np.linspace(0.5, 5.5-1e-9, n_iter)) *2.0#0.1
    return mis.astype(np.float32)

def mi_to_rho(dim, mi):
    return np.sqrt(1-np.exp(-2.0 / dim * mi))



class GaussianDataset:
    def __init__(self, rho, dim):
        self.generator = lambda: np.random.multivariate_normal(
            mean=[0 for _ in range(dim)], cov=[[1 if i == j else 0 for i in range(dim)] for j in range(dim)]
        )
        self.rho = rho
        self.dim = dim

    def sample(self):
        x = self.generator()
        y = self.rho * x + np.sqrt(1-self.rho**2) * self.generator()
        return x, y

for target_mi in (2, 4, 6, 8, 10):
    gd = GaussianDataset(rho=mi_to_rho(DIM, target_mi), dim=DIM)
    xs, ys = [], []
    for _ in tqdm.tqdm(range(10000)):
        x, y = gd.sample()
        xs.append(x)
        ys.append(y)
    xs = np.stack(xs)
    ys = np.stack(ys)

    for ksg_class in (KSG, Mixed_KSG, Noisy_KSG):
        print(target_mi, ksg_class, ksg_class(xs, ys))




