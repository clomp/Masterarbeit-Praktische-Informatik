# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:15:04 2022

@author: lomp
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi as PI
from recursiveGP import recursiveGP
import itertools
import gpflow
from gpflow.utilities import print_summary

from sklearn.model_selection import train_test_split

# example sin
# latente function    
#f = lambda x: np.sin(2*x)    
#model = recursiveGP(variance=1.0, lengthscales=0.5, sigma=0.1)
# N=30; lower=0; upper=6
#xp = np.linspace(lower,upper, 200).reshape(-1,1);
#plt.plot(xp, f(xp), color="grey");
# model.initialise_line(lower, upper, N)
# print(model.loss(model.X,f(model.X)))
# model.training_function(20,f,lower,upper)
# model.training_function(30,f,lower,upper)
# model.training_function(30,f,lower,upper, plotting=True,color='m--')
# print(model.loss(model.X,f(model.X)))

# dataset 4 - x-value
dataframe = np.loadtxt("datasets/dataset4.csv",delimiter=",")

X_values    = dataframe[:,:6]
Y_values = { 
    'x': dataframe[:, 6:7],
    'y': dataframe[:, 7:8],
    'z': dataframe[:, 8:9]
    }
    
N   = X_values.shape[0]



# XTrain, XTest, YTrain_x, YTest_x = train_test_split(X_values,Y_values['x'], test_size=0.01, random_state=42)


# modelGPx = gpflow.models.GPR(data=(XTrain, YTrain_x), 
#                        kernel=gpflow.kernels.SquaredExponential(variance=1.0,lengthscales=1.0), 
#                        mean_function=None)
# modelGPx.likelihood.variance.assign(1E-05)

# opt = gpflow.optimizers.Scipy()
# opt_logs = opt.minimize(modelGPx.training_loss, 
#                          modelGPx.trainable_variables, 
#                          options=dict(maxiter=50))
# print_summary(modelGPx)

EQUIDIST = True
chunksize = 10
iterations=50
num_base_vectors = 50
vx = 1.5;   lsx = 2.5;  snx=0.0001
vy = 2.97;  lsy = 2.84; sny=0.0001
vz = 1.0;  lsz = 3; snz=0.0001

models ={
    'x': recursiveGP(variance=vx, lengthscales=lsx, sigma=snx), 
    'y': recursiveGP(variance=vy, lengthscales=lsy, sigma=sny), 
    'z': recursiveGP(variance=vz, lengthscales=lsz, sigma=snz),
}

if(EQUIDIST):
    print("Choosing base vectors equally distributed.")
    # input space of dataset 4 consists of 
    # [0,pi/2] x [-pi/6,pi/3] x [-pi/6, pi/6] x [-pi,pi] x [-pi/2, pi/2] x [-pi, pi]
    # interval lengths: pi/2, pi/2, pi/3, 2pi, pi, 2pi
    lb_theta = np.array([0,-30,-30,-180,-90,-180])*np.pi/180;   
    ub_theta = np.array([90,60,30,180,90,180])*np.pi/180;   
    list_base_coordinates=[np.linspace(lb_theta[i], ub_theta[i], 3) for i in range(6)]
    list_base_vectors=[]
    for b in itertools.product(*list_base_coordinates):
        list_base_vectors.append(list(b))
    X_base = np.array(list_base_vectors)
    num_base_vectors =  len(list_base_vectors) # 3^6 
    C = models['x'].covfunc(X_base,X_base)
    print(np.linalg.det(C))
else:
    print("Choosing random base vectors from the data set.")
    I           = np.random.randint(0, N-1, size=num_base_vectors)
    X_base      = X_values[I,:6]    
    C           = models['x'].covfunc(X_base,X_base)  
    while( np.linalg.det(C) == 0.0):
        print("another try to initialise...")
        I           = np.random.randint(0, N-1, size=num_base_vectors)
        X_base      = X_values[I,:6]    
        C           = models['x'].covfunc(X_base,X_base)  
    Y_base ={
        'x':  dataframe[I, 6:7],
        'y':  dataframe[I, 7:8],
        'z':  dataframe[I, 8:9]     
    }
    

for m in models:
    models[m].initialise(X_base)

print("Models initialized with "+ str(num_base_vectors) + " base vectors.")

if(EQUIDIST==False):        
    for m in models:
        models[m].recursiveGP(X_base, Y_base[m])


def train(models, X,Ys, offset, chunksize):    
    for m in models:
        models[m].recursiveGP(X[offset:offset+chunksize],Ys[m][offset:offset+chunksize])       

def plot_histogram(ax, YY, YYp, nbins=50):
    ax.hist(abs(YY-YYp),bins=nbins)
    return(ax)

def plot_prediction(ax,XX,YY, YYp, C, labelstr):
    N=len(XX)
    ax.plot(range(N), YY , "b--")
    ax.plot(range(N), YYp, "r--")   
    ax.set_xlabel(labelstr) 
    D=np.diag(C)
    if(np.min(D)<0):
        print(np.min(D))
    C=np.sqrt(D)
    C.reshape(-1,1)
    CI = 1.96 * C/np.sqrt(N)  
    CI = CI.reshape(-1,1)
    Ylower = (YYp-CI).reshape((N,))
    Yupper = (YYp+CI).reshape((N,))
    ax.fill_between(range(N), Ylower, Yupper , color='orange', alpha=.25)   
    return(ax)
        

def plot_predict_diagram(model,XX,YY,ax1, ax2,label):
    YYp, C = model.predict(XX)    
    YYp = YYp.numpy()        
    ax1 = plot_histogram(ax1,YY,YYp)
    ax2 = plot_prediction(ax2,XX, YY, YYp, C, labelstr=label)
    


def next_chunk(offset):
    train(models, X_values, Y_values, offset, chunksize)    
    offset+=chunksize
    figure, ax = plt.subplots(3,2, figsize = (10,6))
    figure.suptitle('after '+str(offset)+ ' recursiveGPs')
    #plot_predict_diagram(models['x'],X_values[:offset],Y_values['x'][:offset],ax[0], ax[1],"x-pos")
    
    plot_predict_diagram(models['x'],X_values[:offset],Y_values['x'][:offset],ax[0,0], ax[0,1],"x-pos")
    plot_predict_diagram(models['y'],X_values[:offset],Y_values['y'][:offset],ax[1,0], ax[1,1],"y-pos")
    plot_predict_diagram(models['z'],X_values[:offset],Y_values['z'][:offset],ax[2,0], ax[2,1],"z-pos")
        
    # plot_predict_diagram(models['x'],X_values[:offset],Y_values['x'][:offset],ax[0,0], ax[1,0],"x-pos")
    # plot_predict_diagram(models['y'],X_values[:offset],Y_values['y'][:offset],ax[0,1], ax[1,1],"y-pos")
    # plot_predict_diagram(models['z'],X_values[:offset],Y_values['z'][:offset],ax[0,2], ax[1,2],"z-pos")
    plt.show()
    return(offset)

offset = 0
for i in range(iterations):
    offset=next_chunk(offset)
