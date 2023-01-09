import itertools
import numpy as np
from robotModel import robot
from math import pi as PI


mu=0
sigma=0.001

robo = robot()

def create_trainings_test_data(param="standard",interval_length=3):
    if(param=="standard"):
        Input_data = np.array([[-14, 10, 25, 0, -25, 0], \
                               [0  , 10, 25, 0, -25, 0], \
                               [13 , 10, 25, 0, -25, 0], \
                               [27 , 10, 25, 0, -25, 0], \
                               [40 , 10, 25, 0, -25, 0], \
                               [53 , 10, 25, 0, -25, 0], \
                               [61 , 10, 25, 0, -25, 0], \
                               [-20, 10, 25, 0, -25, -70], \
                               [-6 , 10, 25, 0, -25, -70], \
                               [8  , 10, 25, 0, -25, -70], \
                               [21 , 10, 25, 0, -25, -70], \
                               [34 , 10, 25, 0, -25, -70], \
                               [47 , 10, 25, 0, -25, -70], \
                               [61 , 10, 25, 0, -25, -70]]) *PI/180
    else:
        step=PI/(interval_length-1)
        I=np.arange(-PI/2,PI/2+step,step)
        Input_data = np.array([x for x in itertools.product(I,I,I,I,I,I)])

    TotalNum=len(Input_data)
    #Create matrices Tws using forward Kinematic
    Trainings_data = [np.matrix(robo.forwardKinematics(theta)[0]) for theta in Input_data]
    #Make some noise with mean mu and standard deviation sigma
    noise = np.random.normal(mu, sigma, size=( TotalNum, 12))
    #Transform the noise in 4x4 matrices to be added to the Tws-matrices
    noise_mat=[]
    for x in noise:
        y=np.matrix(x.reshape((4,3)))
        y.resize((4,4))
        noise_mat.append(y)
    #Create Test data.    
    Test_data=[Trainings_data[i]+noise_mat[i] for i in range(0,TotalNum)]
    return(Input_data, Trainings_data, Test_data)

In, Out, Noise = create_trainings_test_data()


