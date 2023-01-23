#!/usr/bin/env python
# coding: utf-8




def getOffsets(A, option): #normalizing an array
    import numpy as np
    _,m=A.shape
    
    #lower and upper bounds joint angles
    lb_theta = np.array([0,-30,-30,-180,-90,-180])*np.pi/180;   
    ub_theta = np.array([90,60,30,180,90,180])*np.pi/180;     
    df_theta = ub_theta-lb_theta
    
    #lower and upper bounds force
    lb_force = np.array([-2000,-2000,-2000,0,0,0]); ub_force = np.array([2000,2000,2000,0,0,0]); df_force=ub_force-lb_force
    
    #lower and upper bounds of absolute pose xyz coordinates
    lb_pose = np.array([-1,-1,-1]); ub_pose = np.array([3,3,3]); df_pose=ub_pose-lb_pose          
    
    #lower and upper bounds of quaternions
    lb_quat=np.array([-1,-1,-1,-1]); ub_quat=np.array([1,1,1,1]); df_quat=ub_quat-lb_quat
    
    # relative pose and angular error shall not be normalized
    offset_z = np.array([0,0,0]); length_z=np.array([1,1,1])
    
    if (option=="minmax"):
        offset=[np.min(A[:,i:i+1]) for i in range(m)]
        length=[np.max(A[:,i:i+1])-np.min(A[:,i:i+1]) for i in range(m)]
    
    elif(option=="standard"):
        offset=[np.mean(A[:,i:i+1]) for i in range(m)]
        length=[np.std(A[:,i:i+1]) for i in range(m)]        
    
    elif(option=="dataset1"): #dataset 1 
        # input (6): six joint angles (rad); 
        # output (7): absolute pose xyz, i.e. position (m) and quaternions
        offset=np.concatenate((lb_theta, lb_pose,lb_quat))
        length=np.concatenate((df_theta, df_pose, df_quat))
    
    elif(option=="dataset2"): #dataset 2 
        # input (12): six joint angles (rad), wrench vector (force(N), torque(Nm)=zeros)
        # output (7): absolute pose, i.e. position (m) and quaternions
        offset = np.concatenate((lb_theta, lb_force, lb_pose, lb_quat))
        length = np.concatenate((df_theta, df_force, df_pose, df_quat))        

    elif(option=="dataset3"): #dataset 3 
        # input (12): six joint angles (rad), wrench vector (force(N), torque(Nm)=zeros)
        # output (7): relative pose, i.e. position error (m) and quaternion error
        offset = np.concatenate((lb_theta, lb_force, offset_z, lb_quat))
        length = np.concatenate((df_theta, df_force, length_z, df_quat))                
        
    elif(option=="dataset4"): #dataset 4 
        # input (6): six joint angles (rad)
        # intput (7): absolute pose, i.e. position (m) and quaternions
        offset = np.concatenate((lb_theta, lb_pose, lb_quat))
        length = np.concatenate((df_theta, df_pose, df_quat))            
        
    elif(option=="dataset5"): #dataset 5 
        # input (6): six joint angles (rad)
        # output (6): relative pose, i.e. position error (m) and angular error (rad)
        offset = np.concatenate((lb_theta, offset_z, offset_z))
        length = np.concatenate((df_theta, length_z, length_z))

    elif(option=="dataset6"): #dataset 6 
        # input (9): six joint angles (rad), force vector (N)
        # output (6): absolute pose, i.e. position (m) and angular error (rad)
        offset=np.concatenate((lb_theta, lb_force[:3],lb_pose,offset_z))
        length=np.concatenate((df_theta, df_force[:3],df_pose,length_z))        
    
    elif(option=="none"): #no normalization
        offset=np.zeros(m)
        length=np.zeros(m)
        
    else:
        print("Error: unknown option for normalization")                
        raise Exception("")
        
    return(offset,length)


def normalize(A,offset,length):
    _,m=A.shape
    for i in range(m):
        if(length[i]!=0):
            A[:,i:i+1] = (A[:,i:i+1]-offset[i])/length[i]        

        
def denormalize(A,offset,length): #normalizing an array
    _,m=A.shape
    for i in range(m):
        if(length[i]!=0):
            A[:,i:i+1] = A[:,i:i+1]*length[i] + offset[i]
    



