# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:47:36 2022

@author: blumberg
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.optimize import lsq_linear

class robot:
    def __init__(self,Tool=np.array([0,0,0]),measSystem='ART'):
        #Intialize the robot with the DH-Parameters
        #Input: Tool, in [X,Y,Z]
        self.d = np.array([0,0,0,1.285,0,0.3])
        self.alpha = np.array([-np.pi/2,0,-np.pi/2,np.pi/2,-np.pi/2,0])
        self.a = np.array([0.41,1.12,0.25,0,0,0])
        self.dTheta = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        #Parameters of Parallegram mechanismmm
        self.a2s = 0.75
        self.a3s = 1.12
        self.a4s = 0.75
        self.beta = 0   #Hayati Parameter
        self.AW0 = np.identity(4)   #initialize world to base frame trafo
        self.A6S = np.identity(4)   #initialize hand to sensor trafo
        self.A6EE = self.transMatrix('x',Tool[0]) @ self.transMatrix('y', Tool[1]) @ self.transMatrix('z', Tool[2]) @ self.rotMatrix('y', np.pi/2, 4) @ self.rotMatrix('z', np.pi, 4)
        self.measSystem = measSystem     #Choose between 'ART' and 'Lasertracker', used for orientation representation
        self.parallelogram = True

    def rotMatrix(self,coordinate, angle, size):
        #Rotation matrix either in 3x3 or 4x4
        R = np.identity(size)
        if coordinate == 'x':
            R[1,1] = np.cos(angle)
            R[1,2] = -np.sin(angle)
            R[2,1] = np.sin(angle)
            R[2,2] = np.cos(angle)
        elif coordinate == 'y':
            R[0,0] = np.cos(angle)
            R[0,2] = np.sin(angle)
            R[2,0] = -np.sin(angle)
            R[2,2] = np.cos(angle)
        elif coordinate == 'z':
            R[0,0] = np.cos(angle)
            R[0,1] = -np.sin(angle)
            R[1,0] = np.sin(angle)
            R[1,1] = np.cos(angle)
        return R
            
    def transMatrix(self,coordinate, displacement):
        #Homogeneous translational matrix 4x4
        T = np.identity(4)
        if coordinate == 'x':
            T[0,3] = displacement
        elif coordinate == 'y':
            T[1,3] = displacement
        elif coordinate == 'z':
            T[2,3] = displacement
        return T
    
    def A_Matrix_DH(self,theta,d,a,alpha):
        #homogeneous transformation matrix according to DH-convention
        A = self.rotMatrix('z',theta,4) @ self.transMatrix('z',d) @ self.transMatrix('x',a) @ self.rotMatrix('x',alpha,4)
        return A
    
    def vex(self,S):
        #Corresponds to the vex() function in Matlab Robotics Toolbox
        #Input: S - 3x3 rotation matrix
        v = 0.5*np.array(([S[2,1]-S[1,2]], [S[0,2]-S[2,0]] , [S[1,0]-S[0,1]]))
        return v
    
    def geometricJacobian(self,T1,T2):
        #Geometric Jacobian between two homogeneous transformation matrices
        #Input: T1 - the local homogeneous transformation matrix, which is investigated
        #       T2 - the frame, to which the Jacobian matrix should point
        J_x = np.hstack((T1[0:3,0],np.zeros((3,))))
        J_y = np.hstack((T1[0:3,1],np.zeros((3,))))
        J_z = np.hstack((T1[0:3,2],np.zeros((3,))))
        J_w = np.hstack((np.cross(T1[0:3,0], T2[0:3,3]-T1[0:3,3]), T1[0:3,0]))
        J_p = np.hstack((np.cross(T1[0:3,1], T2[0:3,3]-T1[0:3,3]), T1[0:3,1]))
        J_r = np.hstack((np.cross(T1[0:3,2], T2[0:3,3]-T1[0:3,3]), T1[0:3,2]))
        return np.transpose(np.vstack((J_x,J_y,J_z,J_w,J_p,J_r)))
    
    def jointJacobian(self,Tbase,T,Tend):
        #Jacobian matrix with respect to the joint angles
        #Input: Tbase - homogeneous transformation matrix describing the origin frame
        #       T - 3D array with all frames of interest
        #       Tend - the frame to which the jacobian should point
        J_q1 = np.hstack((np.cross(Tbase[0:3,2], Tend[0:3,3]-Tbase[0:3,3]) , Tbase[0:3,2]))
        J_q2 = np.hstack((np.cross(T[0][0:3,2], Tend[0:3,3]-T[0][0:3,3]) , T[0][0:3,2]))
        J_q3 = np.hstack((np.cross(T[1][0:3,2], Tend[0:3,3]-T[1][0:3,3]) , T[1][0:3,2]))
        J_q4 = np.hstack((np.cross(T[2][0:3,2], Tend[0:3,3]-T[2][0:3,3]) , T[2][0:3,2]))
        J_q5 = np.hstack((np.cross(T[3][0:3,2], Tend[0:3,3]-T[3][0:3,3]) , T[3][0:3,2]))
        J_q6 = np.hstack((np.cross(T[4][0:3,2], Tend[0:3,3]-T[4][0:3,3]) , T[4][0:3,2]))
        return np.transpose(np.vstack((J_q1,J_q2,J_q3,J_q4,J_q5,J_q6)))
    
    def passiveJoint(self,theta,option=None):
        #Calculates the joint angles for a passive joint with parallelogram mechanism
        if self.a[1] == self.a3s and self.a2s == self.a4s and option != 'calib':
            self.sita3 = theta[2] - theta[1]
            self.sita3pie = theta[1] - theta[2] + np.pi/2
            self.sita4pie = np.pi/2 - theta[1] + theta[2]
        else:
            sita2 = np.pi/2*3 + theta[1]
            sita2pie = theta[2] + np.pi
            xtemp = self.a2s*np.cos(sita2pie) - self.a[1]*np.cos(sita2)
            ytemp = self.a2s*np.sin(sita2pie) - self.a[1]*np.sin(sita2)
            phi2 = np.arctan2(ytemp,xtemp) + np.arccos((self.a4s**2-xtemp**2-ytemp**2-self.a3s**2)/(2*self.a3s*np.sqrt(xtemp**2+ytemp**2)))
            phi1 = np.arctan2((ytemp+self.a3s*np.sin(phi2))/self.a4s , (xtemp+self.a3s*np.cos(phi2))/self.a4s)
            self.sita3pie = phi2 - sita2pie
            self.sita4pie = np.pi + phi1 - phi2
            self.sita3 = theta[2] + self.sita3pie + self.sita4pie - theta[1] - np.pi
        return self.sita3, self.sita3pie, self.sita4pie 
     
    def forwardKinematics(self,theta,returnVal='HiB',calibration=None,plotOption=None,saveConfig=False):
        #forward kinematics
        #Input: theta - joint angle vector [J1,J2,J3,J4,J5,J6] in rad
        #       returnVal - string, see below
        theta = theta + self.dTheta     #if offset is calibrated
        if self.parallelogram == True:
            sita3,_,_ = self.passiveJoint(theta,option=calibration)
        else:
            sita3 = theta[2]
            
        A1 = self.A_Matrix_DH(theta[0],self.d[0],self.a[0],self.alpha[0])
        A2 = self.A_Matrix_DH(theta[1]-np.pi/2,self.d[1],self.a[1],self.alpha[1])
        A2 = A2 @ self.rotMatrix('y',self.beta,4)   #Hayati Parameter
        A3 = self.A_Matrix_DH(sita3,self.d[2],self.a[2],self.alpha[2])
        A4 = self.A_Matrix_DH(theta[3],self.d[3],self.a[3],self.alpha[3])
        A5 = self.A_Matrix_DH(theta[4],self.d[4],self.a[4],self.alpha[4])
        A6 = self.A_Matrix_DH(theta[5]+np.pi,self.d[5],self.a[5],self.alpha[5])
        #
        T1 = A1
        T2 = T1 @ A2
        T3 = T2 @ A3
        T4 = T3 @ A4
        T5 = T4 @ A5
        T6 = T5 @ A6
        TS = T6 @ self.A6S
        TEE = T6 @ self.A6EE
        
        if saveConfig:
            self.T1 = T1
            self.T2 = T2
            self.T3 = T3
            self.T4 = T4
            self.T5 = T5
            self.T6 = T6
            self.TS = TS
            self.TEE = TEE
        #
        if plotOption:
            xJoints = np.concatenate((np.array([0]),T1[0:1,3],T2[0:1,3],T3[0:1,3],T4[0:1,3],T5[0:1,3],T6[0:1,3],TEE[0:1,3]))
            yJoints = np.concatenate((np.array([0]),T1[1:2,3],T2[1:2,3],T3[1:2,3],T4[1:2,3],T5[1:2,3],T6[1:2,3],TEE[1:2,3]))
            zJoints = np.concatenate((np.array([-0.94]),T1[2:3,3],T2[2:3,3],T3[2:3,3],T4[2:3,3],T5[2:3,3],T6[2:3,3],TEE[2:3,3]))
            plt.figure()
            plt.ion()
            ax = plt.axes(projection='3d')
            ax.plot(xJoints,yJoints,zJoints,'ko-')
            ax.set_xlabel('x-axis')
            ax.set_ylabel('y-axis')
            ax.set_zlabel('z-axis')
            ax.set_xlim((-1,3))
            ax.set_ylim((-2,2))
            ax.set_zlim((-1,3))
            plt.show()
            
                #
        if returnVal == 'SiB':      #Sensor in Base
            J = self.jointJacobian(np.identity(4), [T1,T2,T3,T4,T5], TS)
            return TS, J
        elif returnVal == 'HiB':    #Hand in Base
            J = self.jointJacobian(np.identity(4), [T1,T2,T3,T4,T5], T6)    
            return T6, J 
        elif returnVal == 'EEiB':   #End-effector in Base
            J = self.jointJacobian(np.identity(4), [T1,T2,T3,T4,T5,T6], TEE)
            return TEE, J
        elif returnVal == 'SiW':    #Sensor in World
            J = self.jointJacobian(self.AW0, [self.AW0@T1,self.AW0@T2,self.AW0@T3,self.AW0@T4,self.AW0@T5,self.AW0@T6], self.AW0@TS)
            return (self.AW0 @ TS) , J
        elif returnVal == 'HiW':    #Hand in World
            J = self.jointJacobian(self.AW0, [self.AW0@T1,self.AW0@T2,self.AW0@T3,self.AW0@T4,self.AW0@T5,self.AW0@T6], self.AW0@T6)
            return (self.AW0 @ T6) , J
        elif returnVal == 'EEiW':   #End-effector in World
            J = self.jointJacobian(self.AW0, [self.AW0@T1,self.AW0@T2,self.AW0@T3,self.AW0@T4,self.AW0@T5,self.AW0@T6], self.AW0@TEE)
            return (self.AW0 @ TEE) , J
        elif returnVal == 'B_all':  #all transformations in the world frame
            return (T1, T2, T3, T4, T5, T6, TEE, TS)
        elif returnVal == 'W_all':  #all transformations in the world frame
            return (self.AW0@T1, self.AW0@T2, self.AW0@T3, self.AW0@T4, self.AW0@T5, self.AW0@T6, self.AW0@TEE, self.AW0@TS)
        
    def inverseKinematics(self,pose, config=np.array([1,1,0]), ang=None):
        #inverse kinematic algorithm
        #Input: pose - homogeneous transformationmatrix
        #       config - the configuration [1,1,1] corresponds to F,U,T
        #       ang - 'deg' to get the joint angles in degree
        d = self.d
        a = np.concatenate((np.array([0]),self.a[0:5]))
        theta0 = np.array([0,-np.pi/2,0,0,0,np.pi])
        joint_limit = np.array([[- 180,- 64,- 30,- 360,- 122,- 360],[180,90,130,360,122,360]]) * np.pi / 180
        joint_limit_Ori = joint_limit + theta0
        #Calculate the objective pose
        Tobj = pose
        #n = Tobj[0:3,0]
        o = Tobj[0:3,1]
        aa = Tobj[0:3,2]
        p = Tobj[0:3,3]
        #Calculate Joint Angle 1
        pWristCenter = Tobj[0:4,3] - d[5] * Tobj[0:4,2]
        dphi1 = np.arctan2(d[3],a[3])
        Theta1_1 = np.arctan2(p[1] - aa[1] * d[5] , p[0] - aa[0] * d[5])
        Theta1_2 = np.arctan2(-(p[1] - aa[1] * d[5]) , -(p[0] - aa[0] * d[5]))
        # arm configuration : front and back config determination
        # judgement by the angle between the x axis of link frame{1} and
        # the vector form original of {1} to pWristCenter
        Link1_X_1 = np.transpose(np.array([np.cos(Theta1_1),np.sin(Theta1_1),0]))
        Link1_X_2 = np.transpose(np.array([np.cos(Theta1_2),np.sin(Theta1_2),0]))
        PwristCentetInLinkframe1 = pWristCenter[0:3] - np.transpose(np.array([0,0,d[0]]))
        ArmConf_angle1 = np.arccos(np.transpose(Link1_X_1) @ PwristCentetInLinkframe1 / np.linalg.norm(PwristCentetInLinkframe1))
        ArmConf_angle2 = np.arccos(np.transpose(Link1_X_2) @ PwristCentetInLinkframe1 / np.linalg.norm(PwristCentetInLinkframe1))
        if ArmConf_angle1 > ArmConf_angle2:
            q1_front = Theta1_2
            q1_back = Theta1_1
        else:
            q1_front = Theta1_1
            q1_back = Theta1_2
        
        if config[0] == 1:
            q1 = q1_front
        else:
            q1 = q1_back
        #Calculate Joint Angle 2    
        flag2 = 0
        A2 = pWristCenter[0] * np.cos(q1) + pWristCenter[1] * np.sin(q1)
        B2 = pWristCenter[2] - d[0]
        C2 = (a[3]**2 + d[3]**2 - B2**2 - (A2 - a[1])**2 - a[2]**2) / ((2*B2*a[2])**2 + (2*(A2 - a[1]) * a[2])**2)**0.5
        if np.abs(C2) < 1:
            q2_1 = np.arctan2(C2 , -(1-C2**2)**0.5) + np.arctan2(2*(A2 - a[1])*a[2] , 2*B2*a[2])
            q2_2 = np.arctan2(C2,(1 - C2 ** 2) ** 0.5) + np.arctan2(2 * (A2 - a[1]) * a[2] , 2*B2* a[2])
        else:
            flag2 = 1
            result = np.zeros((1,6)) - 9
            return result    
        #elbow congfiguration judgement
        px = pWristCenter[0]
        py = pWristCenter[1]
        pz = pWristCenter[2]
        PWristCenter_Y_InLink2Frame_1 = d[0] * np.cos(q2_1) - pz * np.cos(q2_1) + a[1] * np.sin(q2_1) - px * np.cos(q1) * np.sin(q2_1) - py * np.sin(q1) * np.sin(q2_1)
        if PWristCenter_Y_InLink2Frame_1 >= 0:
            q2_front = q2_1
            q2_back = q2_2
        else:
            q2_front = q2_2
            q2_back = q2_1
       
        if config[1] == 1:
            q2 = q2_front
        else:
            q2 = q2_back
        #Calculate Joint Angle 3
        A3 = - (B2 + a[2] * np.sin(q2)) / (a[3]**2 + d[3]**2)**0.5
        B3 = (A2 - a[1] - a[2] * np.cos(q2)) / (a[3]**2 + d[3]**2)**0.5
        q3 = np.arctan2(A3,B3) - q2 - dphi1
        #Calculate Joint Angle 4
        B4 = - (aa[0] * np.cos(q1) * np.cos(q2 + q3) + aa[1] * np.sin(q1) * np.cos(q2 + q3) - aa[2] * np.sin(q2 + q3))
        A4 = aa[1] * np.cos(q1) - aa[0] * np.sin(q1)
        if np.abs(A4) < 10000.0 * np.finfo(float).eps and np.abs(B4) < 10000.0 * np.finfo(float).eps:
            q4_1 = 0
            q4_2 = np.pi
        else:
            q4_1 = np.arctan2(A4,B4)
            if q4_1 >= 0:
                q4_2 = np.arctan2(A4,B4) - np.pi
            else:
                q4_2 = np.arctan2(A4,B4) + np.pi
        #Calculate Joint Angle 5
        T46_13 = aa[0] * (np.sin(q1) * np.sin(q4_1) - np.cos(q4_1) * (np.cos(q1) * np.sin(q2) * np.sin(q3) - np.cos(q1) * np.cos(q2) * np.cos(q3))) - aa[1] * (np.cos(q1) * np.sin(q4_1) + np.cos(q4_1) * (np.sin(q1) * np.sin(q2) * np.sin(q3) - np.cos(q2) * np.cos(q3) * np.sin(q1))) - aa[2] * np.cos(q4_1) * np.sin(q3 + q2)
        T46_33 = - aa[0] * np.cos(q1) * np.sin(q3 + q2) - aa[1] * np.sin(q1) * np.sin(q3 + q2) - aa[2] * np.cos(q3 + q2)
        q5_1 = np.arctan2(- T46_13 , T46_33)
        T46_13 = aa[0] * (np.sin(q1) * np.sin(q4_2) - np.cos(q4_2) * (np.cos(q1) * np.sin(q2) * np.sin(q3) - np.cos(q1) * np.cos(q2) * np.cos(q3))) - aa[1] * (np.cos(q1) * np.sin(q4_2) + np.cos(q4_2) * (np.sin(q1) * np.sin(q2) * np.sin(q3) - np.cos(q2) * np.cos(q3) * np.sin(q1))) - aa[2] * np.cos(q4_2) * np.sin(q3 + q2)
        T46_33 = - aa[0] * np.cos(q1) * np.sin(q3 + q2) - aa[1] * np.sin(q1) * np.sin(q3 + q2) - aa[2] * np.cos(q3 + q2)
        q5_2 = np.arctan2(- T46_13 , T46_33)
        if q5_1 >= 0:
            q5_up = q5_1
            q5_down = q5_2
            q4_up = q4_1
            q4_down = q4_2
        else:
            q5_up = q5_2
            q5_down = q5_1
            q4_up = q4_2
            q4_down = q4_1
        if config[2] == 1:
            q5 = q5_up
            q4 = q4_up
        else:
            q5 = q5_down
            q4 = q4_down
        #Calculate Joint Angle 6
        T56_12 = o[0] * (np.cos(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q4) * (np.cos(q1) * np.sin(q2) * np.sin(q3) - np.cos(q1) * np.cos(q2) * np.cos(q3))) - np.sin(q5) * (np.cos(q1) * np.cos(q2) * np.sin(q3) + np.cos(q1) * np.cos(q3) * np.sin(q2))) - o[1] * (np.cos(q5) * (np.cos(q1) * np.sin(q4) + np.cos(q4) * (np.sin(q1) * np.sin(q2) * np.sin(q3) - np.cos(q2) * np.cos(q3) * np.sin(q1))) + np.sin(q5) * (np.cos(q2) * np.sin(q1) * np.sin(q3) + np.cos(q3) * np.sin(q1) * np.sin(q2))) - o[2] * (np.sin(q5) * (np.cos(q2) * np.cos(q3) - np.sin(q2) * np.sin(q3)) + np.cos(q4) * np.cos(q5) * (np.cos(q2) * np.sin(q3) + np.cos(q3) * np.sin(q2)))
        T56_32 = o[1] * (np.cos(q1) * np.cos(q4) - np.sin(q4) * (np.sin(q1) * np.sin(q2) * np.sin(q3) - np.cos(q2) * np.cos(q3) * np.sin(q1))) - o[0] * (np.cos(q4) * np.sin(q1) + np.sin(q4) * (np.cos(q1) * np.sin(q2) * np.sin(q3) - np.cos(q1) * np.cos(q2) * np.cos(q3))) - o[2] * np.sin(q4) * np.sin(q3 + q2)
        q6 = np.arctan2(- T56_12 , - T56_32)
        # display / assembly
        F = np.array([[q1],[q2],[q3],[q4],[q5],[q6]])
        # select the feasible solution sets
        j_l = joint_limit_Ori
        TrueFalse = 1
        for j in range(6):
            if j == 3 or j == 5:
                TrueFalse = TrueFalse * ((F[j] > j_l[0,j]) * (F[j] < j_l[1,j]) or (F[j] + 2* np.pi > j_l[0,j]) * (F[j] + 2* np.pi < j_l[1,j]) or (F[j] - 2 * np.pi > j_l[0,j]) * (F[j] - 2 * np.pi < j_l[1,j]))
                if TrueFalse == 1 and not (F[j] > j_l[0,j])  * (F[j] < j_l[1,j]):
                    if (F[j] + 2 * np.pi > j_l[0,j]) * (F[j] + 2 * np.pi < j_l[1,j]):
                        F[j] = F[j] + 2 * np.pi
                    else:
                        F[j] = F[j] - 2 * np.pi
            else:
                TrueFalse = TrueFalse * ((F[j] > j_l[0,j]) * (F[j] < j_l[1,j]))
        #
        if TrueFalse and flag2 == 0:
            if self.parallelogram == True:
                F = np.transpose(np.transpose(F) - theta0)
                F[2] = F[2] + F[1]
                # including the joint constraint limit by paralleogram mechanism
                if F[2] - F[1] < - 70 * np.pi / 180 and F[2] - F[1] > 70 * np.pi / 180:
                    result = np.zeros((1,6)) - 9
                else:
                    result = F
                    if np.abs(F[5]) > np.abs(F[5] - 2 * np.pi):
                        F[5] = F[5] - 2 * np.pi
                    else:
                        if np.abs(F[5]) > np.abs(F[5] + 2 * np.pi):
                            F[5] = F[5] + 2 * np.pi
                    result[5] = F[5]
            else:
                F = np.transpose(np.transpose(F) - theta0)
                result = F
                if np.abs(F[5]) > np.abs(F[5] - 2 * np.pi):
                    F[5] = F[5] - 2 * np.pi
                else:
                    if np.abs(F[5]) > np.abs(F[5] + 2 * np.pi):
                        F[5] = F[5] + 2 * np.pi
                result[5] = F[5]
                if np.abs(F[3]) > np.abs(F[3] - 2 * np.pi):
                    F[3] = F[3] - 2 * np.pi
                else:
                    if np.abs(F[3]) > np.abs(F[3] + 2 * np.pi):
                        F[3] = F[3] + 2 * np.pi
                result[3] = F[3]
        else:
            result = np.zeros((1,6)) - 9
        if ang == 'deg':
            return (result * 180/np.pi)
        else:
            return result
    
    def inverseKinematicsPath(self,data,zeroPt=np.array([0,0,0,0,0,0]),config=np.array([1,1,0]), ang=None):
        #calculates inverse kinematics for a path
        #input: data - data array with N data points, defined as TCP in zeroPt
        #       zeroPt - the zero point to which the data corresponds
        #       config - the configuration of the robot needed for inverse kinematic algorithm
        numData = len(data)
        jointAngles = np.zeros((numData,6))
        zero_R = self.rotMatrix('x', zeroPt[3], 3) @ self.rotMatrix('y', zeroPt[4], 3) @ self.rotMatrix('z', zeroPt[5], 3)
        for i in range(numData):
            T0EE = np.vstack((np.hstack((zero_R , np.array([data[i,0:3].T + zeroPt[0:3]]).T)) , np.array([0,0,0,1]))) @ self.rotMatrix('z', data[i,5], 4) @ self.rotMatrix('y', data[i,4], 4) @ self.rotMatrix('x', data[i,3], 4)
            T06 = T0EE @ np.linalg.inv(self.A6EE)
            jointAngles[i:i+1,:] = self.inverseKinematics(T06, config=config).T
        return jointAngles
    
    def measData2rotM(self,measData):
        e = np.empty((3,3))
        #transforms data from ART or Lasertracker to 3x3 rotation matrix
        if self.measSystem =='ART':
            e[0,0],e[0,1],e[0,2] = measData[0],measData[3],measData[6]
            e[1,0],e[1,1],e[1,2] = measData[1],measData[4],measData[7]
            e[2,0],e[2,1],e[2,2] = measData[2],measData[5],measData[8]
            #rotM = np.array([measData[0],measData[3],measData[6]],[measData[1],measData[4],measData[7]],[measData[2],measData[5],measData[8]])
            rotM = e
        elif self.measSystem == 'Lasertracker':
            r = Rotation.from_quat(np.array([measData[1],measData[2],measData[3],measData[0]])/np.linalg.norm(measData))
            rotM = r.as_matrix()
        return rotM
    
    def measData2Trans(self,measData):
        T = np.empty((4,4))
        if self.measSystem == 'ART':
            T[0,0], T[1,0], T[2,0], T[3,0] = measData[3], measData[4], measData[5], 0
            T[0,1], T[1,1], T[2,1], T[3,1] = measData[6], measData[7], measData[8], 0
            T[0,2], T[1,2], T[2,2], T[3,2] = measData[9], measData[10], measData[11], 0
            T[0,3], T[1,3], T[2,3], T[3,3] = measData[0], measData[1], measData[2], 1
        elif self.measSystem == 'Lasertracker':
            rotM = self.measData2rotM(measData[3:])
            pos = np.array([measData[0:3]]).T
            T = np.vstack((np.hstack((rotM , pos)) , np.array([0,0,0,1])))
        return T        
    
    def rodriguezRotation(self,Pts,n0,n1):
        n0 = n0 / np.linalg.norm(n0)
        n1 = n1 / np.linalg.norm(n1)
        theta = np.arccos(np.dot(n0,n1))
        rotNormal = np.cross(n0,n1)
        rotNormal = rotNormal / np.linalg.norm(rotNormal)
        #Compute rotated Points
        rotPts = np.zeros((Pts.shape[0],3))
        if not np.sum(np.isnan(rotNormal)):
            for i in range(Pts.shape[0]):
                rotPts[i] = Pts[i]*np.cos(theta) + np.cross(rotNormal,Pts[i])*np.sin(theta) + rotNormal*np.dot(rotNormal,Pts[i])*(1-np.cos(theta))
        else:
            rotPts = Pts
        return rotPts
    
    def circFit(self,x,y):
        atemp = np.transpose(np.vstack((x,y,np.ones(x.shape))))
        a = np.linalg.pinv(atemp) @ (-(x**2+y**2))
        # a = np.linalg.lstsq(atemp,-(x**2+y**2))
        # a = a[0]
        xc = -0.5 * a[0]
        yc = -0.5 * a[1]
        R = np.sqrt((a[0]**2+a[1]**2)/4 - a[2])
        return xc, yc, R
    
    def circFit3D(self,circLocs):
        meanLoc = np.mean(circLocs,axis=0)
        numCurPts = circLocs.shape[0]
        movedToOrigin = circLocs - np.ones((numCurPts,1))*meanLoc
        U,s,V = np.linalg.svd(movedToOrigin)
        V = np.transpose(V)
        circleNormal = V[:,2]
        circleLocsXY = self.rodriguezRotation(movedToOrigin,circleNormal,np.array([0,0,1]))
        xc,yc,radius = self.circFit(circleLocsXY[:,0], circleLocsXY[:,1])
        centerLoc = self.rodriguezRotation(np.array([[xc,yc,0]]),np.array([0,0,1]),circleNormal) + meanLoc
        return centerLoc, circleNormal, radius
    
    def calibRobotFrames(self,measData,theta=None,method='iterative'):
        #calibrates the transformation between world and baseframe AW0 and between hand and sensor A6S
        #Input - measData: measured Data from 14 calibration poses
        if method == 'iterative':
            theta = np.array([[-14, 10, 25, 0, -25, 0], \
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
                             [61 , 10, 25, 0, -25, -70]]) *np.pi/180 #Joint Angles used in Calibration Movement. Change, if changed on the controller!
            num_perCircle = 7
            #Calculate initial values with circfit function
            base_origin1, base_zaxis1, radius1 = self.circFit3D(measData[0:num_perCircle,0:3])
            base_origin2, base_zaxis2, radius2 = self.circFit3D(measData[num_perCircle:,0:3])
            base_origin = base_origin1
            base_zaxis = base_zaxis1 / np.linalg.norm(base_zaxis1)
            # radius = radius1
            #
            if base_zaxis[2]<0:
                base_zaxis = -base_zaxis
            #
            base_yaxis = np.cross(base_zaxis, (measData[0,0:3]-base_origin)/np.linalg.norm(measData[0,0:3]-base_origin))
            base_xaxis = np.cross(base_yaxis, base_zaxis)
            base_R = np.concatenate((base_xaxis.T, base_yaxis.T, np.array([base_zaxis]).T),axis=1) @ self.rotMatrix('z', -theta[0,0], 3)
            self.AW0 = np.vstack([np.hstack([base_R,base_origin.T]),np.array([0,0,0,1])])
            #Start iterative method
            pNumCalibration = theta.shape[0]
            x = np.ones((8,))
            it = 0
            while np.linalg.norm(x,ord=np.inf) > 1e-15 and it < 20:
                it+=1
                posErr = np.zeros((pNumCalibration,6))
                J_Sensor = np.zeros((6,6))
                for i in range(pNumCalibration):
                    qa = theta[i,:]
                    TWS,_ = self.forwardKinematics(qa,returnVal='SiW')
                    TW6,_ = self.forwardKinematics(qa,returnVal='HiW')
                    posErr[i,0:3] = measData[i,0:3] - TWS[0:3,3]
                    posErr[i,3:6] = self.vex(self.measData2rotM(measData[i,3:]) @ np.linalg.inv(TWS[0:3,0:3])).T
                    #Calculate Jacobian Matrix of Base Frame
                    J_base = self.geometricJacobian(self.AW0, TWS)
                    #Calculate Jacobian Matrix of Sensor Frame
                    J_Sensor[:,0:3] = np.concatenate((TW6[0:3,0:3],np.zeros((3,3))),axis=0)
                    J_Sensor[:,3:4] = np.concatenate((np.zeros((3,1)),TW6[0:3,0:1]),axis=0)
                    eul_tmac = Rotation.from_matrix(self.A6S[0:3,0:3])
                    eul_tmac = eul_tmac.as_euler('XYZ')
                    Trx = TW6 @ self.rotMatrix('x', eul_tmac[0], 4)
                    Trxy = Trx @ self.rotMatrix('y', eul_tmac[1], 4)
                    J_Sensor[:,4:5] = np.concatenate((np.zeros((3,1)),Trx[0:3,1:2]))
                    J_Sensor[:,5:6] = np.concatenate((np.zeros((3,1)),Trxy[0:3,2:3]))
                    #
                    A = np.hstack((J_base[:,5:6],J_base[:,2:3],J_Sensor))
                    #Stack vector b and matrix A
                    if i == 0:
                        b = np.transpose(posErr[i,:])
                        A_concat = A
                    else:
                        b = np.append(b,np.transpose(posErr[i,:]),axis=0)
                        A_concat = np.append(A_concat,A,axis=0)                
                #Solve linear least square
                out = lsq_linear(A_concat,b)
                x = out.x
                #
                self.AW0 = self.AW0 @ self.rotMatrix('z', x[0], 4)
                self.AW0 = self.AW0 @ self.transMatrix('z', x[1]) 
                transl_vector = self.A6S[0:3,3] + x[2:5]
                rtemp = Rotation.from_matrix(self.A6S[0:3,0:3])
                rotate_vector = rtemp.as_euler('XYZ') + x[5:8]
                rotate_vector = Rotation.from_euler('XYZ',rotate_vector)
                rotate_matrix = rotate_vector.as_matrix()
                rotate_matrix = np.vstack((np.hstack((rotate_matrix,np.zeros((3,1)))), np.array([0,0,0,1])))
                self.A6S = self.transMatrix('x', transl_vector[0]) @ self.transMatrix('y', transl_vector[1]) @ self.transMatrix('z', transl_vector[2]) @ rotate_matrix
        elif method == 'analytic':
            numData = len(theta)
            I3 = np.identity(3)
            for i in range(numData):
                T06,_ = self.forwardKinematics(theta[i,:],returnVal='HiB',calibration='calib')
                if i == 0:
                    tBH = T06[0:3,3]
                    K = np.hstack((measData[i,0]*I3, measData[i,1]*I3, measData[i,2]*I3, I3, -T06[0:3,0:3]))
                else:
                    tBH = np.hstack((tBH,T06[0:3,3]))
                    K = np.vstack((K, np.hstack((measData[i,0]*I3, measData[i,1]*I3, measData[i,2]*I3, I3, -T06[0:3,0:3]))))
            x = np.linalg.pinv(K) @ tBH
            RBW = np.vstack((x[0:3],x[3:6],x[6:9])).T
            tBW = x[9:12]
            pH = x[12:15]
            U,S,V = np.linalg.svd(RBW)
            RBW = np.sign(np.linalg.det(np.diag(S))) * U @ V
            self.AW0 = np.linalg.inv(np.vstack((np.hstack((RBW,np.reshape(tBW,(3,1)))), np.array([0,0,0,1]))))
            R6S = np.array([[1,0,0],[0,0,1],[0,-1,0]])
            self.A6S = np.vstack((np.hstack((R6S,np.reshape(pH,(3,1)))), np.array([0,0,0,1])))
        return 0
    
    def transMeasData(self,measData,outOption='tool'):
        #Transforms measurement data from the world to the base frame
        #calibRobotFrames must be performed before using this function
        #Input: - measData: measured data from Lasertracker or ART, each row is one measured data point
        #       - outOption: specify if transformed data should be given for the hand or the tool
        flag = 0
        if np.shape(np.shape(measData))[0] == 1:    #needed when a single measData Vector is used with shape (X,)
            measData = np.array([measData])
            #temp = np.empty((1,12))
            #temp[0,:] = measData
            #measData = temp
            flag = 1
        P06inBase = np.zeros((measData.shape))
        P0EEinBase = np.zeros((measData.shape))
        for i in range(len(measData)):
            rotM = self.measData2rotM(measData[i,3:])
            Tmeas = np.vstack((np.hstack((rotM,measData[i:i+1,0:3].T)) , np.array([0,0,0,1])))
            T06meas = np.linalg.inv(self.AW0) @ Tmeas @ np.linalg.inv(self.A6S)
            T0EEmeas = T06meas @ self.A6EE
            if self.measSystem == 'Lasertracker':
                r06 = Rotation.from_matrix(T06meas[0:3,0:3])
                r0EE = Rotation.from_matrix(T0EEmeas[0:3,0:3])
                P06inBase[i,0:3] = T06meas[0:3,3]
                P06inBase[i,3:7] = r06.as_quat()
                # temp = r0EE.as_euler('zyx')*180/np.pi
                P0EEinBase[i,0:3] = T0EEmeas[0:3,3]
                P0EEinBase[i,3:7] = r0EE.as_quat()
            elif self.measSystem == 'ART':
                P06inBase[i,0:3] = T06meas[0:3,3]
                P06inBase[i,3:12] = np.array([T06meas[0,0],T06meas[1,0],T06meas[2,0],T06meas[0,1],T06meas[1,1],T06meas[2,1],T06meas[0,2],T06meas[1,2],T06meas[2,2]])
                P0EEinBase[i,0:3] = T0EEmeas[0:3,3]
                P0EEinBase[i,3:12] = np.array([T0EEmeas[0,0],T0EEmeas[1,0],T0EEmeas[2,0],T0EEmeas[0,1],T0EEmeas[1,1],T0EEmeas[2,1],T0EEmeas[0,2],T0EEmeas[1,2],T0EEmeas[2,2]])
        if flag == 1:
            #temp1 = np.empty((np.shape(P06inBase)))
            #temp2 = np.empty((np.shape(P0EEinBase)))
            #temp1 = P06inBase[0,:]
            #temp2 = P0EEinBase[0,:]
            #P06inBase = temp1
            #P0EEinBase = temp2
            P06inBase = P06inBase[0]
            P0EEinBase = P0EEinBase[0]

        
        if outOption == 'hand':
            return P06inBase
        elif outOption == 'tool':
            return P0EEinBase            
    
    def DHcalibration(self,theta,measData):
        #Calibrates the DH parameter based on 6D measurement data
        #Input: theta - joints angles Nx6 array
        #       measData - corresponding measurement data
        numData = len(theta)
        #Initialize transformations
        T0W = self.AW0
        bx = T0W[0,3]
        by = T0W[1,3]
        bz = T0W[2,3]
        a0 = np.arctan2(-T0W[1,2] , T0W[2,2])
        b0 = np.arctan2(T0W[0,2] , -np.sin(a0)*T0W[1,2]+np.cos(a0)*T0W[2,2])
        g0 = np.arctan2(-T0W[0,1] , T0W[0,0])
        self.AW0 = self.transMatrix('x',bx) @ self.transMatrix('y',by) @ self.rotMatrix('x',a0,4) @ self.rotMatrix('y',b0,4) @ self.transMatrix('z',bz) @ self.rotMatrix('z',g0,4)
        tx = self.A6S[0,3]
        ty = self.A6S[1,3]
        tz = self.A6S[2,3]
        rx = -np.pi/2
        ry = 0
        rz = 0
        self.A6S = self.transMatrix('x',tx) @ self.transMatrix('y',ty) @ self.transMatrix('z',tz) @ self.rotMatrix('x',rx,4) @ self.rotMatrix('y',ry,4) @ self.rotMatrix('z',rz,4)
        #Start calibration
        normDelta = 1
        tol = 1e-10
        it = 1
        J_Base = np.zeros((6,4))
        J_Sensor = np.zeros((6,6))
        while normDelta > tol and it < 20:
            P_calc = np.zeros((numData,3))
            Err_Ang = np.zeros((numData,3))
            for i in range(numData):
                T06,_ = self.forwardKinematics(theta[i,:],returnVal='HiB',calibration='calib')
                T0S = T06 @ self.A6S
                TWS = self.AW0 @ T0S
                P_calc[i,0:3] = TWS[0:3,3]
                R_C = TWS[0:3,0:3]
                R_M = self.measData2rotM(measData[i,3:])
                Err_Ang[i:i+1,:] = self.vex(R_M @ np.linalg.inv(R_C)).T
                #Calculate Jacobian
                TW1,TW2,TW3,TW4,TW5,TW6,_,TWS_temp = self.forwardKinematics(theta[i,:],returnVal='W_all',calibration='calib')
                J0 = self.geometricJacobian(self.AW0, TWS)
                J1 = self.geometricJacobian(TW1, TWS)
                J2 = self.geometricJacobian(TW2, TWS)
                J3 = self.geometricJacobian(TW3, TWS)
                J4 = self.geometricJacobian(TW4, TWS)
                J5 = self.geometricJacobian(TW5, TWS)
                J6 = self.geometricJacobian(TW6, TWS)
                if self.parallelogram == True:
                    g1 = -1 +self.a[1] * np.cos(theta[i,1]-theta[i,2]-self.sita3pie) / (self.a4s*np.sin(self.sita4pie))
                    g2 = self.a2s * np.sin(self.sita3pie) / (self.a4s*np.sin(self.sita4pie))
                    g3 = np.sin(theta[i,1]-theta[i,2]-self.sita3pie) / (self.a4s*np.sin(self.sita4pie))
                    g4 = np.cos(self.sita3pie) / (self.a4s*np.sin(self.sita4pie))
                    g5 = 1/ (self.a4s*np.sin(self.sita4pie))
                    g6 = np.cos(self.sita4pie) / (self.a4s*np.sin(self.sita4pie))
                if i == 0:
                    J_a = np.hstack((J1[:,0:1],J2[:,0:1],J3[:,0:1],J4[:,0:1],J5[:,0:1],J6[:,0:1]))
                    J_d = np.hstack((J0[:,2:3],J1[:,2:3],J2[:,2:3],J3[:,2:3],J4[:,2:3],J5[:,2:3]))
                    J_alpha = np.hstack((J1[:,3:4],J2[:,3:4],J3[:,3:4],J4[:,3:4],J5[:,3:4],J6[:,3:4]))
                    J_beta = np.hstack((J1[:,4:5],J2[:,4:5],J3[:,4:5],J4[:,4:5],J5[:,4:5],J6[:,4:5]))
                    J_theta = np.hstack((J0[:,5:6],J1[:,5:6],J2[:,5:6],J3[:,5:6],J4[:,5:6],J5[:,5:6]))
                    #Base jacobian
                    J_Base1 = np.array([[1],[0],[0],[0],[0],[0]])
                    J_Base2 = np.array([[0],[1],[0],[0],[0],[0]])
                    temp1 = np.cross(np.array([1,0,0]), TWS[0:3,3]-np.array([bx,by,0]))
                    J_Base3 = np.array([[temp1[0]],[temp1[1]],[temp1[2]],[1],[0],[0]])
                    temp1 = np.cross(np.array([0,np.cos(a0),np.sin(a0)]), TWS[0:3,3]-np.array([bx,by,0]))
                    J_Base4 = np.array([[temp1[0]],[temp1[1]],[temp1[2]],[0],[np.cos(a0)],[np.sin(a0)]])
                    J_Base = np.hstack((J_Base1,J_Base2,J_Base3,J_Base4))
                    #Sensor jacobian
                    temp1 = self.AW0 @ T06 @ self.transMatrix('x', tx) @ self.transMatrix('y', ty) @ self.transMatrix('z', tz)
                    temp2 = temp1 @ self.rotMatrix('x', rx, 4)
                    temp3 = temp2 @ self.rotMatrix('y', ry, 4)
                    temp4 = temp3 @ self.rotMatrix('z', rz, 4) #must equal TWS
                    J_Sensor1 = np.array([[temp1[0,0]],[temp1[1,0]],[temp1[2,0]],[0],[0],[0],])
                    J_Sensor2 = np.array([[temp1[0,1]],[temp1[1,1]],[temp1[2,1]],[0],[0],[0],])
                    J_Sensor3 = np.array([[temp1[0,2]],[temp1[1,2]],[temp1[2,2]],[0],[0],[0],])
                    temp5 = np.cross(temp1[0:3,0], temp4[0:3,3]-temp1[0:3,3])
                    J_Sensor4 = np.array([[temp5[0]],[temp5[1]],[temp5[2]],[temp1[0,0]],[temp1[1,0]],[temp1[2,0]]])
                    temp5 = np.cross(temp2[0:3,1], temp4[0:3,3]-temp1[0:3,3])
                    J_Sensor5 = np.array([[temp5[0]],[temp5[1]],[temp5[2]],[temp2[0,1]],[temp2[1,1]],[temp2[2,1]]])
                    temp5 = np.cross(temp3[0:3,2], temp4[0:3,3]-temp1[0:3,3])
                    J_Sensor6 = np.array([[temp5[0]],[temp5[1]],[temp5[2]],[temp3[0,2]],[temp3[1,2]],[temp3[2,2]]])
                    J_Sensor = np.hstack((J_Sensor1,J_Sensor2,J_Sensor3,J_Sensor4,J_Sensor5,J_Sensor6))
                    if self.parallelogram == True:
                        Ja2 = g3*J_theta[:,2:3] + J_a[:,1:2]
                        Jth2 = J_theta[:,1:2] + g1*J_theta[:,2:3]
                        Ja2s = g4*J_theta[:,2:3]
                        Ja3s = g5*J_theta[:,2:3]
                        Ja4s = g6*J_theta[:,2:3]
                        Jth2s = g2*J_theta[:,2:3] 
                else:
                    J_a = np.vstack((J_a, np.hstack((J1[:,0:1],J2[:,0:1],J3[:,0:1],J4[:,0:1],J5[:,0:1],J6[:,0:1]))))
                    J_d = np.vstack((J_d, np.hstack((J0[:,2:3],J1[:,2:3],J2[:,2:3],J3[:,2:3],J4[:,2:3],J5[:,2:3]))))
                    J_alpha = np.vstack((J_alpha, np.hstack((J1[:,3:4],J2[:,3:4],J3[:,3:4],J4[:,3:4],J5[:,3:4],J6[:,3:4]))))
                    J_beta = np.vstack((J_beta, np.hstack((J1[:,4:5],J2[:,4:5],J3[:,4:5],J4[:,4:5],J5[:,4:5],J6[:,4:5]))))
                    J_theta = np.vstack((J_theta, np.hstack((J0[:,5:6],J1[:,5:6],J2[:,5:6],J3[:,5:6],J4[:,5:6],J5[:,5:6]))))
                    #Base jacobian
                    J_Base1 = np.vstack((J_Base[:,0:1], np.array([[1],[0],[0],[0],[0],[0]])))
                    J_Base2 = np.vstack((J_Base[:,1:2], np.array([[0],[1],[0],[0],[0],[0]])))
                    temp1 = np.cross(np.array([1,0,0]), TWS[0:3,3]-np.array([bx,by,0]))
                    J_Base3 = np.vstack((J_Base[:,2:3], np.array([[temp1[0]],[temp1[1]],[temp1[2]],[1],[0],[0]])))
                    temp1 = np.cross(np.array([0,np.cos(a0),np.sin(a0)]), TWS[0:3,3]-np.array([bx,by,0]))
                    J_Base4 = np.vstack((J_Base[:,3:4], np.array([[temp1[0]],[temp1[1]],[temp1[2]],[0],[np.cos(a0)],[np.sin(a0)]])))
                    J_Base = np.hstack((J_Base1,J_Base2,J_Base3,J_Base4))
                    #Sensor jacobian
                    temp1 = self.AW0 @ T06 @ self.transMatrix('x', tx) @ self.transMatrix('y', ty) @ self.transMatrix('z', tz)
                    temp2 = temp1 @ self.rotMatrix('x', rx, 4)
                    temp3 = temp2 @ self.rotMatrix('y', ry, 4)
                    temp4 = temp3 @ self.rotMatrix('z', rz, 4) #must equal TWS
                    J_Sensor1 = np.vstack((J_Sensor[:,0:1], np.array([[temp1[0,0]],[temp1[1,0]],[temp1[2,0]],[0],[0],[0],])))
                    J_Sensor2 = np.vstack((J_Sensor[:,1:2], np.array([[temp1[0,1]],[temp1[1,1]],[temp1[2,1]],[0],[0],[0],])))
                    J_Sensor3 = np.vstack((J_Sensor[:,2:3], np.array([[temp1[0,2]],[temp1[1,2]],[temp1[2,2]],[0],[0],[0],])))
                    temp5 = np.cross(temp1[0:3,0], temp4[0:3,3]-temp1[0:3,3])
                    J_Sensor4 = np.vstack(( J_Sensor[:,3:4], np.array([[temp5[0]],[temp5[1]],[temp5[2]],[temp1[0,0]],[temp1[1,0]],[temp1[2,0]]])))
                    temp5 = np.cross(temp2[0:3,1], temp4[0:3,3]-temp1[0:3,3])
                    J_Sensor5 = np.vstack((J_Sensor[:,4:5], np.array([[temp5[0]],[temp5[1]],[temp5[2]],[temp2[0,1]],[temp2[1,1]],[temp2[2,1]]])))
                    temp5 = np.cross(temp3[0:3,2], temp4[0:3,3]-temp1[0:3,3])
                    J_Sensor6 = np.vstack((J_Sensor[:,5:6], np.array([[temp5[0]],[temp5[1]],[temp5[2]],[temp3[0,2]],[temp3[1,2]],[temp3[2,2]]])))
                    J_Sensor = np.hstack((J_Sensor1,J_Sensor2,J_Sensor3,J_Sensor4,J_Sensor5,J_Sensor6))
                    if self.parallelogram == True:
                        Ja2 = np.vstack((Ja2, g3*J_theta[-6:,2:3] + J_a[-6:,1:2]))
                        Jth2 = np.vstack((Jth2, J_theta[-6:,1:2] + g1*J_theta[-6:,2:3]))
                        Ja2s = np.vstack((Ja2s, g4*J_theta[-6:,2:3]))
                        Ja3s = np.vstack((Ja3s, g5*J_theta[-6:,2:3]))
                        Ja4s = np.vstack((Ja4s, g6*J_theta[-6:,2:3]))
                        Jth2s = np.vstack((Jth2s, g2*J_theta[-6:,2:3])) 
            Err_Pos = measData[:,0:3] - P_calc
            Err = np.hstack((Err_Pos,Err_Ang))
            Err = np.reshape(Err,(Err.size,1))
            #total jacobian
            #with joint angles
            J = np.hstack((J_theta[:,0:1],Jth2,Jth2s,J_theta[:,3:5],J_alpha[:,0:5],J_a[:,0:5],J_d[:,0:1],J_d[:,2:5],J_beta[:,1:2],Ja2s,Ja3s,J_Base,J_Sensor))
            #without joint angles
            # J = np.hstack((J_alpha[:,0:5],J_a[:,0:5],J_d[:,0:1],J_d[:,2:5],J_beta[:,1:2],J_Base,J_Sensor))
            dP = np.linalg.pinv(J) @ Err
            normDelta = np.linalg.norm(dP)
            #print
            print('{0}: norm(Err) = {1} , norm(deltaParms) = {2}'.format(it, np.linalg.norm(Err), normDelta))
            #update parameters
            self.dTheta[0:5] = self.dTheta[0:5] + dP[0:5].T
            self.alpha[0:5] = self.alpha[0:5] + dP[5:10].T
            self.a[0:5] = self.a[0:5] + dP[10:15].T
            self.d[0] = self.d[0] + dP[15]
            self.d[2:5] = self.d[2:5] + dP[16:19].T
            self.beta = self.beta + float(dP[19])
            self.a2s = self.a2s + float(dP[20])
            self.a3s = self.a3s + float(dP[21])
            bx = bx + float(dP[22])
            by = by + float(dP[23])
            a0 = a0 + float(dP[24])
            b0 = b0 + float(dP[25])
            self.AW0 = self.transMatrix('x',bx) @ self.transMatrix('y',by) @ self.rotMatrix('x',a0,4) @ self.rotMatrix('y',b0,4) @ self.transMatrix('z',bz) @ self.rotMatrix('z',g0,4)
            tx = tx + float(dP[26])
            ty = ty + float(dP[27])
            tz = tz + float(dP[28])
            rx = rx + float(dP[29])
            ry = ry + float(dP[30])
            rz = rz + float(dP[31])
            self.A6S = self.transMatrix('x',tx) @ self.transMatrix('y',ty) @ self.transMatrix('z',tz) @ self.rotMatrix('x',rx,4) @ self.rotMatrix('y',ry,4) @ self.rotMatrix('z',rz,4)
            it +=1
        return 0
    
    def calcError(self,Tmeas,Tcalc, in_mm=True, in_deg=True):
        err = np.empty((6))
        err[0:3] = Tcalc[0:3,3] - Tmeas[0:3,3]
        temp = self.vex(Tcalc[0:3,0:3] @ np.linalg.inv(Tmeas[0:3,0:3]))
        err[3:6] = temp[:,0]
        if in_mm:
            err[0:3] *= 1e3
        if in_deg:
            err[3:6] *= 180 / np.pi
        return err
    
    def offlineCompensation(self,theta,dataInBase):
        #not validated yet!!!
        #not validated yet!!!
        #not validated yet!!!
        #not validated yet!!!
        #not validated yet!!!
        #not validated yet!!!
        #not validated yet!!!
        #not validated yet!!!
        #not validated yet!!!
        num = len(theta)
        thetaComp = np.zeros(theta.shape)
        poseComp = np.zeros((len(theta),6))
        for i in range(num):
            q0 = theta[i,:]
            it = 1
            dq = 1
            while np.linalg.norm(dq) > 1e-12 and it < 30:
                T0EE,J = self.forwardKinematics(q0,returnVal='EEiB') #Calculate forward kinematics and joint jacobian
                if self.parallelogram == True:
                    J[:,1] = J[:,1] - J[:,2]
                errPos = T0EE[0:3,3] - dataInBase[i,0:3]    #calculate error in position between measured data and nominal forward kinematic
                rotM = self.measData2rotM(dataInBase[i,3:])
                errAng = self.vex(T0EE[0:3,0:3] @ rotM)     #calculate error in orientation between measured data and nominal forward kinematic
                err = np.hstack((errPos,errAng))
                
                dq = np.linalg.inv(J) @ err     #calculate difference in joint angles
                q0 += dq    #update joint angles
                it += 1
            thetaComp[i,:] = q0
            poseComp[i,0:3] = T0EE[0:3,3]
            temp = Rotation.from_matrix(T0EE[0:3,0:3])
            poseComp[i,3:] = temp.as_euler('zyx')
        return thetaComp, poseComp