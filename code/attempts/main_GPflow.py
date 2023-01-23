# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 16:26:34 2022

@author: Blumberg
"""
import gpflow
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints, JulierSigmaPoints, SimplexSigmaPoints


class GPmodel():
    def __init__(self,X,Y, test_amount=0.2, 
                 kernel=gpflow.kernels.SquaredExponential(),
                 mean=gpflow.functions.Constant(0)):
        self.X = X
        self.Y = Y
        #Split Data into Training and Testing
        #Train-Test Split with ratio test_size, 
        #random_state should be set so that the shuffling is reproducible, e.g. =42
        # X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, test_size=0.2, random_state=42)
        self.X_Train, self.X_Test, self.Y_Train, self.Y_Test = train_test_split(X,Y, test_size=test_amount, random_state=42)
    
        self.inputSize = X.shape[1]
        self.outputSize = Y.shape[1]
        self.kernel = kernel
        self.mean = mean
                
        self.Model = []
        
            
    def intializeGPR(self, X=None, Y=None):
        if all(v is None for v in [X, Y]):
            X = self.X_Train
            Y = self.Y_Train
        #Save Basis Vectors
        if not hasattr(self, 'X_base'):
            self.X_base = X
            
        # Model = gpflow.models.GPR((X,Y), kernel=self.kernel, mean_function=self.mean)
        Model = gpflow.models.GPR((X,Y), kernel=self.kernel)
        # opt = gpflow.optimizers.Scipy()
        # opt.minimize(Model.training_loss, Model.trainable_variables)
        self.Model = Model
        self.kxx = Model.kernel.K(X,X).numpy()
        # self.kxx = np.round(self.kxx, decimals=6)
        self.kxx_inv = np.linalg.pinv(self.kxx)
        # self.kxx_inv = np.round(self.kxx_inv, decimals=6)
        self.mx = Model.mean_function(X).numpy()
        return self
    
    
    def intialize_recursiveGPR_HPO(self, Xbase, params=np.array([0.5, 1.0, 0.01])):
        self.params = params
        self.X_base = Xbase
        
        self.Nbase = len(Xbase)
        self.Nparams = len(params)
        
        self.kernel = gpflow.kernels.SquaredExponential(lengthscales=params[0], variance=params[1])
        self.k_prior = self.kernel(Xbase).numpy()
        self.kxx_inv = np.linalg.pinv(self.k_prior)
        
        self.mean = gpflow.functions.Constant(0)
        self.mu_prior = self.mean(X_base).numpy()
        
        self.k_ge = np.zeros((len(X_base),len(params))) #zero initializing according to Huber
        
        self.Points = MerweScaledSigmaPoints(len(params), alpha=1e-3, beta=2.0, kappa=0.0)
        # self.Points = JulierSigmaPoints(len(params),kappa=-2.0)  #not working! produces negative lenghtscales! why?
        # self.covHPO = 0.2*np.eye(len(params)) #To Do!!!! initialize like stated by Huber!!!!
        
        var_lengthscale = 0.2
        var_variance = 0.3
        var_noise = 0.01
        
        self.covHPO = np.array([[var_lengthscale, 0.0, var_noise*0.5], 
                                [0.0, var_variance, var_noise*0.5],
                                [var_noise*0.5, var_noise*0.5, var_noise]])
        return self
       
    def get_Mean_and_Variance(self, Xt, covIn=None, meanIn=None):
        #Check if Model if already intialized, if not initialize it!
        if not self.Model:
            self.intializeGPR()
        mt = self.Model.mean_function(Xt).numpy()
        kxt = self.Model.kernel.K(self.X_Train,Xt).numpy()
        ktt = self.Model.kernel.K(Xt,Xt).numpy()
        
        sigma = self.Model.likelihood.variance.numpy()
        I = np.eye(self.kxx.shape[0],self.kxx.shape[1])
        
        kxt_trans = np.transpose(kxt)
        
        if all(v is None for v in [covIn, meanIn]):
            Kx = self.kxx + sigma*I
            m = self.mx 
        else:
            Kx = covIn + sigma*I
            m = meanIn
            
        Kx_inv = np.linalg.pinv(Kx)
        
        mean_g = mt + kxt_trans @ Kx_inv @ (self.Y_Train - m)
        # mean_g = kxt_trans @ Kx_inv @ (self.Y_Train)
        var_g  = ktt - kxt_trans @ Kx_inv @ kxt
        var_g = np.diagonal(var_g)
        
        return mean_g, var_g
    
    def getCov(self,X1,X2=[]):                  
        if X2:
            K = self.kernel.K(X1,X2)
        else:
            K = self.kernel.K(X1)
        self.K = K
        return K
    
    def recursiveGPR(self,Xt,Yt):
        #Model must already be initialized befor using recursiveGPR!
        
        #Initialize k_prior and mu_prior in first run
        if not hasattr(self, 'k_prior'): 
            self.k_prior = self.Model.kernel.K(self.X_base, self.X_base).numpy()
        if not hasattr(self, 'mu_prior'):
            self.mu_prior = self.Model.mean_function(self.X_base).numpy()    
        
        # self.temp = self.mu_prior
        
        
        #Covariances
        ktx = self.Model.kernel.K(Xt, self.X_base).numpy()
        kxt = self.Model.kernel.K(self.X_base, Xt).numpy()
        ktt = self.Model.kernel.K(Xt, Xt).numpy()
        # kxx = self.kxx
        kxx_inv = self.kxx_inv
        # C = np.vstack((np.hstack((kxx,kxt)), np.hstack((ktx,ktt))))

        #Means
        mt = self.Model.mean_function(Xt).numpy()
        mx = self.mx
        # mu_prior = np.vstack((mx,mt))
        
        #Inference    
        Jt = ktx @ kxx_inv
        Jt_T = np.transpose(Jt)
        mu_p = mt + Jt @ (self.mu_prior - mx)
        B = ktt - Jt @ kxt
        Cpt = B + Jt @ self.k_prior @ Jt_T
        
        #Update
        sigma = self.Model.likelihood.variance.numpy()
        # sigma = 0.01
        I = np.eye(Cpt.shape[0],Cpt.shape[1])
        Gt = self.k_prior @ Jt_T @ np.linalg.pinv(Cpt + sigma*I)
        mu_post = self.mu_prior + Gt @ (Yt - mu_p)
        # mu_post = self.mu_prior
        k_post = self.k_prior - Gt @ Jt @ self.k_prior
        # k_post = self.k_prior
        
        self.k_prior = k_post
        self.mu_prior = mu_post
        return self
    
    def recursivePredict(self, Xt):
        mt = self.Model.mean_function(Xt).numpy()   #mn
        ktt = self.Model.kernel.K(Xt,Xt).numpy()    #Kn
        kxt = self.Model.kernel.K(self.X_base, Xt).numpy() #Knx
        ktx = self.Model.kernel.K(Xt, self.X_base).numpy() #Knx
        Jt = ktx @ self.kxx_inv
        B = ktt - Jt @ kxt 
        Cpt = B + Jt @ self.k_prior @ Jt.T
        mean_t = mt + Jt @ (self.mu_prior - self.mx)
        I = np.eye(Cpt.shape[0])
        # var_t = Cpt + 0.01 * I
        var_t = Cpt + self.Model.likelihood.variance.numpy() * I
        var_t = np.diagonal(var_t)
        return mean_t, var_t          
    
    def At(self,Jt):
        I1 = np.eye(self.Nbase)
        Z1 = np.zeros((self.Nparams,self.Nbase))
        I2 = np.eye(self.Nparams)
        Z2 = np.zeros((self.Nbase,self.Nparams))
        Z3 = np.zeros((Jt.shape[0], self.Nparams))
        
        At1 = np.hstack((I1,Z2))
        At2 = np.hstack((Z1,I2))
        At3 = np.hstack((Jt,Z3))
        
        At = np.vstack((At1,At2,At3))
        
        return At
    
    def mu_w(self,Jt, Xt):
        Z1 = np.zeros((self.Nbase,1))
        Z2 = np.zeros((self.Nparams,1))
        b = self.mean(Xt) - Jt @ self.mean(self.X_base)
        
        mu_w = np.vstack((Z1,Z2,b))
        return mu_w
    
    def k_w(self,Jt, Xt):
        B = self.kernel(Xt,Xt) - Jt @ self.kernel(self.X_base, Xt)
        B = B.numpy()
        
        Z1 = np.zeros((self.Nbase+self.Nparams, self.Nbase+self.Nparams))
        Z2 = np.zeros((Xt.shape[0], self.Nbase+self.Nparams))
        
        temp1 = np.hstack((Z1, Z2.T))
        temp2 = np.hstack((Z2, B))
        
        k_w = np.vstack((temp1, temp2))
        
        return k_w
    
    def decompose(self,mu_pt, k_pt):
        mu_ut = mu_pt[0:self.Nbase+self.Nparams-1] #assumes noise parameter as the last parameter in params
        mu_ot = mu_pt[self.Nbase+self.Nparams-1:] #assumes noise parameter as the last parameter in params
        
        k_ut  = k_pt[0:self.Nbase+self.Nparams-1, 0:self.Nbase+self.Nparams-1]
        k_ot  = k_pt[self.Nbase+self.Nparams-1:, self.Nbase+self.Nparams-1:]
        k_uo  = k_pt[0:self.Nbase+self.Nparams-1, self.Nbase+self.Nparams-1:]
                
        
        #mu_ot[0] = noise
        #mu_ot[1:] = new observations
        #mu_ut[0:self.Nbase] = base vectors
        #mu_ut[Nbase:] = hyperparameters except noise
        return mu_ut, mu_ot, k_ut, k_ot, k_uo
    
    def recursiveGPR_HPO(self, Xt, Yt):        
        #Draw Sigma Points
        sigmaPoints = self.Points.sigma_points(x=self.params, P=self.covHPO)
        weights = self.Points.Wm
        
        # print(sigmaPoints)
        # print(weights)
        mu_pt = 0
        k_pt = 0
        for i in range(len(weights)):
            self.kernel = gpflow.kernels.SquaredExponential(lengthscales=sigmaPoints[i,0], variance=sigmaPoints[i,1])
            kxx = self.kernel(self.X_base).numpy()
            kxx_inv = np.linalg.pinv(kxx) 
            ktx = self.kernel(Xt, self.X_base).numpy() 
            Jt = ktx @ kxx_inv
            At = self.At(Jt)
            
            St = self.k_ge @ np.linalg.pinv(self.covHPO)
            mu_w = self.mu_w(Jt,Xt)
            k_w = self.k_w(Jt, Xt)
            
            mu_p_row1 = self.mu_prior + np.array([St @ (sigmaPoints[i,:] - self.params)]).T
            mu_pi = At @ np.vstack((mu_p_row1, np.array([sigmaPoints[i,:]]).T)) + mu_w
            
            k_p_row1 = np.hstack((self.k_prior - St @ self.k_ge.T , np.zeros((self.Nbase,self.Nparams))))
            k_p_row2 = np.zeros((self.Nparams,self.Nbase+self.Nparams))
            k_pi = At @ np.vstack((k_p_row1, k_p_row2)) @ At.T + k_w
            
            mu_pt += weights[i] * mu_pi
            k_pt  += weights[i] * ((mu_pi-mu_pt) @ (mu_pi-mu_pt).T + k_pi)
            
        mu_ut, mu_ot, k_ut, k_ot, k_uo = self.decompose(mu_pt, k_pt)
                
        mu_yt = mu_ot[1:]
        k_yt  = k_ot[1:,1:] + k_ot[0,0] + mu_ot[0]**2
        k_oyt = k_ot[0,1:]      #To Do!! Check, if this is correct! Reduces to a vector!
        
        Gt = k_oyt @ np.linalg.pinv(k_yt)
        Lt = k_uo @ np.linalg.pinv(k_ot)
        hT = np.hstack((np.array([[1.0]]), np.zeros((1,len(mu_ot)-1))))
        
        mu_et = mu_ot + Gt @ (Yt - mu_yt)
        k_et  = k_ot - Gt @ k_yt @ Gt.T
        
        mu_ut += Lt @ (mu_et - mu_ot)
        k_ut  += Lt @ (k_et - k_ot) @ Lt.T
        
        mu_zt = np.vstack((hT @ mu_et, mu_ut))
        k_zt_row1  = np.hstack((hT @ k_et @ hT.T, hT @ k_et @ Lt.T))
        k_zt_row2  = np.hstack((Lt @ k_et @ hT.T, k_ut))
        k_zt = np.vstack((k_zt_row1, k_zt_row2))
        
        self.params = np.array([mu_zt[-2], mu_zt[-1], mu_zt[0]]).reshape(3)
        self.mu_prior = mu_zt[1:self.Nbase+1]
        self.k_prior  = k_zt[1:self.Nbase+1, 1:self.Nbase+1]
        temp1_covHPO = k_zt[-(self.Nparams-1):,-(self.Nparams-1):]
        temp2_covHPO = np.hstack((k_zt[0,0], k_zt[0,-(self.Nparams-1):]))
        temp3_covHPO = np.hstack((temp1_covHPO, k_zt[-(self.Nparams-1):,0:1]))
        covHPO  = np.vstack((temp3_covHPO, temp2_covHPO)) #does not stay positive definite!! But must be updated to change sigma points
        # print(covHPO)
        
        #Update of covHPO is missing
        #Update of self.k_ge is missing
        #Update of mu_ut and k_ut is missing (??? needed???)
        
        return self

    def predict_Confidence(self,Y):
        Y_lower = Y[0] - 1.96 * np.sqrt(Y[1])
        Y_upper = Y[0] + 1.96 * np.sqrt(Y[1])
        Y_Conf = np.hstack((Y_lower,Y_upper))
        return Y_Conf
    
    def test_simpleGPR(self, X=None, Y=None):
        self.intializeGPR(X=X, Y=Y)
        
        #Predict Output
        Y_Train_pred = self.Model.predict_f(self.X_Train)
        Y_Test_pred = self.Model.predict_f(self.X_Test)
        
        #Transform EagerTensor to numpy array
        Y_Train_pred = np.array(Y_Train_pred)
        Y_Test_pred = np.array(Y_Test_pred)
        
        #Calculate Confidence Interval
        Y_Train_pred_Conf = self.predict_Confidence(Y_Train_pred)
        Y_Test_pred_Conf = self.predict_Confidence(Y_Test_pred)
        
        # Y_Train_pred,_ = self.get_Mean_and_Variance(self.X_Train)
        # Y_Test_pred,_ = self.get_Mean_and_Variance(self.X_Test)
        
        #Plot
        plt.figure()
        plt.plot(self.X,self.Y,'k.-',label='True Data')
        plt.plot(self.X_Train,Y_Train_pred[0],'b.', label='Predicted Train Data')
        plt.plot(self.X_Test,Y_Test_pred[0],'r.', label='Predicted Test Data')

        plt.plot(self.X_Train, Y_Train_pred_Conf[:,0],'.',markersize=1,alpha=0.5,color=(0.25,0.25,0.25))  #lower bound confidence
        plt.plot(self.X_Train, Y_Train_pred_Conf[:,1],'.',markersize=1,alpha=0.5,color=(0.25,0.25,0.25))  #upper bound confidence
        plt.plot(self.X_Test, Y_Test_pred_Conf[:,0],'.',markersize=1,alpha=0.5,color=(0.25,0.25,0.25))  #lower bound confidence
        plt.plot(self.X_Test, Y_Test_pred_Conf[:,1],'.',markersize=1,alpha=0.5,color=(0.25,0.25,0.25))  #upper bound confidence
        
        plt.legend()
        plt.title('GPR Model')
        plt.show()
    
    def test_recursiveGPR_HPO(self,Xt,Yt,batchSize=1):      
        for i in range(0, Xt.shape[0]+1, batchSize):
            self.recursiveGPR_HPO(Xt[i:i+batchSize,:], Yt[i:i+batchSize,:])
                   
        # mean_g, var_g = self.recursivePredict(Xt)    
    
    
    def test_recursiveGPR(self,Xt,Yt,batchSize=1):      
        for i in range(0, Xt.shape[0]+1, batchSize):
            self.recursiveGPR(Xt[i:i+batchSize,:], Yt[i:i+batchSize,:])
            # if i == 0:
            #     temp = self.temp.T
            # else:
            #     temp = np.append(temp,self.temp.T, axis=0)
                
        # plt.figure()
        # for i in range(temp.shape[1]):
        #     plt.plot(temp[:,i])
        # plt.show()
        
        # mean_g, var_g = self.get_Mean_and_Variance(Xt, covIn=self.k_prior, meanIn=self.mu_prior)
        mean_g, var_g = self.recursivePredict(Xt)
        
        Y_lower = mean_g - 1.96 * np.sqrt(var_g)
        Y_upper = mean_g + 1.96 * np.sqrt(var_g)
        Y_conf = np.vstack((Y_lower,Y_upper)).T
        
        #Plot
        plt.figure()
        plt.plot(self.X,self.Y,'k.-',label='True Data')
        plt.plot(Xt,mean_g,'g.', label='Recursivly Predicted Data')
        
        plt.fill_between(Xt.ravel(), Y_conf[:,0], Y_conf[:,1], color='lightgrey',label='95% Confidence Interval')
        
        plt.legend()
        plt.title('Recursive GPR Model')
        plt.show()
        
        

#------------------------------------------------------------------------------
#Read data into numpy array
# X = np.linspace(0,1,1000)
# X = np.array([X]).T
# f = 30
# Y = np.sin(X*f)

N = 2000
X = np.linspace(-10,10,N)
X = np.array([X]).T
X_base = np.linspace(-10,10,25)
X_base = np.array([X_base]).T

def testFunc(X):
     N = len(X)
     Y = X/2 + 25*X/(1+X**2)*np.cos(X) + 0.1*np.random.randn(N,1)
     return Y

Y = testFunc(X)
Y_base = testFunc(X_base)

myGPmodel = GPmodel(X,Y,test_amount=0.20, kernel=gpflow.kernels.SquaredExponential(variance=1,lengthscales=0.5)) 

# X_Train = myGPmodel.X_Train

Xt = myGPmodel.X_Test
Yt = myGPmodel.Y_Test
   
if __name__ == '__main__':
    # myGPmodel.intializeGPR(X=X_base, Y=Y_base)
    
    # covMod = myGPmodel.Model.kernel.K(X_Train, X_Train).numpy()
    # meanMod = myGPmodel.Model.mean_function(X_Train).numpy()
    # mean_g, var_g = myGPmodel.get_Mean_and_Variance(myGPmodel.X_Test, covIn=covMod, meanIn=meanMod)
    
    
    # myGPmodel.test_simpleGPR()
    # mean_g, var_g = myGPmodel.get_Mean_and_Variance(myGPmodel.X_Test)
    
    
    #The following should equal mean_g and var_g from own function
    # out_GPflow = myGPmodel.Model.predict_f(myGPmodel.X_Test)
    # mean_g0 = out_GPflow[0].numpy()
    # var_g0 = out_GPflow[1].numpy()
    
    # myGPmodel.recursiveGPR(Xt,Yt)
    
    # myGPmodel.test_simpleGPR(X=X_base, Y=Y_base)
    # myGPmodel.test_recursiveGPR(X, Y, batchSize=10)
    
    
    myGPmodel.intialize_recursiveGPR_HPO(Xbase=X_base)
    # myGPmodel.recursiveGPR_HPO(Xt, Yt)
    myGPmodel.test_recursiveGPR_HPO(X, Y, batchSize=10)
    
    





# #------------------------------------------------------------------------------
# #Extrapolation Data only used for Testing
# X_ext = np.hstack((np.linspace(-1,0,500), np.linspace(1,2,500)))
# X_ext = np.array([X_ext]).T
# Y_ext = np.sin(X_ext*f)
# #------------------------------------------------------------------------------
# #Split Data into Training and Testing
# #Train-Test Split with ratio test_size, 
# #random_state should be set so that the shuffling is reproducible, e.g. =42
# X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, test_size=0.2, random_state=42)
# #------------------------------------------------------------------------------

# myModel = gpflow.models.GPR((X_Train,Y_Train), kernel=gpflow.kernels.SquaredExponential())

# opt = gpflow.optimizers.Scipy()
# opt.minimize(myModel.training_loss, myModel.trainable_variables)


# Y_Train_pred = myModel.predict_f(X_Train)
# Y_Test_pred = myModel.predict_f(X_Test)

# Y_Train_pred = np.array(Y_Train_pred)
# Y_Test_pred = np.array(Y_Test_pred)

# #95% Confidence Interval
# Y_Train_pred_Conf = predict_Confidence(Y_Train_pred)
# Y_Test_pred_Conf = predict_Confidence(Y_Test_pred)

# #Plot
# plt.figure()
# plt.plot(X,Y,'k.-',label='True Data')
# plt.plot(X_Train,Y_Train_pred[0],'b.', label='Predicted Train Data')
# plt.plot(X_Test,Y_Test_pred[0],'r.', label='Predicted Test Data')

# plt.plot(X_Train, Y_Train_pred_Conf[:,0],'.',markersize=1,alpha=0.5,color=(0.25,0.25,0.25))  #lower bound confidence
# plt.plot(X_Train, Y_Train_pred_Conf[:,1],'.',markersize=1,alpha=0.5,color=(0.25,0.25,0.25))  #upper bound confidence
# plt.plot(X_Test, Y_Test_pred_Conf[:,0],'.',markersize=1,alpha=0.5,color=(0.25,0.25,0.25))  #lower bound confidence
# plt.plot(X_Test, Y_Test_pred_Conf[:,1],'.',markersize=1,alpha=0.5,color=(0.25,0.25,0.25))  #upper bound confidence

# #Test Extrapolation Capability
# if Extrapolation:
#     Y_ext_pred = myModel.predict_f(X_ext)
#     Y_ext_pred_Conf = predict_Confidence(Y_ext_pred)
#     plt.plot(X_ext,Y_ext, 'k.', label='Extrapolated True Data')
#     plt.plot(X_ext, Y_ext_pred[0], 'g.', label='Extrapolated Predicted Data')
#     plt.plot(X_ext, Y_ext_pred_Conf[:,0],'.',markersize=1,alpha=0.5,color=(0.25,0.25,0.25))  #lower bound confidence
#     plt.plot(X_ext, Y_ext_pred_Conf[:,1],'.',markersize=1,alpha=0.5,color=(0.25,0.25,0.25))  #upper bound confidenc
    
# plt.legend()
# plt.title('GPR Model')
# plt.show()