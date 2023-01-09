import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


from tqdm.notebook import trange, tqdm

NUMERICALTHRESHOLD = 1e-10

class SLN:
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_weights = np.random.rand(input_size,hidden_size)
        self.output_weights = np.random.rand(hidden_size,output_size)        
        self.biases = np.random.rand(hidden_size)
    
    def hidden_layer(self,X):
        G= np.dot(X, self.input_weights)
        G = G + self.biases
        return(self.sigmoid(G))
    
    def predict(self,X):
        H=self.hidden_layer(X)
        return(np.dot(H, self.output_weights))        
               
    def train(self, X, y):
        if( (X.shape[1] == self.input_size) and  (y.shape[1] == self.output_size) ):
            H=self.hidden_layer(X)
            self.clear(H)
            self.H=H
            Hplus = np.linalg.pinv(H)
            self.clear(Hplus)
            self.Hplus=Hplus
            self.output_weights = np.dot(Hplus,y)
            self.clear(self.output_weights)
            #self.data=X
            self.labels=y
        else:
            print("Error: cannot match shapes:", X.shape, " with ", self.input_weights.shape, " and ", self.output_weights.shape, " with ", y.shape)  

    def loss(self,X,y):         
        diff = self.predict(X) - y
        n=len(diff)
        return(1/n*sum([np.linalg.norm(x)**2 for x in diff]))
            
    def sigmoid(self,z):
        return 1/(1 + np.exp(-z))

    def clear(self, A):
        n,m=A.shape
        for i in range(n):
            for j in range(m):
                if(abs(A[i,j]) < NUMERICALTHRESHOLD):
                    A[i,j]=0.0                 
    
    def recPinv(self,A,a,Ap):        
        d=np.dot(a,Ap)
        c=a-np.dot(d,A)
        flag=False
        self.clear(c)
        if(not np.all(c==0)):
            b=(1/np.dot(c,c.transpose()))*c.transpose()
        else:
            flag=True
            k = (1 + np.dot(d,d.transpose()))
            b = (1/k)*np.dot(Ap,d.transpose())
        B = Ap-np.dot(b,d)
        self.clear(B)
        return(np.hstack((B,b)),flag)
                    
    def sequential_learning(self, x, y):                
        a=self.hidden_layer(x)
        A=self.H
        #p=self.Hplus
        #new,flag = self.recPinv(A,a,Ap)    
        #elf.clear(Hnew) 
        Anew = np.vstack((A,a))        
        Hnew2 = np.linalg.pinv(Anew)
        #U1=np.dot(Hnew, Anew)
        #self.clear(U1)
        #U2=np.dot(Hnew2, Anew)
        #print(U1);print(U2)
        #print("---")
        ynew = np.vstack((self.labels,y))        
        #weights = np.dot(Hnew,ynew)
        weights = np.dot(Hnew2,ynew)
        #self.clear(weights)
        self.output_weights = weights
        return(False)

    
####################### ELM ende########################

 
def testit(model, N=100):
    plotx = np.arange(N)*(1/N)
    ploty = np.sin(plotx*np.pi*6)
    plotx.resize((N,1))
    ploty.resize((N,1))
    x=plotx.copy()
    y=ploty.copy()
    x.resize((N,1))
    y.resize((N,1))
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
    
    model.train(xtrain,ytrain)        
    
    ytrain_pred = model.predict(xtrain)
              
    for i in trange(len(xtrain), desc="learning"):            
    #for i in range(len(xtrain)):            
            xx=np.array([xtrain[i]])
            yy=np.array([ytrain[i]])
            model.sequential_learning(xx,yy)
   
    ytest_pred = model.predict(xtest)
    
    #Plot
    plt.figure()
    plt.plot(x,y,'k.-',label='True Data')
    plt.plot(xtrain,ytrain_pred,'b.', label='Predicted Train Data')
    plt.plot(xtest,ytest_pred,'r.', label='Predicted Test Data')
    plt.legend()
    plt.show()


testit(SLN(1,500,1), N=500)

