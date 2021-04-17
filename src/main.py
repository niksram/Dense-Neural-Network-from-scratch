# Design of a Neural Network from scratch

# *************<IMP>*************
# Mention hyperparameters used and describe functionality in detail in this space
'''
PESU-MI_1903_1957_1972

PRESENT HYPER-PARAMETERS

   The present neural network configuration is set to consist of 2 Hidden layers
      -1st Hidden layer = 5, activation='sigmoid'
      -2nd Hidden layer = 4, activation='sigmoid'
   The current learning rate = 0.01
   epochs = 350
   train:test split = 70:30
   batch-size in fit = (set to default 0.3)
   Adam Optimiser parameters
      beta1 parameter - 0.9
      beta2 parameter - 0.99
      epsilon = 10^-8
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''


class NN:
    def activate(self, activation, X): # can be chosen as required
        if activation == 'sigmoid':
            return  np.vectorize(lambda x: 1/(1 + np.exp(-x)))(X)
        elif activation == 'relu':
            return np.vectorize(lambda x: x if x > 0 else 0)(X)
        elif activation == 'tanh':
            return np.tanh(X)

    def differentials_of_activations(self, activation, X): # these are the differentials of activations which are called during back propogation
        if activation == 'sigmoid':
            return np.vectorize(lambda x: x*(1-x))(X)
        elif activation == 'relu':
            return np.vectorize(lambda x: 1 if x >= 0 else 0)(X)
        elif activation == 'tanh':
            return np.vectorize(lambda x: 1-np.tanh(x)**2)(X)

    def __init__(self, hidden_layers_config=[]): # to initilise the Neural Network
        self.Units = [] # list of the number of neurons in each hidden layer and the final layer
        self.Weights = [] # list of the weight matrix of hidden layers
        self.Activations = [] # the activation functions of the hidden layers are listed here
        self.Units = [i[0] for i in hidden_layers_config]+[1] # 1 for output layer 
        self.Activations = [i[1] for i in hidden_layers_config]+["sigmoid"]# sigmoid for output layer
    ''' X and Y are dataframes '''

    def init_weight(self, X): #initialize weight matrices dimensions based on number of columns of X 
        r,c=len(X.columns),self.Units[0] 
        act=self.Activations[0]
        if(act in ('sigmoid', 'relu')):
            self.Weights.append(np.random.randn(r+1,c)*np.sqrt(2/c)) # He initialisation input to first hidden         
        else:
            self.Weights.append(np.random.randn(r+1,c)*np.sqrt(1/(r+c))) # Xavier initialisation input to first hidden        
        for i in range(1,len(self.Units)):
            r,c=c,self.Units[i]
            if self.Activations[i] in ('sigmoid', 'relu'):
                self.Weights.append(np.random.randn(r+1,c)*np.sqrt(2/c)) # He initialisation 
            elif self.Activations[i] == 'tanh':
                self.Weights.append(np.random.randn(r+1,c)*np.sqrt(1/(r+c))) # Xavier initialisation 
    
    def forward(self,X): # performs forward propogation
        outputs=[] #stores intermediate outputs for every layer
        outputs.append(self.activate(self.Activations[0],np.dot(np.concatenate((np.ones((X.shape[0],1)),X),axis=1),self.Weights[0]))) # np.ones is added to get multipled with the bias
        for i in range(1,len(self.Weights)):
            outputs.append(self.activate(self.Activations[i],np.dot(np.concatenate((np.ones((outputs[-1].shape[0],1)),outputs[-1]),axis=1),self.Weights[i])))
        return outputs

    def fit(self, X, y,epochs=200,lr=0.05,batch_ratio=1,beta1 = 0.9,beta2 = 0.99 ,epsilon = 0.00000001): # the wrapper function that trains the neural network
        self.init_weight(X) # weights are initialised
        m_v_list = [] # the Adam Optimiser mean and varience parameters are preserved for next iteration
        for i in range(len(self.Weights)-1):
            m_v_list.append([0,0])
        for i in range(epochs):
            if batch_ratio==1:
                xtrain=X
                ytrain=y
            else: # split based on batch ratio is made when the fraction is less then 1
                xtrain, xtest, ytrain, ytest = train_test_split(X,y, train_size=batch_ratio)
            outputs = self.forward(xtrain) #forward propogation 
            m_v_list = self.backward(xtrain, ytrain, outputs,m_v_list,i+1,lr,beta1,beta2,epsilon) 

    def backward(self,X,y,outputs,m_v_list,t,lr,beta1,beta2,epsilon):
        op_error=y-outputs[-1] # differential of MSE
        delta=op_error*(self.differentials_of_activations("sigmoid",outputs[-1]))
        updates=[delta] # the list of correction values is preserved and later the weights are updated accordingly
        for i in range(len(self.Weights)-1,0,-1):
            error=delta.dot(self.Weights[i].T)[:,1:] # bias component is removed while back propogating to the previous layer
            delta=error*self.differentials_of_activations(self.Activations[i],outputs[i-1])
            updates.append(delta)
        self.Weights[0]+=lr*np.concatenate((np.ones((X.shape[0],1)),X),axis=1).T.dot(updates[-1])
        idx=1
        for i in range(len(updates)-2,-1,-1): # weight correction iteratively using Adam Optimising techniques
            grad = np.concatenate((np.ones((outputs[idx-1].shape[0],1)),outputs[idx-1]),axis=1).T.dot(updates[i]) # the gradient of the particular weight matrix
            m_v_list[i][0] = beta1 * m_v_list[i][0] + (1 - beta1) * grad # mean moment
            m_corrected = m_v_list[i][0] / (1 - np.power(beta1, t)) # corrected mean moment based on the parameter
            m_v_list[i][1] = beta2 * m_v_list[i][1] + (1 - beta2) * np.square(grad) # varience moment
            v_corrected = m_v_list[i][1] / (1 - np.power(beta2, t)) # # corrected varience moment based on the parameter
            self.Weights[idx]+= lr * (m_corrected / (np.sqrt(v_corrected) + epsilon)) # updating weights accordingly
            idx+=1
        return m_v_list

    def predict(self, df):
        return np.round(self.forward(df)[-1]) # round value of the sigmoid output

    def CM(self,y_test, y_test_obs):
        '''
        Prints confusion matrix
        y_test is list of y values in the test dataset
        y_test_obs is list of y values predicted by the model

        '''

        for i in range(len(y_test_obs)):
            if(y_test_obs[i] > 0.6):
                y_test_obs[i] = 1
            else:
                y_test_obs[i] = 0

        cm = [[0, 0], [0, 0]]
        fp = 0
        fn = 0
        tp = 0
        tn = 0

        for i in range(len(y_test)):
            if(y_test[i] == 1 and y_test_obs[i] == 1):
                tp = tp+1
            if(y_test[i] == 0 and y_test_obs[i] == 0):
                tn = tn+1
            if(y_test[i] == 1 and y_test_obs[i] == 0):
                fp = fp+1
            if(y_test[i] == 0 and y_test_obs[i] == 1):
                fn = fn+1
        cm[0][0] = tn
        cm[0][1] = fp
        cm[1][0] = fn
        cm[1][1] = tp
        p = tp/(tp+fp)
        r = tp/(tp+fn)
        f1 = (2*p*r)/(p+r)

        print("Confusion Matrix : ")
        print(cm)
        print("Accuracy : ",(tp+tn)/(tp+fp+tn+fn))
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")
        print("\n\n\n")


X = pd.read_csv('../data/pre_processed.csv')
y = np.array(X.pop('Result')).reshape(-1,1) #extracting last(Result) column 

xtrain, xtest, ytrain, ytest = train_test_split(X,y, train_size=0.7) # 70 30 split


nn = NN([[5, 'relu'], [3, 'tanh'],[2, 'tanh']]) #corresponsing setup of 5 nodes in the first layer and 4 nodes in the next hidden layer was found to give reasonably optimal results
nn.fit(xtrain, ytrain,350,0.032,0.4)
# nn.fit(xtrain, ytrain,300,0.01,0.3)
print("*"*6,"Training Dataset","*"*6,end='\n')
nn.CM(ytrain,nn.predict(xtrain))
print("*"*6,"Testing Dataset","*"*6)
nn.CM(ytest,nn.predict(xtest))         
