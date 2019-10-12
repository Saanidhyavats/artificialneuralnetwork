import numpy as np
feature_set=np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,0,0],[1,1,0],[0,1,1]])
labels=np.array([[1],[1],[1],[1],[1],[0],[1],[1]])
labels.reshape(8,1)
np.random.seed(0)
weights=np.random.rand(3,1)
bias=np.random.rand(1)
alpha=0.05  #learning rate

def sigmoid(x):
  return(1/1+np.exp(-x)) 
def sigmoid_deriv(x):
  return (sigmoid(x)*(1-sigmoid(x)))
for epoch in range(0,1000):
 inputs=feature_set
 xw=np.dot(feature_set,weights)+bias
 z=sigmoid(xw) #ypred
 error=z-labels
 print(np.sum(error))
 #backpropagation algorithm
 dcost_dpred=error
 dz_dxw=sigmoid_deriv(xw)
 dcost_dxw=dcost_dpred* dz_dxw #corresponding elements of matrices will get multiplied
 inputs=np.transpose(feature_set)
 weights-=alpha *np.dot(inputs,dcost_dxw)
 for num in dcost_dxw:
    bias-=alpha*num
  
  
  
   

