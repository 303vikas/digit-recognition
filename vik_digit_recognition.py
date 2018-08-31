import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,metrics,datasets
import pandas as pd


str1=[898,1257,1437]

str3=['poly','rbf','linear']
y = ['50:50','70:30','80:20']
poly = []
rbf =[]
linear =[]
acc = np.zeros((3,3))

data = datasets.load_digits()

for i in range(0,len(str1)):
    
        
            
            train_data = data.data[0:str1[i],:]
            train_target = data.target[0:str1[i]]

            test_data = data.data[str1[i]:,:]
            test_target = data.target[str1[i]:]

            for k in range (0,len(str3)):
                    svm_model = svm.SVC(kernel = str3[k] )
                
                    svm_model = svm_model.fit(train_data,train_target)
                    output = svm_model.predict(test_data)
                            
                    acc[i][k] = metrics.accuracy_score(test_target,output)
                    
result = pd.DataFrame(acc,index = y,columns = str3)
print("Dataframe with kernel model and their accuracy: ")
print(result)