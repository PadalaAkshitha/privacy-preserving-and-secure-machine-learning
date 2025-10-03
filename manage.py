#import require python classes and packages
import pandas as pd #pandas to read and explore dataset
import numpy as np
import matplotlib.pyplot as plt #use to visualize dataset vallues
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm #SVM class
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics 
#class to apply differential privacy on dataset to secured model from attacks
from pydp.algorithms.laplacian import BoundedSum
#kmeans to cluster related data and then removed out unrelated data to increase model security 
from sklearn.cluster import KMeans
#homomorphic class to perturb training data
from Homomorphic import perturbData



#loading and displaying heart disease dataset
dataset = pd.read_csv("Dataset/heart.csv")
columns = dataset.columns
dataset




#loading and displaying heart disease dataset
dataset = pd.read_csv("Dataset/heart.csv")
columns = dataset.columns
dataset


#visualizing distribution of numerical data
dataset.hist(figsize=(10, 8))
plt.title("Representation of Dataset Attributes")
plt.show()




#finding & plotting graaph of normal and heart disease patinets available in dataset
#visualizing class labels count found in dataset
names, count = np.unique(dataset['target'].ravel(), return_counts = True)
height = count
bars = ['Normal', 'Heart Patients']
y_pos = np.arange(len(bars))
plt.figure(figsize = (4, 3)) 
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.xlabel("Dataset Class Label Graph")
plt.ylabel("Count")
plt.show()





#securing ML model by applying extension KMEANS clustering which will group similar data into same cluster and put
#unrelated data into different cluster and by applying this extension technique we will train model with related data
print("Dataset Size before removing unrelated Data : "+str(dataset.shape[0]))
data = dataset.values
X = data[:,0:data.shape[1] - 1]
Y = data[:,data.shape[1] - 1]
XX = []
YY = []
#defining KMEANS to group related data
kmeans = KMeans(n_clusters=3, n_init=50, random_state=1)
kmeans.fit(X)
clusters = kmeans.labels_
labels, count = np.unique(clusters, return_counts=True)
irrelevant = 0
counter = X.shape[0]
#find out label of unrelated data
for i in range(len(count)):
    if count[i] < counter:
        counter = count[i]
        irrelevant = labels[i]
#collect only related data and avoid unrelated data        
for i in range(len(clusters)):
    if clusters[i] != irrelevant:
        XX.append(X[i])
        YY.append(Y[i])
X = np.asarray(XX)
Y = np.asarray(YY)
print("Dataset Size after removing unrelated Data : "+str(X.shape[0]))





#now apply Differential Privacy algorithm on dataset training features to provide security to model
df_X = []
dp = BoundedSum(epsilon= 1.5, lower_bound =  0.1, upper_bound = 100, dtype ='float') 
noise = dp.quick_result(dataset['age'].to_list())
for i in range(len(X)):
    temp = []
    for j in range(len(X[i])):
        temp.append(X[i,j] + noise)
    df_X.append(temp)    
df_X = np.asarray(df_X)   
temp = pd.DataFrame(df_X, columns = columns[0:len(columns)-1].values)
print("Training Features after applying Differential Privacy")
temp




#define global variables to save accuracy and other metrics
accuracy = []
precision = []
recall = []
fscore = []




#function to calculate all metrics
def calculateMetrics(algorithm, testY, predict):
    labels = ['Normal', 'Heart Patient']
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    print(algorithm+" Accuracy  : "+str(a))
    print(algorithm+" Precision : "+str(p))
    print(algorithm+" Recall    : "+str(r))
    print(algorithm+" FSCORE    : "+str(f))
    conf_matrix = confusion_matrix(testY, predict)
    fig, axs = plt.subplots(1,2,figsize=(10, 4))
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axs[0]);
    ax.set_ylim([0,len(labels)])
    axs[0].set_title(algorithm+" Confusion matrix") 

    random_probs = [0 for i in range(len(testY))]
    p_fpr, p_tpr, _ = roc_curve(testY, random_probs, pos_label=1)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='orange',label="True classes")
    ns_fpr, ns_tpr, _ = roc_curve(testY, predict, pos_label=1)
    axs[1].plot(ns_fpr, ns_tpr, linestyle='--', label='Predicted Classes')
    axs[1].set_title(algorithm+" ROC AUC Curve")
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive rate')
    plt.show()





#split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(df_X, Y, test_size = 0.2)
print("Total records found in dataset = "+str(X.shape[0]))
print("Total features found in dataset= "+str(X.shape[1]))
print("80% dataset for training : "+str(X_train.shape[0]))
print("20% dataset for testing  : "+str(X_test.shape[0]))



#training and evaluating performance of decision tree algorithm
dt_cls = DecisionTreeClassifier()
dt_cls.fit(X_train, y_train)#train algorithm using training features and target value
predict =dt_cls.predict(X_test)#perform prediction on test data
#call this function with true and predicted values to calculate accuracy and other metrics
calculateMetrics("Decision Tree Differential Privacy", y_test, predict)



#now perturbed training data using Homomorphic Encryption
homo_X = perturbData(X)#calling PerturnData function from Homomorphic class to encrypt dataset
temp = pd.DataFrame(homo_X, columns = columns[0:len(columns)-1].values)
print("Training Features after applying Homomorphic Encryption")
temp




X_train, X_test, y_train, y_test = train_test_split(homo_X, Y, test_size = 0.2)
#training and evaluating performance of decision tree algorithm
dt_cls = DecisionTreeClassifier()
dt_cls.fit(X_train, y_train)#train algorithm using training features and target value
predict =dt_cls.predict(X_test)#perform prediction on test data
#call this function with true and predicted values to calculate accuracy and other metrics
calculateMetrics("Decision Tree Homomorphic Encryption", y_test, predict)




#comparison graph between all algorithms
df = pd.DataFrame([['Differential Privacy','Accuracy',accuracy[0]],['Differential Privacy','Precision',precision[0]],['Differential Privacy','Recall',recall[0]],['Differential Privacy','FSCORE',fscore[0]],
                   ['Homomorphic Encryption','Accuracy',accuracy[1]],['Homomorphic Encryption','Precision',precision[1]],['Homomorphic Encryption','Recall',recall[1]],['Homomorphic Encryption','FSCORE',fscore[1]],
                  ],columns=['Parameters','Algorithms','Value'])
df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', figsize=(6, 3))
plt.title("All Algorithms Performance Graph")
plt.show()




#display all algorithm performnace
algorithms = ['Differential Privacy', 'Homomorphic Encryption']
data = []
for i in range(len(accuracy)):
    data.append([algorithms[i], accuracy[i], precision[i], recall[i], fscore[i]])
data = pd.DataFrame(data, columns=['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'FSCORE'])
data






