# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.

import api
from pacman import Directions
from game import Agent
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from math import sqrt

np.seterr(divide = 'ignore')

class Classifier:

    #api.getFeatureVector(state)

    def __init__(self):
        self.class_counts = []
        self.class_prob = []
        self.means = None
        self.variances = None

    def reset(self):
        pass
    
    def organise_dataset(self, X_train, y_train):
        df = pd.DataFrame(X_train)
        df['Class'] = y_train
        df_sorted = df.sort_values('Class')
        return df_sorted

    def get_mean_std(self, df):
        # Need to calculate the mean and standard deviation for each feature 
        for i in range(len(np.unique(df.Class))):
            self.class_counts.append(str(np.count_nonzero(df.Class == (i))))
        feature_class_means = np.zeros((len(np.unique(df.Class)), len(df.iloc[0])-1))
        feature_class_variance = np.zeros((len(np.unique(df.Class)), len(df.iloc[0])-1))
        for i in range(len(np.unique(df.Class))):
            df_label = df.loc[df['Class'] == i]
            for j in range(len(df.iloc[0])-1):
                feature_class_means[i, j] = df_label[j].sum()/float(self.class_counts[i])
                feature_class_variance[i, j] = df_label[j].var()
        return feature_class_means, feature_class_variance

    def get_class_probabilities(self, df):
        # Find the probabilities of finding each class in the training set
        for i in range(len(self.class_counts)):
            self.class_prob.append(str(int(self.class_counts[i]) / len(df)))
        print(self.class_prob)
        return self.class_prob

    def density_function(self, x, mean, variance):
        # input the arguments into the probability density function
        try:
            prob = 1/(np.sqrt(2*np.pi*variance)) * np.exp((-(x-mean)**2)/(2*variance))
        except ZeroDivisionError:
            prob = 0
        return prob

    def get_pred(self, X_test, num_classes, num_inputs):
        y_pred = []
        for x in range(num_inputs):
            y_probabilities = []
            for i in range(num_classes):
                probability = 0
                for j in range(len(X_test.iloc[0])):
                    mean = self.means[i, j]
                    variance = self.variances[i, j]
                    if variance != 0:
                        if X_test.ndim == 1:
                            probability = probability + np.log(np.nan_to_num(self.density_function(X_test.iloc[j], mean , variance)))
                        else:
                            probability = probability + np.log(np.nan_to_num(self.density_function(X_test.iloc[x, j], mean , variance)))
                probability = np.log(float(self.class_prob[i])) + probability
                y_probabilities.append(probability)
            y_pred.append(np.argmax(y_probabilities))
        if X_test.ndim == 1:
            return y_pred[0]
        return y_pred

    def fit(self, data, target):
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
        df_sorted = self.organise_dataset(X_train, y_train)
        self.means, self.variances = self.get_mean_std(df_sorted)
        self.class_prob = self.get_class_probabilities(df_sorted)
        X_test = pd.DataFrame(X_test)
        num_inputs = len(X_test)
        y_pred = self.get_pred(X_test, len(self.class_counts), num_inputs)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        print("Naive Bayes Accuracy:", accuracy)
        prf = precision_recall_fscore_support(y_test, y_pred, average='macro')
        print("Naive Bayes Precision:", prf[0])
        print("Naive Bayes Recall:", prf[1])
        print("Naive Bayes F Score:", prf[2])
        print(y_pred)

    def predict(self, data, legal=None):
        print("New input: ", data)
        data = pd.DataFrame(data).T
        pred = self.get_pred(data, len(self.class_counts), 1)
        print("Predicted move:", pred)
        return pred
        
