# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 21:06:24 2019

@author: Ocio
"""

# Natural Language Processing

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting = 3)

# Limpieza de texto
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000): 
    review = re.sub("[^a-zA-Z]", " ", dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    review = " ".join(review)
    corpus.append(review)

# Crear el Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Crear diccionario para alimentarlo al modelo de Máxima Entropía
feature_set = []
for i in range(0,1000):
    row_dict = {}
    for j in range(0,1500):
        if X[i, j] > 0:
            row_dict[j] = X[i, j]
    feature_set.append((row_dict, y[i]))
            
# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
feature_set_train, feature_set_test = train_test_split(feature_set, test_size = 0.20, random_state = 0)

# Ajustar el clasificador de Máxima Entropía en el Conjunto de Entrenamiento

from nltk.classify import MaxentClassifier
classifier = MaxentClassifier.train(feature_set_train, algorithm='iis', trace=0, max_iter=30, min_lldelta=0.5)

# Predicción de los resultados con el Conjunto de Testing
y_pred = []
for k in range(0,200):
    result  = classifier.classify(feature_set_test[k][0])
    y_pred.append(result)
y_test = [feature_set_test[m][1] for m in range(0,200)]
# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Calcular métricas
TN = cm[0, 0]
FN = cm[1, 0]
TP = cm[1, 1]
FP = cm[0, 1]
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
specificity = TN / (TN + FP)
F1_score = 2 * precision * recall / (precision + recall)
