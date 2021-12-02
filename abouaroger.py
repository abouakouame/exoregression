#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression


# In[6]:


os.getcwd()


# In[11]:


#1- importation du jeu de Données
= pd.read_excel(r'C:\Users\HPPC\Downloads\null.xlsx')
print(data)


# In[12]:


# 2-copy du dataset
df = data.copy()
df


# In[25]:


#3-verification si des doublons si une ligne se reperte
df.duplicated()


# In[26]:


# Compter les doublons
df.duplicated().value_counts()


# 

# In[14]:


#4-verifier si les données manquantes 
df.isna().value_counts()


# In[29]:


df.nunique()


# valeur_qualitative =  df.select_dtypes('object').columns
# valeur_qualitative

# In[15]:


#supprimer une donnée manquant
df.dropna()


# In[24]:


#savoir comment supprimer les doublons
df.drop_duplicates(keep=False)


# In[17]:


#verifier si pas de constante (les variables qui n'ont pas de variables de modalité) supprimer
df.nunique()


# In[21]:


valeur_qualitative =  df.select_dtypes('object').columns
valeur_qualitative


# In[30]:


df['SEXE'].astype('category')


# In[31]:


df['SEXE'].astype('category').cat.codes


# In[33]:


#normaliser les variables quantitatives chaque valeur/ valeurs max
def recoder(df):
    for i in df.select_dtypes('object').columns:
        df[i]=df[i].astype('category').cat.codes
    return(df)
recoder(df)


# In[34]:


#creer un dataframe x qui va contenir de age a pente et y coeur
x = df.iloc[:,:11]
x


# In[35]:


y = data.iloc[:,11:]
y


# In[36]:


# repatition de nos données en données d'entrainement train et de test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)


# In[ ]:


#(regression logistique avec lineaire_model)

#instanciation du modèle
modele_regLog = linear_model.LogisticRegression(random_state=0, solver = 'liblinear', multi_class = 'auto')

#training
modele_regLog.fit(x_train, y_train)

#prediction
prediction = modele_regLog.predict(x_test)
print('prédiction du modèle:', prediction)

#calcule de précision
precision = modele_regLog.score(x_test, prediction)
print('la précision du modèle', precision * 100)


# In[45]:


#(regression logistique avec lineaire_model)

#instanciation du modèle
modele_regLog = linear_model.LogisticRegression(random_state=0, solver = 'liblinear', multi_class = 'auto')

#training
modele_regLog.fit(x_train, y_train)

#prediction
prediction = modele_regLog.predict(x_test)
print('prédiction du modèle:', prediction)

#calcule de précision
precision = modele_regLog.score(x_test, prediction)
print('la précision du modèle', precision * 100)


# In[ ]:





# In[ ]:




