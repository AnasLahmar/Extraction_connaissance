from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
# import category encoders
import category_encoders as ce

#======>ENCODER

def encoder(df): # cette fonction encode notre dataset et renvoie X_train,X_test,y_train,y_test
    X = df.drop([df.columns[-1]], axis=1)
    y = df[df.columns[-1]]
    # split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
    # encode variables with ordinal encoding
    col=[]
    for i in range(len(df.columns)-1):
        col.append(df.columns[i])
    encoder = ce.OrdinalEncoder(cols=col)
    x_train = encoder.fit_transform(X_train)
    x_test = encoder.transform(X_test)
    return x_train,x_test,y_train,y_test,X_train

#========>ENTRAINER
def entrainer(x_train,y_train,index):
    # Entrainement du modele
    model= DecisionTreeClassifier(criterion=index, max_depth=3, random_state=0)
    model.fit(x_train, y_train)
    return model

#========>PLOT
def plot_tree(df,model):
    tree.plot_tree(model,
                feature_names = df.columns[0:-1], 
                class_names=df[df.columns[-1]].unique(),
                filled = True)

#=========>Value_ecoded
def value_encoded(x_train,X_train):
     for i in range(x_train.shape[1]):
          val_not_enoded=X_train[X_train.columns[i]].unique()
          val_encoded=np.sort(x_train[x_train.columns[i]].unique())
          my_dict = {val_not_enoded[j] : val_encoded[j] for j in range(len(val_not_enoded))} 
          st.info("Pour l'attribut {} on a lui encodé de cette façon: {} ".format(x_train.columns[i],my_dict))
        
#==========>Predicted Value
def prediction_tree_viz(model,x_train,X_train):
    ## Prediction Value===================================================================
    df=pd.read_csv('data.csv')
    param=[]
    for i in range(len(df.columns)-1):
        if type(df.columns[i])==str:
            param.append(str(st.selectbox(df.columns[i],df[df.columns[i]].drop_duplicates())))
        else:
            param.append(st.number_input(df.columns[i]))
    #=======================================encoder param==============================
    l=[]
    a=[]
    for i in range(x_train.shape[1]):
        val_not_enoded=X_train[X_train.columns[i]].unique()
        val_encoded=np.sort(x_train[x_train.columns[i]].unique())
        for j in range(len(val_not_enoded)):
            if val_not_enoded[j]==str:
                l.append([val_not_enoded[j],val_encoded[j]])
            else:
                l.append([str(val_not_enoded[j]),val_encoded[j]])


    for i in range(len(param)):
        for j in range(len(l)):
            if param[i] in l[j]:
                a.append(l[j][1])
                break

    t=st.button("Predict")
    if t:            
        data_ = {df.columns[j] : [a[j]] for j in range(len(a))} 
        # Create DataFrame  
        data_= pd.DataFrame(data_)  
        y_pred= model.predict(data_)
        st.info("La décision prise est :")
        st.success(y_pred)
