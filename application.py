import streamlit as st
import seaborn as sns
import json
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# visualize confusion matrix with seaborn heatmap
import seaborn as sns
from sklearn.metrics import confusion_matrix
def confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                    index=['Predict Positive:1', 'Predict Negative:0'])
    fig=plt.figure(figsize=(4,4))
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')        
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig)
# Interface
st.title("Extraction de connaissances")

selected=option_menu(
    menu_title="Main Menu",
    options=["Home","Data Overview","Decision Algorithms"],
    icons=["house","bar-chart"],
    menu_icon="cast",  # optional
    default_index=0,
    orientation="horizontal",  
    styles={
        "nav-link-selected": {"background-color": "#4B9DFF"},
    } 

     )
   
#==================================================================================================================

    # Accueil
if selected=="Home":

    st.write("Cette application permet de faire une visualisation de votre jeu de données et d'appliquer les différents algorithmes de classification comme Logistic Regression, SVM, Random Forest et KNN, puis visualiser leurs performances.")
    # creer une animation
    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)
    lottie_coding = load_lottiefile("loteri.json")  # replace link to local lottie file
    st_lottie(
    lottie_coding,
    speed=1,
    reverse=False,
    loop=True,
    quality="low", # medium ; high

    height=None,
    width=None,
    key=None,
)

if selected=="Data Overview":
    with st.sidebar:
    # creer une animation
    # Creer un slider
        def load_lottiefile(filepath: str):
                with open(filepath, "r") as f:
                    return json.load(f)
        lottie_coding = load_lottiefile("lottie2.json")  # replace link to local lottie file
        st_lottie(lottie_coding,speed=1,reverse=False,loop=True,quality="high",height=None, width=None, key=None,)
    # Chose csv file
    st.sidebar.title("Select Your Dataset")
    upload_file=st.sidebar.file_uploader("Select:",type=["csv"])
    if upload_file is not None:
        data=pd.read_csv(upload_file)
        data.to_csv('data.csv', index=False)
        st.success("Dataset has selected successfully")
        st.info("This part is useful when we use the data that contains a variable value and it helps to discover it and gives us a general information ")
        if st.checkbox("Discover your Data") :
            st.write(""" ## Discover your Data :""")
            radiodicover=st.radio("",("Shape","Columns Name","Description","Missing Value"))
            if radiodicover=="Shape":
                st.write(""" ### Results : """)
                st.success(data.shape)
            if radiodicover=="Columns Name":
                st.write(""" ### Results : """)
                st.success(data.columns)
                st.write(data.info)
            if radiodicover=="Description":
                st.write(""" ### Results : """)
                st.write(data.describe())
            if radiodicover=="Missing Value":
                st.write(""" ### Results : """)
                st.write(data.isnull().sum())
        if st.checkbox("Data Visualisation"):
            radio_vis=st.radio("Choose :",("Heat Map","Plot"))
            if radio_vis=="Heat Map":
                data=pd.read_csv("data.csv")
                fig, ax = plt.subplots()
                heatmap = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)
                heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
                st.pyplot(fig)
            if radio_vis=="Plot":
                data=pd.read_csv("data.csv")
                df=data.select_dtypes(include='number')
                st.write(data.shape)
                st.write("""## Visualize the relation between data variables """)
                st.write("""### X_features """)
                x=st.selectbox("Variables !",df.columns)
                st.write("""### Y_features """)
                y=st.selectbox("Target !",df.columns)
                fig, ax = plt.subplots()
                ax.scatter(df[x],df[y],c='b')
                plt.xlabel(x)
                plt.ylabel(y)
                st.pyplot(fig)
        
    else:
        st.info("Select your Dataset")
        
if selected=="Decision Algorithms":
    with st.sidebar:
    # creer une animation
    # Creer un slider
        def load_lottiefile(filepath: str):
                with open(filepath, "r") as f:
                    return json.load(f)
        lottie_coding = load_lottiefile("lottie2.json")  # replace link to local lottie file
        st_lottie(lottie_coding,speed=1,reverse=False,loop=True,quality="high",height=None, width=None, key=None,)
    # Chose csv file
    st.sidebar.title("Select Your Dataset")
    upload_file=st.sidebar.file_uploader("Select:",type=["csv"])
    if upload_file is not None:
        data=pd.read_csv(upload_file)
        data.to_csv('data.csv', index=False)
        ##### SPLITTING
        X=data.iloc[:,0:-1]
        y=data.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        #####
        st.success("Dataset has selected successfully")
         # define Button style  
        algorithm=st.selectbox("Choose an algorithm",("Select an Algorithm","Logistic Regression","SVM","Random Forest","KNN"))
        if algorithm=="Logistic Regression":
            st.write(''' ### Adjust some parameters of Logistic Regression model ''')
            st.info("The parameter C: Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive and smaller value specify stronger regularization.")
            c = st.slider("C : ", 1, 100, 20)
            st.info("penalty :Specify the norm of the penalty")
            pl=st.selectbox("Penalty : ",('none', "l2"))
            m = st.markdown("""
            <style>
            div.stButton > button:first-child {
                background-color: #0099ff;
                color:#ffffff;
            }
            div.stButton > button:hover {
                background-color: #00ff00;
                color:#ff0000;
                }
            </style>""", unsafe_allow_html=True)
            button2=st.button("Appliquer le modèle")
            if button2:
                # instantiate classifier with default hyperparameters
                model=LogisticRegression(penalty=str(pl),C=c)
                # fit classifier to training set
                model.fit(X_train,y_train)
                # make predictions on test set
                y_pred=model.predict(X_test)
                # compute and print accuracy score
                st.success('Model accuracy score is: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

                st.info("Show Confusion Matrix")
                confusion(y_test, y_pred)
        if algorithm=="SVM":
            # instantiate classifier with default hyperparameters
            st.write(''' ### Adjust some parameters of SVM model ''')
            st.info("The parameter C: Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive and smaller value specify stronger regularization.")
            C = st.slider("C : ", 1, 100, 25)
            st.info("Specifies the kernel type to be used in the algorithm.")
            kernel=st.selectbox("kernel",('linear', 'poly', 'rbf', 'sigmoid'))
            m = st.markdown("""
            <style>
            div.stButton > button:first-child {
                background-color: #0099ff;
                color:#ffffff;
            }
            div.stButton > button:hover {
                background-color: #00ff00;
                color:#ff0000;
                }
            </style>""", unsafe_allow_html=True)
            button2=st.button("Appliquer le modèle")
            if button2:
                model=SVC(kernel=str(kernel),C=C)
                # fit classifier to training set
                model.fit(X_train,y_train)
                # make predictions on test set
                y_pred=model.predict(X_test)
                # compute and print accuracy score
                st.success('Model accuracy score is: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
                st.info("Show Confusion Matrix")
                confusion(y_test, y_pred)
        if algorithm=="Random Forest":
            st.write(''' ### Adjust some parameters of Random Forest model ''')
            st.info("n_estimators: The number of trees in the forest.")
            n_estimators= st.slider("n_estimators : ", 1, 200, 100)
            st.info("criterion: The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain")
            criterion=st.selectbox("criterion: ",('gini', 'entropy', 'rbf', 'log_loss'))
            m = st.markdown("""
            <style>
            div.stButton > button:first-child {
                background-color: #0099ff;
                color:#ffffff;
            }
            div.stButton > button:hover {
                background-color: #00ff00;
                color:#ff0000;
                }
            </style>""", unsafe_allow_html=True)
            button2=st.button("Appliquer le modèle")
            if button2:
                # instantiate classifier with default hyperparameters
                model=RandomForestClassifier(n_estimators=n_estimators,criterion=str(criterion))
                # fit classifier to training set
                model.fit(X_train,y_train)
                # make predictions on test set
                y_pred=model.predict(X_test)
                # compute and print accuracy score
                st.success('Model accuracy score is: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
                st.info("Show Confusion Matrix")
                confusion(y_test, y_pred)
        if algorithm=="KNN":
            st.write(''' ### Adjust some parameters of KNN model ''')
            st.info("n_neighbors: Number of neighbors to use by default for kneighbors queries.")
            K= st.slider("n_neighbors: ", 1, 10, 5)
            # instantiate classifier with default hyperparameters
            #=============================================================================   
            m = st.markdown("""
            <style>
            div.stButton > button:first-child {
                background-color: #0099ff;
                color:#ffffff;
            }
            div.stButton > button:hover {
                background-color: #00ff00;
                color:#ff0000;
                }
            </style>""", unsafe_allow_html=True)         
            button2=st.button("Appliquer le modèle")
            if button2:
                model=KNeighborsClassifier(n_neighbors=K)
                # fit classifier to training set
                model.fit(X_train,y_train)
                # make predictions on test set
                y_pred=model.predict(X_test)
                # compute and print accuracy score
                st.success('Model accuracy score is: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
                st.info("Show Confusion Matrix")
                confusion(y_test, y_pred)


    else:
        st.info("Select your Dataset that contains NUMERICAL VALUE or ENCODED Variables")
        
        

        #=====================================================================================================
       