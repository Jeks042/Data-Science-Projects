#!/usr/bin/env python
# coding: utf-8

# # CREDIT CARD DEFAULT PREDICTION USING MACHINE LEARNING MODELS 
# ## B1190869 - Ezema, Chukwujekwu Joseph 
# 
# ### Msc Data Science
# ### School of Computing, Enigineering and Digital Technology (SCEDT)
# ### Teesside University Middlesbrough, UK
# #### 20th May 2022

# # Introduction
# 
# ![image.png](attachment:image.png)

# # 1. Get Dataset

# ## 1.1 Import and Install Libraries

# In[1]:


# 1.1 Import and Install Libraries
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# display all columns of the dataset
pd.pandas.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')

#Confirmation
print('Libraries successfully installed')


# In[2]:


""" Step 1: Get dataset """
# 1.2 Load dataset

# Reading the Application File - dataset for the features
apps = pd.read_csv('application_record.csv')

apps_copy = apps.copy() #making a copy
apps_copy


# In[3]:


# Reading the Credit File - dataset for the target class
credit = pd.read_csv('credit_record.csv')

credit_copy = credit.copy() #making a copy
credit_copy


# # 2. Exploratory Data Analysis and Feature Engineering

# In[4]:


""" Step 2: Summarise the data """
# 2.1. Explore the features dataset

apps.info()


# In[5]:


# 2.2 Check features with missing value

apps.isnull().sum()


# In[6]:


# 2.2.1 Check percentage of missing values

print(f'Percentage of missing values: = {round(apps["OCCUPATION_TYPE"].isnull().sum()/len(apps) * 100,2)}%')


# In[7]:


# 2.2.2 Fill missing cells with "Unknown" => This is because missing % is high enough

apps['OCCUPATION_TYPE'].fillna("Unknown", inplace = True)

apps['OCCUPATION_TYPE'].value_counts()


# In[8]:


# 2.3 Check unique ID:

print("Number of unique IDs: {}".format(len(apps.ID.unique())))


# In[9]:


# 2.4 Check duplicated IDs:

print("Number of duplicated records before dropping:",apps.ID.duplicated().sum())


# In[10]:


# 2.4.1 Drop duplicated IDs => remember to put in a new variable when executing
df1 = apps.drop_duplicates(subset=['ID'], keep='last')

# Confirming non duplicated
print("Number of duplicated records after dropping: {}".format(df1.ID.duplicated().sum()))


# In[11]:


# 2.5 Convert the categorical data to integer 1s and 0s values
def to_integers(self):

    # Converting the univariate categorical data to binary values for best results
    self.replace({'FLAG_OWN_CAR' : {'Y' : 1,  'N' : 0}}, inplace=True)
    self.replace({'FLAG_OWN_REALTY' : {'Y' : 1,  'N' : 0}},inplace=True)

    # Converting CNT_FAM_MEMBERS to whole number
    self['CNT_FAM_MEMBERS'] = self['CNT_FAM_MEMBERS'].astype(np.int64)

    # Converting the days to real age using pandas timedelta format
    self['AGE'] = np.ceil(pd.to_timedelta(self['DAYS_BIRTH'], unit='D').dt.days / -365.25).astype(np.int64)

    # values greater than zero means that the applicant doesn't work
    self.loc[(self['DAYS_EMPLOYED'] > 0), 'DAYS_EMPLOYED'] = 0

    # Converting the days of employment to total years using pandas timedelta format
    self['YEARS_EMPLOYED'] = np.ceil(pd.to_timedelta(self['DAYS_EMPLOYED'], unit='D').dt.days / -365.25).astype(np.int64)

    # converting categorical feature to 1s and 0s to enhance feature selection model
    self["Has_Partner"] = self["NAME_FAMILY_STATUS"].replace(["Civil marriage","Married","Single / not married",
                                                                              "Separated","Widow"],[1,1,0,0,0])

    # custom column creation for Household_Size: this adds 1 to count of children for only those with partners
    self["Household_Size"] = self["CNT_CHILDREN"] + self["Has_Partner"].apply(lambda x: 2 if x==1 else 1)

    return self

to_integers(df1)


# In[12]:


# 2.6 label encoding
def to_encoding(self):
    # housing type
    housing_type = {'House / apartment' : 'apartment',
                       'With parents': 'with_parents',
                        'Municipal apartment' : 'apartment',
                        'Rented apartment': 'apartment',
                        'Office apartment': 'apartment',
                        'Co-op apartment': 'apartment'}

    self['NAME_HOUSING_TYPE'] = self['NAME_HOUSING_TYPE'].map(housing_type)

    # family status
    family_status = {'Single / not married':'Single',
                         'Separated':'Single',
                         'Widow':'Single',
                         'Civil marriage':'Married',
                        'Married':'Married'}

    self['NAME_FAMILY_STATUS'] = self['NAME_FAMILY_STATUS'].map(family_status)

    # education type
    education_type = {'Secondary / secondary special':'Secondary',
                         'Lower secondary':'Secondary',
                         'Higher education':'Tertiary',
                         'Incomplete higher':'Tertiary',
                         'Academic degree':'Tertiary'}

    self['NAME_EDUCATION_TYPE'] = self['NAME_EDUCATION_TYPE'].map(education_type)

    # occupation type
    occupation_type = { 'Laborers'   :'unskilled',
                        'Sales staff': 'skilled',
                        'Core staff' : 'skilled',
                        'Managers'   : 'skilled',
                        'Drivers'    : 'unskilled',
                        'High skill tech staff' : 'skilled',
                        'Accountants'           : 'skilled',
                        'Medicine staff'        : 'skilled',
                        'Cooking staff'         : 'unskilled',
                        'Security staff'        : 'unskilled',
                        'Cleaning staff'        : 'unskilled',
                        'Private service staff' : 'unskilled',
                        'Low-skill Laborers'    : 'unskilled',
                        'Secretaries'           : 'skilled',
                        'Waiters/barmen staff'  : 'unskilled',
                        'Realty agents'         : 'skilled',
                        'HR staff'              : 'skilled',
                        'IT staff'              : 'skilled',
                        'Unknown'              : 'unknown'}

    self['OCCUPATION_TYPE'] = self['OCCUPATION_TYPE'].map(occupation_type)

    # income type
    income_type = {'Commercial associate':'Working',
                      'State servant':'Working',
                      'Working':'Working',
                      'Pensioner':'Pensioner',
                      'Student':'Student'}

    self['NAME_INCOME_TYPE'] = self['NAME_INCOME_TYPE'].map(income_type)

    #Dropping unused columns
    self.drop(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED', 'CNT_CHILDREN',"Has_Partner"], inplace =True)

    return self

to_encoding(df1)


# In[13]:


# 2.7 covert to dummy features
def hot_encoding(self):
    # Renaming long-labelled columns for better tags
    self.columns = ['ID', 'Gender', 'Car', 'Realty', 'Income', 'IncomeType',                    'Education', 'Marital', 'Rental', 'MobilePhone', 'WorkPhone',                     'OtherPhone', 'Email', 'Job', 'FamilySize', 'Age', 'Experience', 
                    'Household']

    # Setting the dummy features (one-hot encoding) to enhance feature selection
    df = pd.get_dummies(self, columns=['Gender', 'IncomeType', 'Education','Marital',"Rental", 'Job'])
    
    return df

df2 = hot_encoding(df1)
df2


# In[14]:


# 2.8 Structure the Target Class according unique IDs from the credit dataset

# defining a function to categorise Target Class from credit dataset - "Default (1)" or "Not_Default(0)" on a new column, Target
def transform_target(self):
    '''
    for every unique ID, if status is "X" or "0" or "C", 
    let the new column say 0, otherwise say 1
    '''
    self["Target"] = [0 if Target in ["X", "0", "C"] else 1 for Target in self["STATUS"]]
    df = self[['ID', 'Target']].groupby("ID").Target.agg(lambda x : x.mode()[0]).reset_index() # aggregates the Target Class by highest 1s and 0s for each ID

    return df

df3 = transform_target(credit)
df3


# In[15]:


# 2.9 merge the target class to the data table
credit_app = pd.merge(df2, df3, on ='ID', how='inner')

# confirm for any duplicated ID
print("Number of duplicated IDs:",credit_app.ID.duplicated().sum())

credit_app


# In[16]:


# 2.10 drop ID Key
credit_app.drop(columns = 'ID', inplace = True)

# Target Distribution
print("Shape after transforming to Target Class:", credit_app.shape)
print(f'Number of Non-Default Class = {credit_app["Target"].value_counts()[0]}')
print(f'Number of Default Class = {credit_app["Target"].value_counts()[1]}')
print(f'Percent of Non-Default Class = {round(credit_app["Target"].value_counts()[0]/len(credit_app) * 100,2)}%')
print(f'Percent of Default Class = {round(credit_app["Target"].value_counts()[1]/len(credit_app) * 100,2)}%')

sns.countplot('Target', data=credit_app, palette="Set2")
plt.title('Non-Default Vs Default Customers', fontsize=14)
plt.show()


# In[17]:


# 2.11 Data Visualization Analysis
#Customizing the seaborn chart designs

sns.set_context("notebook",font_scale=.8,rc={"grid.linewidth": 0.1,'patch.linewidth': 0.0,
    "axes.grid":True,
    "grid.linestyle": "-",
    "axes.titlesize" : 15,                                       
    "figure.autolayout":True})
                
color = '#FF5E5B'
palette2 = 'pastel'
palette3 = 'inferno'
palette4 = 'Set2'


# 2.11.1 Plot the numerical distribution
plt.figure(figsize=(10,10))

col_plot = ["Household","Income","Age","Experience"]
credit_app[col_plot].hist(edgecolor='black', linewidth=1.2, color=color)
fig=plt.gcf()
fig.set_size_inches(12,6)


# In[18]:


# 2.11.2 visualize the outliers in respect to the target

#income
fig, (ax1, ax2,) = plt.subplots(ncols=2, figsize=(12,6))
sns.boxplot(ax = ax1, x="Target", y="Income", hue="Target",data=credit_app, palette=palette4,showfliers=True)
sns.boxplot(ax = ax2, x="Target", y="Income", hue="Target",data=credit_app, palette=palette4,showfliers=False)

plt.suptitle('Target Distribution of Income', fontsize=14)
plt.show()


# In[19]:


#household
fig, (ax1, ax2,) = plt.subplots(ncols=2, figsize=(12,6))
sns.boxplot(ax = ax1, x="Target", y="Household", hue="Target",data=credit_app, palette=palette2,showfliers=True)
sns.boxplot(ax = ax2, x="Target", y="Household", hue="Target",data=credit_app, palette=palette2,showfliers=False)

plt.suptitle('Target Distribution of Household', fontsize=14)
plt.show()


# In[20]:


# 2.11.3 dummy feature distribution

#job type
fig, axes = plt.subplots(1,3)

Skilled= credit_app['Job_skilled'].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',shadow=True, colors=["#76B5B3","#EC9B9A"], textprops = {'fontsize':12}, ax=axes[0])
Skilled.set_title("Skilled Workers")

Unskilled= credit_app['Job_unskilled'].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',shadow=True,colors=["#80DE99","#00CECB"],textprops = {'fontsize':12}, ax=axes[1])
Unskilled.set_title("Unskilled Workers")

Unknown= credit_app['Job_unknown'].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',shadow=True,colors=['#FFED66','#FF5E5B'],textprops = {'fontsize':12}, ax=axes[2])
Unknown.set_title("Unknown Workers")

fig.set_size_inches(14,5)
plt.tight_layout()
plt.show()


# In[21]:


#earning type
fig, axes = plt.subplots(1,3)

Working= credit_app['IncomeType_Working'].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',shadow=True, colors=["#EC9B9A", "#76B5B3"], textprops = {'fontsize':12}, ax=axes[0])
Working.set_title("Working")

Pensioner= credit_app['IncomeType_Pensioner'].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',shadow=True,colors=["#00CECB", "#80DE99"],textprops = {'fontsize':12}, ax=axes[1])
Pensioner.set_title("Pensioner")

Student= credit_app['IncomeType_Student'].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',shadow=True,colors=['#FF5E5B', '#FFED66'],textprops = {'fontsize':12}, ax=axes[2])
Student.set_title("Student")

fig.set_size_inches(14,5)
plt.tight_layout()
plt.show()


# In[22]:


# 2.12 Checking statistics of the numerical features
num_col = ["Household","Income","Age","Experience"]
credit_app[num_col].describe().T


# In[23]:


# 2.13 Correlation analysis
def corr_mat(self):
    colormap = plt.cm.Reds
    plt.figure(figsize=(12,10))
    sns.heatmap(self.corr(),linewidths=0.1,vmax=0.8, 
                square=True, cmap = colormap, linecolor='white')
    plt.title('Correlation matrix', fontsize=14)
    plt.show()
    
corr_mat(credit_app)


# In[24]:


# 2.14 Target Distribution across all features
def density(self):
    num_col = ["Household","Income","Age","Experience"]
    var = self[num_col].columns.values 

    i = 0
    t0 = self.loc[self['Target'] == 0]
    t1 = self.loc[self['Target'] == 1]

    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(1,4,figsize=(15,5))

    for feature in var:
        i += 1
        plt.subplot(1,4,i)
        sns.distplot(t0[feature], label="Target = 0")
        sns.distplot(t1[feature], label="Target = 1")
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()
    
density(credit_app)


# # 3. Feature Selection and Data Preparation

# In[25]:


# Import Libraries
import time
import plotly.express as px
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler  # avoid the outliner effect

import warnings
warnings.filterwarnings('ignore')


# In[26]:


""" Step 3: Prepare and select best features for the training"""
# 3.1 Scale the numeric columns using Standard Scaler

# copy of datasets
df = credit_app.copy()
for i in num_col:
    
    # fit on training data column
    scale = StandardScaler().fit(df[[i]])
    
    # transform the training data column
    df[i] = scale.transform(df[[i]])
    
df


# In[27]:


# 3.2. Remove unused column (FamilySize, IncomeType_Student, MobilePhone)
#FamilySize is same as the Household
#From the Correlation Analysis, MobilePhone has no correlation to the features
#From the data visualization, No student's data was in the dataset after joining with the Target

df.drop(columns=['FamilySize', 'IncomeType_Student', 'MobilePhone'], inplace=True) 
df


# In[28]:


# 3.3. Get sample data
# To reduce training time, resample data
n_sample = 5000
random_state = 42

non_default = df[df['Target'] == 0].sample(n_sample, random_state=random_state) #resample only from the majority class
default =  df[df['Target'] == 1]

# Merge 2 subset
sample = non_default.append(default).sample(frac=1, random_state=random_state).reset_index(drop=True)
y = sample["Target"].values

print("After resampling, Number of Default Instances : {}".format(sum(y==1)))
print("After resampling, Number of Non-Default Instances : {}".format(sum(y==0)))

sample


# In[29]:


# 3.4 Sample visualisation
#correlation
corr_mat(sample)

#density
density(sample)


# In[30]:


# 3.5 Feature Counts
#Independent and Dependent Variable
X = sample.drop(['Target'], axis = 1)
y = sample["Target"].values

# Train_Test Split samples for original dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, shuffle=True, stratify=y)
print(f"For the resample data of {n_sample} non-default instances:")
print("Shape of X_train dataset: ", X_train.shape)
print("Size of y_train dataset: ", y_train.shape[0])
print("Shape of X_test dataset: ", X_test.shape)
print("Size of y_test dataset: ", y_test.shape[0])
print("----------------------------------------------------")
print("Number of Default Instances in train set : {}".format(sum(y_train==1)))
print("Number of Non-Default Instances in train set : {}".format(sum(y_train==0)))
print("Number of Default Instances in test set : {}".format(sum(y_test==1)))
print("Number of Non-Default Instances in test set : {}".format(sum(y_test==0)))


# In[31]:


# 3.6 Train_Test Split samples for SMOTE oversampling
# Using SMOTE as oversampling Technique for handling imbalanced dataset

sm = SMOTE(sampling_strategy='minority', random_state=random_state)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print('The number of target class before oversampling: {}'.format(y_train.shape[0]))
print('The number of target class after oversampling: {}'.format(y_train_sm.shape[0]))
print("----------------------------------------------------")
print("\nAfter OverSampling, counts of default: {}".format(sum(y_train_sm==1)))
print("After OverSampling, counts of non-default: {}".format(sum(y_train_sm==0)))


# In[32]:


#3.7 Dimensionality reduction strategy for feature selection
def apply_PCA(X_train, X_test, COMPONENTS=10):

    # Tranform X train, X test
    pca = PCA(n_components=COMPONENTS, random_state=random_state).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("X_train_pca.shape: {}".format(X_train_pca.shape))

    return [X_train_pca, X_test_pca]


# # 4. Modelling

# In[33]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import tensorflow as tf 
from tensorflow import keras

from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score,roc_auc_score,f1_score,plot_confusion_matrix,plot_roc_curve,roc_curve

import warnings
warnings.filterwarnings('ignore')


# In[34]:


""" Step 4: Model the Algorithms for Prediction"""

# 4.1 Design ML model - Set up
# Design ML

classifiers = {
            "LogisiticRegression": LogisticRegression(random_state=random_state),
            "KNearest": KNeighborsClassifier(),
            "GradientBoost": GradientBoostingClassifier(random_state=random_state),
            "Random Forest Classifier": RandomForestClassifier(random_state=random_state),
            "XGBClassifier": XGBClassifier(random_state=random_state)
}


# In[35]:


# 4.2 Cross Validation - Set up
# For reference, before applying Gridsearch CV to find best parameter
def cross_validate(X_train, y_train, cv=5):
    for key, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        training_score = cross_val_score(classifier, X_train, y_train, cv=cv)
        print("Classifiers: ", classifier.__class__.__name__, 
              "has a training accuracy score of", round(training_score.mean(),2) * 100, "%")


# In[36]:


# 4.3 Best Parameter - Set up
# Use GridSearchCV to find the best parameters suitable for each model:
def model_best_estimator(X_train, y_train, class_weight=None, random_state=random_state, cv=5):
    
    # Logistic Regression 
    t0 = time.time()
    LR_params_grid = {"solver": ["liblinear", "sag", "lbfgs"], "penalty":['l2'],
                       'C': [0.01, 0.1, 1, 100]}

    grid_LR = GridSearchCV(LogisticRegression(random_state=random_state, class_weight=class_weight, max_iter=10000),
                                LR_params_grid, cv=cv, n_jobs=4)
    grid_LR.fit(X_train, y_train)

    # get the logistic regression with the best parameters.
    LR = grid_LR.best_estimator_
    t1 = time.time()

    print("Best fit parameter for Logistic Regression", LR)
    print("Elapsed time {:.2f} s".format(t1 - t0))

    
    # KNN
    t2 = time.time()
    KNN_params_grid = {"n_neighbors": list(range(2,8,1)), 
                          "metric": ('minkowski', 'euclidean', 'manhattan')}
    
    grid_KNN = GridSearchCV(KNeighborsClassifier(), KNN_params_grid, cv=cv)
    grid_KNN.fit(X_train, y_train)
   
    # KNN best estimator
    KNN = grid_KNN.best_estimator_
    t3 = time.time()
    print("\nBest fit parameter for KNN", KNN)
    print("Effective metric:", KNN.effective_metric_)
    print("Elapsed time {:.2f} s".format(t3 - t2))
    
    
    # GradientBoost Classifier:
    t4 = time.time()
    GB_params_grid = {"max_depth": list(range(2,6,1)),
                "min_samples_leaf": list(range(2,7,1))}
    
    grid_GB = GridSearchCV(GradientBoostingClassifier(random_state=random_state),
                             GB_params_grid, cv=cv)
    grid_GB.fit(X_train, y_train)
    
    # gboost best estimator
    GB = grid_GB.best_estimator_
    t5 = time.time()
    
    print("\nBest fit parameter for Gradient Boost:", GB)
    print("Elapsed time {:.2f} s".format(t5 - t4))

    # Random Forest Classifier
    t6 = time.time()
    RF_params_grid = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,6,1)),
                "min_samples_leaf": list(range(2,7,1))}

    grid_RF = GridSearchCV(RandomForestClassifier(random_state=random_state, class_weight=class_weight), 
                           RF_params_grid, cv=cv)
    grid_RF.fit(X_train, y_train)

    # random forest best estimator
    RF = grid_RF.best_estimator_
    t7 = time.time()

    print("\nBest fit parameter for Random Forest:", RF)
    print("Elapsed time {:.2f} s".format(t7 - t6))
    
    # XGBoost Classifier
    t8 = time.time()
    XGB_params_grid = {
        "max_depth": list(range(2, 6, 1)),
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [100, 200]
    }

    grid_XGB = GridSearchCV(
        XGBClassifier(random_state=random_state),
        XGB_params_grid,
        cv=cv
    )
    grid_XGB.fit(X_train, y_train)

    # random forest best estimator
    XGB = grid_XGB.best_estimator_
    t9 = time.time()

    print("\nBest fit parameter for XGBoost:", XGB)
    print("Elapsed time {:.2f} s".format(t9 - t8))
    
    return [LR, KNN, GB, RF, XGB]   


# In[37]:


# 4.4 Evaluate model by using cross validation - setup
def evaluate_model(classifier, X_train, y_train, cv=5):
    classifier.fit(X_train, y_train)
    score = cross_val_score(classifier, X_train, y_train, cv=cv)
    return score


# In[38]:


# 4.5 Get training model results - setup
def train_model(classifier, X_train, y_train, cv=5):
    y_train_pred = cross_val_predict(classifier, X_train, y_train, cv=cv)
    print(classification_report(y_train, y_train_pred, labels=[1,0])) 


# In[39]:


# 4.6 Get testing model results - setup
def predict_model(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred, labels=[1,0]))
    
    # Confusion Matrix
    print('Confusion matrix:', classifier)
    cf_matrix = confusion_matrix(y_test, y_pred, labels=[1,0])
    ax =sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Default', 'Not Default'],
                yticklabels=['Default', 'NOt Default'])
    ax.set(xlabel="Predicted outputs", ylabel = "Actual outputs")
    plt.show()


# In[40]:


# 4.7 Plot ROC - setup
def plot_result(LR, KNN, GB, RF, XGB, X_train, y_train, cv=5):
    # Get probability of y train predict:
    LR_pred = cross_val_predict(LR, X_train, y_train, cv=cv,
                             method="decision_function")
    KNN_pred = cross_val_predict(KNN, X_train, y_train, 
                                method='predict_proba', cv=cv)[:,1]
    GB_pred = cross_val_predict(GB, X_train, y_train, 
                                method='predict_proba', cv=cv)[:,1]
    RF_pred = cross_val_predict(RF, X_train, y_train, 
                                method='predict_proba', cv=cv)[:,1]
    XGB_pred = cross_val_predict(XGB, X_train, y_train, 
                                method='predict_proba', cv=cv)[:,1]
    
    # calculate fpr and tpr and threshold
    LR_fpr, LR_tpr, LR_thresold = roc_curve(y_train, LR_pred, pos_label=1)
    KNN_fpr, KNN_tpr, KNN_threshold = roc_curve(y_train, KNN_pred, pos_label=1)
    GB_fpr, GB_tpr, GB_threshold = roc_curve(y_train, GB_pred, pos_label=1)
    RF_fpr, RF_tpr, RF_threshold = roc_curve(y_train, RF_pred, pos_label=1)
    XGB_fpr, XGB_tpr, XGB_threshold = roc_curve(y_train, XGB_pred, pos_label=1)

    # Plot ROC
    
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(16,8))
    
    ax2.plot(LR_fpr, LR_tpr, 
             label='Logistic Regression Classifier Score: {:.3f}'.format(roc_auc_score(y_train, LR_pred, labels=[1,0])))
    ax2.plot(KNN_fpr, KNN_tpr, 
             label='KNears Neighbors Classifier Score: {:.3f}'.format(roc_auc_score(y_train, KNN_pred, labels=[1,0])))
    ax2.plot(GB_fpr, GB_tpr, 
             label='Gradient Boost Classifier Score: {:.3f}'.format(roc_auc_score(y_train, GB_pred, labels=[1,0])))
    ax2.plot(RF_fpr, RF_tpr, 
             label='Random Forest Classifier Score: {:.3f}'.format(roc_auc_score(y_train, RF_pred, labels=[1,0])))
    ax2.plot(XGB_fpr, XGB_tpr, 
             label='XGBoost Classifier Score: {:.3f}'.format(roc_auc_score(y_train, XGB_pred, labels=[1,0])))
    ax2.plot([0, 1], [0, 1], 'k--')
    #ax2.axis([-0.01, 1, 0, 1])
    ax2.set_xlabel('False Positive Rate', fontsize=16)
    ax2.set_ylabel('True Positive Rate', fontsize=16)
    ax2.set_title('ROC Curve', fontsize=18)
    ax2.legend(loc = 'best')
    
    
    # calc precision, recall and thresholds
    LR_precision, LR_recall, LR_thres_pr = precision_recall_curve(y_train, LR_pred, pos_label=1)
    KNN_precision, KNN_recall, KNN_thres_pr = precision_recall_curve(y_train, KNN_pred,  pos_label=1)
    GB_precision, GB_recall, GB_thres_pr = precision_recall_curve(y_train, GB_pred,  pos_label=1)
    RF_precision, RF_recall, RF_thres_pr = precision_recall_curve(y_train, RF_pred, pos_label=1)
    XGB_precision, XGB_recall, XGB_thres_pr = precision_recall_curve(y_train, XGB_pred, pos_label=1)
    
    # Plot precision-recall curve
    ax1.plot(LR_precision, LR_recall, 
             label="Logistic Regression Classifier avg precision: {:0.3f}".format(average_precision_score(y_train, LR_pred)))
    ax1.plot(KNN_precision, KNN_recall, 
             label='KNears Neighbors Classifier avg precision: {:.3f}'.format(average_precision_score(y_train, KNN_pred)))
    ax1.plot(GB_precision, GB_recall, 
             label='Gradient Boost Classifier avg precision: {:.3f}'.format(average_precision_score(y_train, GB_pred)))
    ax1.plot(RF_precision, RF_recall, 
             label='Random Forest Classifier avg precision: {:.3f}'.format(average_precision_score(y_train, RF_pred)))
    ax1.plot(XGB_precision, XGB_recall, 
             label='XGBoost Classifier avg precision: {:.3f}'.format(average_precision_score(y_train, XGB_pred)))
    ax1.set_xlabel('Precision', fontsize = 16)
    ax1.set_ylabel('Recall', fontsize = 16)
    #ax1.axis([-0.01, 1, 0, 1])
    ax1.set_title('Precision-Recall Curve', fontsize = 18)
    ax1.legend(loc = 'best')
   
    plt.show()


# In[41]:


# 4.8 Set up function for training and testing flow
# Start by finding the best parameter for ML model, train and get result + visualize the results

def train_test(X_train, y_train, X_test, y_test, random_state=random_state, class_weight=None, cv=5):
    
    # Find best parameter for model
    model_select_result = model_best_estimator(X_train, y_train, class_weight=class_weight)
    
    LR, KNN, GB, RF, XGB = model_select_result
    
    
    # Train and get result
    for classifier in model_select_result:
        print("\nPredict model:", classifier)
        evaluate_model(classifier, X_train, y_train, cv=cv)
        print("\nTraining result:")
        train_model(classifier, X_train, y_train, cv=cv)
        print("Testing result:")
        predict_model(classifier, X_test, y_test)
        
    
    # Plot result (ROC, Precision)
    print('Plot for training results')
    plot_result(LR, KNN, GB, RF, XGB, X_train, y_train)
    print('Plot for test results')
    plot_result(LR, KNN, GB, RF, XGB, X_test, y_test)
    
    
    #can add feature importance
    return [LR, KNN, GB, RF, XGB]


# In[42]:


## 4.9 Setup for Deep Learning Model
def ANN_model(X_train, y_train, X_test, y_test, epochs=5):
    
    #design model
    model = keras.Sequential([
        keras.layers.Dense(18, input_shape=(22,), activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    #compile model
    model.compile(optimizer = 'adam',
                 loss = 'binary_crossentropy',
                 metrics =['accuracy'])
    
    #fit model
    model.fit(X_train, y_train, epochs=epochs)
    
    # evaluate model
    print(f'model evaluation :', model.evaluate(X_test, y_test))
    
    # build prediction series
    yp = model.predict(X_test)
    y_pred =[]
    for element in yp:
        if element > 0.5:
            y_pred.append(1)
        else: 
            y_pred.append(0)
            
    mse = np.mean(np.power(X_test - yp, 2), axis=1)
    error = pd.DataFrame({'reconstruction_error': mse,
                            'true_class': y_test})    
    
    print(error.describe())
            
    #result
    print(classification_report(y_test, y_pred, labels=[1,0]))
    
    # Confusion Matrix
    print('Confusion matrix:', model)
    cf_matrix = confusion_matrix(y_test, y_pred, labels=[1,0])
    ax =sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Default', 'Not Default'],
                yticklabels=['Default', 'NOt Default'])
    ax.set(xlabel="Predicted outputs", ylabel = "Actual outputs")
    plt.show()

    # Plot ROC
    fpr, tpr, thres_roc = roc_curve(error.true_class, error.reconstruction_error)
    roc_auc = roc_auc_score(error.true_class, error.reconstruction_error, labels=[1,0])

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label='AUC = %0.3f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.001, 1])
    plt.ylim([0, 1.001])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
     # Plot Precision and recall
    precision, recall, thres_pr = precision_recall_curve(error.true_class, error.reconstruction_error,  pos_label=1)
    plt.plot(precision, recall, label= 'Avg precision = {:0.3f}'.format(average_precision_score(error.true_class, error.reconstruction_error)))
    plt.title('Precision-Recall Curve')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend(loc='best')
    plt.show()


# # 5. Train-Test Results

# In[43]:


# 5.1.1 Classification Models with Normal Dataset
### Experiment 1a: Baseline - Without doing any sampling or pca to resolve imbalance problem

print("\n************* Classification Normal Case - Baseline **************")

# Intially evaluate train model with 5 fold cross-validation
print("Initial evaluate training model")
cross_validate(X_train, y_train)

# Evaluate model and get result
LR, KNN, GB, RF, XGB = train_test(X_train, y_train, X_test, y_test)


# In[44]:


### Experiment 1b: PCA applied on Normal case
# 5.1.2 Apply pca for first n components to reduce dimensions
# and then train model with the new data
n = 10
X_train_pca, X_test_pca = apply_PCA(X_train, X_test, COMPONENTS=n)

print("\n************* PCA applied on Baseline case **************")
train_test(X_train_pca, y_train, X_test_pca, y_test)


# In[45]:


# 5.2.1 Classification Models with Smote Technique - oversampled Dataset
### Experiment 2a: Smote Technique - oversampling to resolve imbalance problem

print("\n************* Classification Oversampling Case - SMOTE **************")

# Intial evaluate train model with 5 fold cross-validation
print("Initial evaluate training model")
cross_validate(X_train, y_train)

# Evaluate model and get result
LR, KNN, GB, RF, XGB = train_test(X_train_sm, y_train_sm, X_test, y_test)


# In[46]:


# Experiment 2b: PCA on Oversampling
# 5.2.2 Apply pca for first n components 
n = 10
X_train_pca, X_test_pca = apply_PCA(X_train_sm, X_test, COMPONENTS=n)

# Evaluate model and get result
print("\n************* PCA applied on Oversampling **************")
train_test(X_train_pca, y_train_sm, X_test_pca, y_test)


# In[43]:


# 5.3.1 Deep Learning Model with Normal Dataset
### Experiment 3a: Apply Artificial Neural Networks - Without doing any sampling

print("\n************* ANN Normal Case - Baseline **************")

# Evaluate model and get result
ANN_model(X_train, y_train, X_test, y_test, epochs=5)


# In[44]:


# 5.3.2 Deep Learning Model with Oversampled Dataset
### Experiment 3b: Apply Artificial Neural Networks using Smote Technique sampling

print("\n************* ANN Oversampling Case - SMOTE **************")

# Evaluate model and get result
ANN_model(X_train_sm, y_train_sm, X_test, y_test, epochs=100)


# In[ ]:




