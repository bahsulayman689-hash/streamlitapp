import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from lightgbm import LGBMClassifier
import warnings as wr
import joblib
wr.filterwarnings('ignore')


data = pd.read_csv("Churn_modelling.csv")
print(data.head())
print(data.shape)
print(data.info())
print(data.columns)
print(data.describe())
print(data.isnull().sum())
print(data.duplicated().sum())
print(data['Exited'].value_counts())
#print(data.drop(columns=['Geography', 'Surname'], axis=0))
data['Gender'] = data['Gender'].map({"Female": 0, "Male": 1})


print(data.head())
print(data.columns)

X =  data.drop(columns=['Geography', 'Surname', 'Exited'], axis=0)
print(X)
y = data['Exited']
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=2,
                                                    stratify=y)
print(X.shape)
print(X_train.shape)
print(X_test.shape)
model = LogisticRegression(max_iter=1000,
                           class_weight='balanced',
                           verbose=0,
                           solver='liblinear',
                           fit_intercept=True)
random_forest_model = RandomForestClassifier(max_depth=None,
                                             min_samples_split=10, 
                                             
                                             n_estimators=400,
                                             random_state=2,
                                             class_weight={0:1, 1:3}
                                            )
dicision_tree_model = DecisionTreeClassifier(max_depth=None,
                                             min_samples_leaf=10,
                                             max_leaf_nodes=10, 
                                             class_weight='balanced',
                                             random_state=2,
                                             
                                             )
model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)
dicision_tree_model.fit(X_train, y_train)
y_test_prediction = model.predict(X_test)
print(f"Accuracy of the model test is:\n {accuracy_score(y_test, y_test_prediction)}")
print(confusion_matrix(y_test, y_test_prediction))
print(classification_report(y_test, y_test_prediction))
print(roc_auc_score(y_test, y_test_prediction))

y_train_prediction = model.predict(X_train)
print(f"Accuracy of the model train is :\n {accuracy_score(y_train, y_train_prediction)}")
print(confusion_matrix(y_train, y_train_prediction))

print("randomforestclassifier_model..")
y_test_prediction_rf = random_forest_model.predict(X_test)
print(accuracy_score(y_test, y_test_prediction_rf))
print(confusion_matrix(y_test, y_test_prediction_rf))
print(classification_report(y_test, y_test_prediction_rf))
print(roc_auc_score(y_test, y_test_prediction_rf))

print("dicision tree_model...")
y_test_prediction_df = dicision_tree_model.predict(X_test)
print(accuracy_score(y_test, y_test_prediction_df))
print(confusion_matrix(y_test, y_test_prediction_df))
print(classification_report(y_test, y_test_prediction_df))

y_train_prediction_rf = random_forest_model.predict(X_train)
print(accuracy_score(y_train, y_train_prediction_rf))

lightmodel = LGBMClassifier(n_estimators=500,
                            boosting_type='gbdt',
                            min_child_samples=20,
                            random_state=2,
                            learning_rate=0.05, 
                            max_depth=-1,
                            num_leaves=31,
                            class_weight={0:1, 1:5})
import joblib

joblib.dump(model, "churn_model.pkl")

 # if you used scaling
print("lightgbm...")
lightmodel.fit(X_train, y_train)
y_test_lightweight = lightmodel.predict(X_test)
print(accuracy_score(y_test, y_test_lightweight))
print(confusion_matrix(y_test, y_test_lightweight))
print(classification_report(y_test, y_test_lightweight))
