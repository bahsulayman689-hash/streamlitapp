import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import  seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRFClassifier, XGBRFRegressor
from sklearn.model_selection import validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score
import warnings as wr
wr.filterwarnings('ignore')
plant_data = pd.read_csv("plant_disease_dataset.csv")
print(plant_data.head())
print(plant_data.info())
print(plant_data.shape)
print(plant_data.isnull().sum())
print(plant_data.duplicated().sum())
print(plant_data['disease_present'].value_counts())
print(plant_data.columns.tolist())

print(plant_data.describe())

plt.figure(figsize=(10, 8))
sns.heatmap(plant_data.corr(), annot=True, fmt='.2f', cbar=True, cmap='coolwarm', annot_kws={"size": 8}, square=True, 
            linewidths=4)

plt.show
X = plant_data.drop(columns='disease_present', axis=0)
print(X)
y = plant_data['disease_present']
print(y)
print('\n' , '#'*60)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

print(X.shape)
print(X_train.shape)
print(X_test.shape)
print( '#'*60, '\n')
model_logistic = LogisticRegression(max_iter=200,
                                    class_weight='balanced',
                                   random_state=42 )
model_logistic.fit(X_train , y_train)
score_1 = cross_val_score(model_logistic,
                          X, 
                          y, 
                          cv=3)
y_pred_test = model_logistic.predict(X_test)
print(accuracy_score(y_test , y_pred_test))
print(confusion_matrix(y_test , y_pred_test))
print(classification_report(y_test , y_pred_test))
print(score_1)
print('#'*60, )

model_random = RandomForestClassifier(max_depth=5,
                                      min_samples_leaf=5,
                                      min_samples_split=10, 
                                      class_weight={0:1, 1:5},
                                      n_estimators=500,
                                      bootstrap=True,
                                      criterion='gini',
                                      random_state=42)
model_random.fit(X_train , y_train)
score_2 = cross_val_score(model_random,
                          X,
                          y,
                          cv=3,
                          )

y_test_random = model_random.predict(X_test)
print(accuracy_score(y_test, y_test_random))
print(confusion_matrix(y_test, y_test_random))
print(classification_report(y_test, y_test_random))
print(score_2)
print( '#'*60, '\n')
model_tree = DecisionTreeClassifier(max_depth=5,
                                    min_samples_leaf=5,
                                    random_state=42, 
                                    min_samples_split=10,
                                    splitter='best',
                                    ccp_alpha=0.01,
                                    class_weight={0:1, 1:4},
                                    )
model_tree.fit(X_train, y_train)
y_pred_tree = model_tree.predict(X_test)
score_3 = cross_val_score(model_tree,
                          X,
                          y, 
                          cv=3)
print(accuracy_score(y_test, y_pred_test))
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))
print(score_3)
plt.figure(figsize=(20, 15))
plot_tree(model_tree, 
          filled=True,
          rounded=True,
          max_depth=5,
          )
plt.tight_layout()
plt.show()
print('#'*60, '\n')

model_boosting = XGBRFClassifier()
model_boosting.fit(X_train, y_train)
y_test_pred_boosting = model_boosting.predict(X_test)
score_4 = cross_val_score(model_boosting,
                          X, y,
                          cv=4,
                          )
print(accuracy_score(y_test, y_test_pred_boosting))
print(confusion_matrix(y_test, y_test_pred_boosting))
print(classification_report(y_test, y_test_pred_boosting))
print(score_4)

print('\n' , '#'*60)
model_light = LGBMClassifier(num_leaves=10,
                             max_depth=5,
                             n_estimators=500,
                             learning_rate=0.5,
                             random_state=42,
                             min_child_samples=20, 
                             
                             class_weight={0:1, 1:5})
model_light.fit(X_train, y_train)
y_test_ligth_pred = model_light.predict(X_test)
score_5 = cross_val_score(model_light,
                          X, 
                          y)
print(accuracy_score(y_test, y_test_ligth_pred))
print(confusion_matrix(y_test, y_test_ligth_pred))
print(classification_report(y_test, y_test_ligth_pred))
print(f"scores: {score_5}")
print("="*60, '\n')
model_gradient = GradientBoostingClassifier(learning_rate=0.5,
                                            n_estimators=500,
                                            min_samples_leaf=5,
                                            max_depth=5, 
                                            min_samples_split=10,
                                            random_state=42,
                                            
                                            
                                            
                                
                                            )
print(model_gradient)
model_gradient.fit(X_train, y_train)
y_pred_gradient = model_gradient.predict(X_test)
score_6 = cross_val_score(model_gradient,
                          X,
                          y)
print(accuracy_score(y_test, y_pred_gradient))
print(confusion_matrix(y_test, y_pred_gradient))
print(classification_report(y_test, y_pred_gradient))
print(f"score: {score_6}")

print('#'*60, '\n')
model_nural_network = MLPClassifier(max_iter=100,
                                    activation='relu',
                                    learning_rate='constant',
                                    momentum=0.1,
                                    verbose=1,
                                    shuffle=True,
                                    random_state=42,
                                    alpha=0.5)
model_nural_network.fit(X_train, y_train)
y_test_pred_network = model_nural_network.predict(X_test)
print(y_test_pred_network)
score_7 = cross_val_score(model_nural_network,
                          X,
                          y)
print(accuracy_score(y_test, y_test_pred_network))
print(confusion_matrix(y_test, y_test_pred_network))
print(classification_report(y_test, y_test_pred_network))
print(score_7)
print('#'*60, '\n')

model_gussian = GaussianNB()
model_gussian.fit(X_train, y_train)
score_8 = cross_val_score(model_gussian,
                          X, y)
y_test_gussain_pred = model_gussian.predict(X_test)
print(accuracy_score(y_test, y_test_gussain_pred))
print(confusion_matrix(y_test, y_test_gussain_pred))
print(classification_report(y_test, y_test_gussain_pred))
print(score_8)
print('#'*60, '\n')
model_neigbor = KNeighborsClassifier()
model_neigbor.fit(X_train, y_train)
score_9 = cross_val_score(model_neigbor,
                          X, 
                          y)
y_pred_neigbor = model_neigbor.predict(X_test)
print(accuracy_score(y_test, y_pred_neigbor))
print(confusion_matrix(y_test, y_pred_neigbor))
print(classification_report(y_test, y_pred_neigbor))
print(r2_score(y_test, y_pred_neigbor))
print(score_9)
   