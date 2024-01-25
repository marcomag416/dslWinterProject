import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer

# Definition of the metric to compute the error during the development phase
def euc_dist(Y, Y_pred):
    return np.sqrt(((Y - Y_pred)**2).sum(axis=1)).mean()

euc_dist_scorer = make_scorer(euc_dist, greater_is_better=False)

# Development data
dataset = pd.read_csv('/Users/martinagalfre/Desktop/DSL_Winter_Project_2024/development.csv',sep=',', header=0)
dataset = dataset.sample(frac=1, random_state=42)

# Train of a first model to identify key features
noisyFeatures= generateColumnsNamesFromIndexes(noisyIndexes)

reg = RandomForestRegressor(100, random_state=42)
reg.fit(dataset.iloc[:, 2:], dataset.loc[:, 'x':'y'])
sorted(zip(X.columns, reg.feature_importances_), key=lambda x: x[1], reverse=True)

# Removal of noisy columns and feature reduction
df_dropped = dataset.drop(columns=["tmax[0]","tmax[1]","tmax[2]", "tmax[3]", "tmax[4]","tmax[5]","tmax[6]","tmax[7]", "tmax[8]", "tmax[9]", "tmax[10]","tmax[11]","tmax[12]", "tmax[13]", "tmax[14]", "tmax[15]","tmax[16]","tmax[17]"])
df_dropped = df_dropped.drop(columns=["rms[0]","rms[1]","rms[2]", "rms[3]", "rms[4]","rms[5]","rms[6]","rms[7]", "rms[8]", "rms[9]", "rms[10]","rms[11]","rms[12]", "rms[13]", "rms[14]", "rms[15]","rms[16]","rms[17]"])
df_dropped = df_dropped.drop(columns=["pmax[0]","pmax[7]","pmax[12]","pmax[15]","pmax[16]","pmax[17]", "area[0]","area[7]","area[12]","area[15]","area[16]","area[17]", "negpmax[0]","negpmax[7]","negpmax[12]","negpmax[15]","negpmax[16]","negpmax[17]"])

# Dataset containing the events
X = df_dropped.iloc[:, 2:]

# Grid search
param_grid = {
    "n_estimators": [125, 150, 300, 500, 1000],
    "max_features": ["sqrt", "log2", None], 
    "random_state": [42],
    "max_depth": [22, 25, 28, None],
}

random_forest = GridSearchCV(reg, param_grid, n_jobs=1, verbose=3, pre_dispatch='n_jobs')
random_forest.fit(X, df_dropped.loc[:, 'x':'y'])

# Hyperparameters of the top-performing model
parameters = random_forest.best_params_
parameters

random_forest.cv_results_

# Evaluation data
dataset_ev = pd.read_csv('/Users/martinagalfre/Desktop/DSL_Winter_Project_2024/evaluation.csv',sep=',', header=0)

# Unnecessary columns removal
df_dropped_ev = dataset_ev.drop(columns=["tmax[0]","tmax[1]","tmax[2]", "tmax[3]", "tmax[4]","tmax[5]","tmax[6]","tmax[7]", "tmax[8]", "tmax[9]", "tmax[10]","tmax[11]","tmax[12]", "tmax[13]", "tmax[14]", "tmax[15]","tmax[16]","tmax[17]"])
df_dropped_ev = df_dropped_ev.drop(columns=["rms[0]","rms[1]","rms[2]", "rms[3]", "rms[4]","rms[5]","rms[6]","rms[7]", "rms[8]", "rms[9]", "rms[10]","rms[11]","rms[12]", "rms[13]", "rms[14]", "rms[15]","rms[16]","rms[17]"])
df_dropped_ev = df_dropped_ev.drop(columns=["Id","pmax[0]","pmax[7]","pmax[12]","pmax[15]","pmax[16]","pmax[17]", "area[0]","area[7]","area[12]","area[15]","area[16]","area[17]", "negpmax[0]","negpmax[7]","negpmax[12]","negpmax[15]","negpmax[16]","negpmax[17]"])

# Prediction of the target values
Y = random_forest.predict(df_dropped_ev)

# Exporting of the results in cvs format
output = pd.DataFrame()
Y_df = pd.DataFrame(Y)
display(Y_df)

output['Predicted'] = (Y_df[0]).astype(str) + "|" + (Y_df[1]).astype(str)
output.to_csv("submission_rf.csv", index_label="Id")