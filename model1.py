import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor


#----- CONSTANT PARAMETERS --------

#dvelopment set path:
DATASETPATH = "./data/DSL_Winter_Project_2024/DSL_Winter_Project_2024/development.csv"

#evaluation set path:
EVALUATIONPATH = "./data/DSL_Winter_Project_2024/DSL_Winter_Project_2024/evaluation.csv"

#----------------------------------

#set initial random state
rs = 328537
np.random.seed(rs)

#generates columns names given the label (e.g. 'area').
#The generated indexes go from 0 to 17; to ingore a specific subset of indexes use 'ingore' param
def generateColumnsNames(title, ignore=[]):
    out = []
    for i in range(0, 18):
        if(not i in ignore):
            out.append( title + "[" + str(i) + "]" )
    return out

#generates column names given the indexes of the signals (e.g. 0, 1, 2)
def generateColumnsNamesFromIndexes(indexes):
    out = []
    for index in indexes:
        for label in ["pmax", "negpmax", "area", "tmax", "rms"]:
            out.append(label + "[" + str(index) + "]")
    return out

#calculate the mean euclidean distance, given the predictions and the groun truth 
#it's the commutative property (e.g. the order of the arguments is irrelevant)
def euc_dist(Y, Y_pred):
    return np.sqrt(((Y - Y_pred)**2).sum(axis=1)).mean()

#extract the unique elements from a sequence preserving the order
def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

#make ad-hoc scorer with the euclidean distance
euc_dist_scorer = make_scorer(euc_dist, greater_is_better=False)

#load development set
dataset = pd.read_csv(DATASETPATH)

#plot target spatial distribution
plt.figure()
dataset.plot.scatter("x", "y", alpha=0.1, s=10, figsize=(6,6), fontsize=14)
plt.savefig("target_spatial_distribution.png", format="png")

#plot tmax distributions
columnTitles = ["tmax"]
for columnTitle in columnTitles:
    plt.figure()
    columns = generateColumnsNames(columnTitle)
    dataset.hist(bins=10, column=columns, figsize=(18,12))
    plt.savefig(columnTitle + "_distributions.pdf", format="pdf")

#declare noisy columns and target columns
regressionTargets = ['x', 'y']
noisyIndexes = [0, 7, 12, 15, 16, 17]
    
# Train of a first model to identify key features
noisyFeatures= generateColumnsNamesFromIndexes(noisyIndexes)
dataset_prov = dataset.drop(columns=noisyFeatures)

reg_prov = RandomForestRegressor(100, random_state=rs)
reg_prov.fit(dataset_prov.iloc[:, 2:], dataset_prov.loc[:, 'x':'y'])
sorted_features = sorted(zip(dataset_prov[:, 2:].columns, reg_prov.feature_importances_), key=lambda x: x[1], reverse=True)

print("Top features:")
print(sorted_features)

#select a subset of features
featuresLabels = generateColumnsNames("area", ignore=noisyIndexes) + generateColumnsNames("pmax", ignore=noisyIndexes) + generateColumnsNames("negpmax", ignore=noisyIndexes)

#shuffle and prepare the dataset
dataset_shuff = dataset.sample(random_state=rs, frac=1)  #shuffle the dataset
X_df = dataset_shuff[featuresLabels]
Y_df = dataset_shuff[regressionTargets]

#grid search
reg = RandomForestRegressor(random_state=rs)
param_grid = {
    "n_estimators": [125, 150, 300, 500, 1000],
    "max_features": ["sqrt", "log2", None], 
    "random_state": [42],
    "max_depth": [22, 25, 28, None],
}

grid_search = GridSearchCV(reg, param_grid=params, scoring=euc_dist_scorer, verbose=3, n_jobs=-1) #uses multiple jobs
grid_search.fit(X_std, Y_df.values)

print(grid_search.best_params_)
print(-grid_search.best_score_)

#save grid search results to file
gs_df = pd.DataFrame(grid_search.cv_results_)
gs_df.to_csv("MLP_GS_results.csv")

#load evaluation set
evaluation = pd.read_csv(EVALUATIONPATH, index_col="Id")

#apply model
X_ev = evaluation[featuresLabels]
Y_ev = grid_search.predict(X_ev)

#generate submission file
output = pd.DataFrame()
Y_ev_df = pd.DataFrame(Y_ev)
display(Y_ev_df)
output['Predicted'] = (Y_ev_df[0]).astype(str) + "|" + (Y_ev_df[1]).astype(str)
output.to_csv("submission_MPL.csv", index_label="Id")

#plot gridsearch scores
layer_sizes = [x["hidden_layer_sizes"] for x in grid_search.cv_results_["params"]]
activations = unique([x["activation"] for x in grid_search.cv_results_["params"]])
scores = -grid_search.cv_results_["mean_test_score"].reshape(len(activations), -1).T
plt.figure(figsize=(6,4))
for i, marker in zip(range(scores.shape[0]), ['o', '^']):
    plt.plot(scores[:, i], marker=marker)
plt.xticks(range(0, scores.shape[0]), layer_sizes[:scores.shape[0]], rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Hidden layer sizes", fontsize=14)
plt.ylabel("Mean Euclidean distance", fontsize=14)
legend = plt.legend(activations, title="Activation function", fontsize=12)
plt.setp(legend.get_title(),fontsize=14)
plt.tight_layout()
plt.savefig("mlp_gridserach_scores.pdf", format="pdf")

#plot gridsearch fit times
layer_sizes = [x["hidden_layer_sizes"] for x in grid_search.cv_results_["params"]]
activations = unique([x["activation"] for x in grid_search.cv_results_["params"]])
times = grid_search.cv_results_["mean_fit_time"].reshape(len(activations), -1).T
plt.figure(figsize=(6,4))
plt.yscale("log")
for i, marker in zip(range(times.shape[0]), ['o', '^']):
    plt.plot(times[:, i], marker=marker)
plt.xticks(range(0, times.shape[0]), layer_sizes[:times.shape[0]], rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Hidden layer sizes", fontsize=14)
plt.ylabel("Mean fit time [s]", fontsize=14)
legend = plt.legend(activations, title="Activation function", fontsize=12)
plt.setp(legend.get_title(),fontsize=14)
plt.tight_layout()
plt.savefig("mlp_gridserach_fit_times.pdf", format="pdf")