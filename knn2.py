import numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import *

X = np.load("X.npy"); y = np.load("y.npy")
y_clf = (y>=15).astype(int)

kf = StratifiedKFold(n_splits=10,shuffle=True)
results=[]

for f,(tr,val) in enumerate(tqdm(list(kf.split(X,y_clf)))):

    clf = KNeighborsClassifier(n_neighbors=5).fit(X[tr],y_clf[tr])
    probs = clf.predict_proba(X[val])[:,1]
    preds = clf.predict(X[val])

    reg = KNeighborsRegressor(n_neighbors=5).fit(X[tr],y[tr])
    rpred = reg.predict(X[val])

    tn,fp,fn,tp = confusion_matrix(y_clf[val],preds).ravel()

    results.append([
        f+1,
        accuracy_score(y_clf[val],preds),
        roc_auc_score(y_clf[val],probs),
        f1_score(y_clf[val],preds),
        tp/(tp+fn),
        tn/(tn+fp),
        np.sqrt(mean_squared_error(y[val],rpred)),
        np.corrcoef(y[val],rpred)[0,1]
    ])

pd.DataFrame(results,columns=["Fold","Acc","AUC","F1","Sens","Spec","RMSE","R"])\
.to_csv("results/knn.csv",index=False)