import numpy as np
from sklearn.model_selection import GridSearchCV,train_test_split, KFold, cross_val_score, StratifiedKFold
import csv
import random
import math
from numpy import interp
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report, confusion_matrix, auc, \
    precision_recall_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def get_stacking(clf, x_train, y_train, x_test, n_folds=5):
    """
    using cross-validation to get the second-level training set and testing set for the LogisticRegression classifier
    """
    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_level_train_set = np.zeros((train_num,))
    second_level_test_set = np.zeros((test_num,))
    test_nfolds_sets = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)

    #5-fold cross-validation
    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst =  x_train[test_index], y_train[test_index]
        clf.fit(x_tra, y_tra)
        second_level_train_set[test_index] = clf.predict_proba(x_tst)[:, 1]
        test_nfolds_sets[:,i] = clf.predict_proba(x_test)[:, 1]

    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, second_level_test_set


def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return


def MyConfusionMatrix(y_real,y_predict):
    CM = confusion_matrix(y_real, y_predict)
    CM = CM.tolist()
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    Acc = (TN + TP) / (TN + TP + FN + FP)
    Sen = TP / (TP + FN)
    Spec = TN / (TN + FP)
    Prec = TP / (TP + FP)
    F1 = (2*Prec*Sen)/(Prec+Sen)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    Result = []
    Result.append(round(Acc, 4))
    Result.append(round(Sen, 4))
    Result.append(round(Spec, 4))
    Result.append(round(Prec, 4))
    Result.append(round(F1, 4))
    Result.append(round(MCC, 4))
    return Result

def MyAverage(matrix):
    SumAcc = 0
    SumSen = 0
    SumSpec = 0
    SumPrec = 0
    SumF1 = 0
    SumMcc = 0
    counter = 0
    while counter < len(matrix):
        SumAcc = SumAcc + matrix[counter][0]
        SumSen = SumSen + matrix[counter][1]
        SumSpec = SumSpec + matrix[counter][2]
        SumPrec = SumPrec + matrix[counter][3]
        SumF1 = SumF1 + matrix[counter][4]
        SumMcc = SumMcc + matrix[counter][5]
        counter = counter + 1
    print('AverageAcc:',SumAcc / len(matrix))
    print('AverageSen:', SumSen / len(matrix))
    print('AverageSpec:', SumSpec / len(matrix))
    print('AveragePrec:', SumPrec / len(matrix))
    print('AverageF1:', SumF1 / len(matrix))
    print('AverageMcc:', SumMcc / len(matrix))
    return


# GridSearch for classifiers
def para_tuning(model,para,X,Y):
    grid_obj = GridSearchCV(model, para, scoring = 'roc_auc',cv=5)
    grid_obj2 = grid_obj.fit(X, Y)
    para_best = grid_obj2.best_estimator_
    print(grid_obj2.best_score_)
    print(grid_obj2.best_params_)
    return(para_best)

SampleFeature = []
feature=[]
ReadMyCsv(feature, "SampleFeature.csv")
for i in range(len(feature)):
    c = []
    for j in range(len(feature[0])):
        c.append(float(feature[i][j]))
    SampleFeature .append(c)

SampleLabel = []
counter = 0
while counter < len(SampleFeature) / 2:
    SampleLabel.append(1)
    counter = counter + 1
counter1 = 0
while counter1 < len(SampleFeature) / 2:
    SampleLabel.append(0)
    counter1 = counter1 + 1

counter = 0
R = []
while counter < len(SampleFeature):
    R.append(counter)
    counter = counter + 1
random.shuffle(R)

RSampleFeature = []
RSampleLabel = []
counter = 0
while counter < len(SampleFeature):
    RSampleFeature.append(SampleFeature[R[counter]])
    RSampleLabel.append(SampleLabel[R[counter]])
    counter = counter + 1
SampleFeature = []
SampleLabel = []
SampleFeature = RSampleFeature
SampleLabel = RSampleLabel
X = np.array(SampleFeature)
y = np.array(SampleLabel)

SplitNum = 5
cv = StratifiedKFold(n_splits=SplitNum) # #5-fold cross-validation

tprs = []
aucs = []
AverageResult = []
mean_fpr = np.linspace(0, 1, 100)

precisions=[]
auprs=[]
mean_recall=np.linspace(0, 1, 100)
    
    
for train, test in cv.split(X, y):

    base_classifiers=[]    #save 5 first-level classifiers
    for i in range(1):
        base_classifiers.append(RandomForestClassifier())
    for i in range(1,2):
        base_classifiers.append(CatBoostClassifier())
    for i in range(2,3):
        base_classifiers.append(ExtraTreesClassifier())
    for i in range(3,4):
        base_classifiers.append(XGBClassifier())
    for i in range(4,5):
        base_classifiers.append(LGBMClassifier())
    base_classifiers=np.array(base_classifiers)
    base_classifiers.flatten()

    train_sets = []
    test_sets = []

    for clf in range(5):
        train_set, test_set = get_stacking(base_classifiers[clf],X[train],y[train],X[test])
        train_sets.append(train_set)
        test_sets.append(test_set)

    meta_train = np.concatenate([result_set.reshape(-1,1) for result_set in train_sets], axis=1)
    meta_test = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in test_sets], axis=1)

    # second-level classifier       
    lr_model = LogisticRegression()

    lr_model.fit(meta_train, y[train])
    df_predict1 = lr_model.predict(meta_test)
    df_predict2 = lr_model.predict_proba(meta_test)
    fpr, tpr, thresholds = roc_curve(y[test], df_predict2[:, 1])

    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0

    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    precision, recall, thresholds = precision_recall_curve(y[test], df_predict2[:, 1])
    precisions.append(interp(mean_recall,precision,recall))
    
    aupr = auc(recall,precision)
    auprs.append(aupr)

    Result1 = MyConfusionMatrix(y[test],df_predict1)
    AverageResult.append(Result1)


MyAverage(AverageResult)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
    
print('mean_auc:')
print(mean_auc)
print('std_auc:')
print(std_auc)

save_fpr=np.array(mean_fpr)
np.savetxt('MAGCNSE_fpr.txt',save_fpr,fmt='%f',delimiter=' ')
save_tpr=np.array(mean_tpr)
np.savetxt('MAGCNSE_tpr.txt',save_tpr,fmt='%f',delimiter=' ')


mean_precision = np.mean(precisions, axis=0)
mean_aupr = auc(mean_recall,mean_precision)
std_aupr=np.std(auprs)
    
print('mean_aupr:')
print(mean_aupr)
print('std_aupr:')
print(std_aupr)

save_precision=np.array(mean_precision)
np.savetxt('MAGCNSE_precision.txt',save_precision,fmt='%f',delimiter=' ')
save_recall=np.array(mean_recall)
np.savetxt('MAGCNSE_recall.txt',save_recall,fmt='%f',delimiter=' ')


