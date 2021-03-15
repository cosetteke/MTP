# Load packages
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import sklearn.metrics as metrics
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def load_data(dataname):
    if dataname in ['ERN', 'SRN']:
        X1 = np.loadtxt('./dataset/'+str(dataname)+'/X1.txt',delimiter=",")
        Y = np.loadtxt('./dataset/'+str(dataname)+'/Y.txt',delimiter=",")
        X2 = np.loadtxt('./dataset/'+str(dataname)+'/X2.txt',delimiter=",")
    else:
        X1 = np.loadtxt('./dataset/'+str(dataname)+'/'+str(dataname)+'_X1.txt')
        Y = np.loadtxt('./dataset/'+str(dataname)+'/'+str(dataname)+'_Y.txt')
        X2 = np.loadtxt('./dataset/'+str(dataname)+'/'+str(dataname)+'_X2.txt')
    Y_T = np.transpose(Y)
    return X1, X2, Y, Y_T


metrics_to_calculate = ['auroc', 'aupr']
metric_values_per_fold = {}
if 'auroc' in metrics_to_calculate:
    metric_values_per_fold['auroc_micro'] = []
    metric_values_per_fold['auroc_macro'] = []
if 'aupr' in metrics_to_calculate:
    metric_values_per_fold['aupr_micro'] = []
    metric_values_per_fold['aupr_macro'] = []

shuffled = [True, False]
datanames = ['ERN', 'SRN', 'dpie','dpii','dpig','dpin']
for dataname in datanames:
    X1, X2, Y, Y_T = load_data(dataname)

    setting = 'B'
    if setting == 'C':
        X1, Y = X2, Y_T
    print(dataname)

    for item in shuffled:
        
        kf = KFold(n_splits=10, shuffle=item, random_state=42)
        fold_counter = 0

        for train_index, test_index in kf.split(X1):
            print('======================= Fold '+str(fold_counter)+' =========================================')

            # split the dataset
            X_train, X_test = X1[train_index], X1[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            # define the oneVSrest classifier with the base classifier
            # clf = OneVsRestClassifier(RandomForestClassifier())
            # clf = OneVsRestClassifier(LogisticRegression(random_state=0))
            # clf = OneVsRestClassifier(MLPClassifier(random_state=1, hidden_layer_sizes=(256), solver='adam', learning_rate='adaptive', max_iter=300)) # binary relevance approach that uses a neural network as the base classifier (so it creates as many neural networks as there are labels)
            clf = MLPClassifier(random_state=1, hidden_layer_sizes=(512), solver='adam', learning_rate='adaptive', max_iter=300) # standard neural network

            # fit the classifier on the training set
            clf.fit(X_train, y_train)

            # generate probability predictions for every sample in the test set
            y_pred = clf.predict_proba(X_test)

            print(str(y_pred.shape))

            # calculate the performance metrics on the test set
            if 'auroc' in metrics_to_calculate:
                metric_values_per_fold['auroc_micro'].append(roc_auc_score(y_test, y_pred, average='micro'))

                # This is not really important as we are only interested in the micro measures.
                # Nevertheless, I basically do the macro averaging by hand so that I can skip labels that have only samples with one class
                roc_auc_per_label = []
                for label_idx in range(Y.shape[1]): # 0是行 1是列
                    if len(set(y_test[:, label_idx])) >= 2: # here test is validation
                        roc_auc_per_label.append(roc_auc_score(y_test[:, label_idx], y_pred[:, label_idx]))
                print(str(len(roc_auc_per_label))+' out of the '+str(y_test.shape[1])+' total labels has more than one classes present')

                metric_values_per_fold['auroc_macro'].append(np.mean(roc_auc_per_label))


            if 'aupr' in metrics_to_calculate:
                metric_values_per_fold['aupr_micro'].append(average_precision_score(y_test, y_pred, average='micro'))

                aupr_per_label = []
                for label_idx in range(Y.shape[1]):
                    if len(set(y_test[:, label_idx])) >= 2:
                        aupr_per_label.append(average_precision_score(y_test[:, label_idx], y_pred[:, label_idx]))

                metric_values_per_fold['aupr_macro'].append(np.mean(aupr_per_label))


            fold_counter += 1
            print('========================================================================')
            print('')
            
        # calculate the mean and std for every metric measured during training and validation
        print('shuffle = ' + str(item))
        for metric_name in metric_values_per_fold.keys():
            print(metric_name+': '+ str('%.4f' % np.mean(metric_values_per_fold[metric_name])) +' ('+ str('%.4f' % np.std(metric_values_per_fold[metric_name])) +')')
            print('')


# In[ ]:





# In[ ]:


# save result

