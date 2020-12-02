import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import sklearn.metrics as metrics
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

# keras
import keras
from keras.layers import Input, Dense
from keras.models import Model

drug_features = np.loadtxt('/Users/mac/OneDrive/thesis/scripts/dataset/DPI_enzyme/drug_feature.txt')
interaction_matrix = np.loadtxt('/Users/mac/OneDrive/thesis/scripts/dataset/DPI_enzyme/dpie_Y.txt')

metrics_to_calculate = ['auroc', 'aupr']

metric_values_per_fold = {}
if 'auroc' in metrics_to_calculate:
    metric_values_per_fold['auroc_micro'] = []
    metric_values_per_fold['auroc_macro'] = []
if 'aupr' in metrics_to_calculate:
    metric_values_per_fold['aupr_micro'] = []
    metric_values_per_fold['aupr_macro'] = []

kf = KFold(n_splits=10, shuffle=True, random_state=42)

fold_counter = 0
for train_index, test_index in kf.split(drug_features):
    print('======================= Fold ' + str(fold_counter) + ' =======================')

    # split the dataset
    X_train, X_test = drug_features[train_index], drug_features[test_index]
    y_train, y_test = interaction_matrix[train_index], interaction_matrix[test_index]
    print(X_train.shape, X_test.shape)  # (597, 664) (67, 664)
    print(y_train.shape, y_test.shape)  # (597, 445) (67, 445)

    # define the oneVSrest classifier with the base classifier

    # model

    inputs = Input(shape=(664,))
    # 所有的模型都是可调用的，就像层一样
    from keras.layers import Dropout, BatchNormalization,Activation
    #x = BatchNormalization()(inputs)

    x = Dense(512, activation='relu')(inputs)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)


    # 得到输出的张量prediction
    predictions = Dense(445, activation='sigmoid')(x) # sigmoid一般用于binary classfication


    # This creates a model that includes
    # the Input layer and three Dense layers
    # 用model生成模型
    model = Model(inputs=inputs, outputs=predictions)
    # 编译模型，指定优化参数(默认方法)、损失函数、效用评估函数
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', # 二分类
                  metrics=['accuracy'])
    # 传入数据进行训练
    model.fit(X_train, y_train, epochs=300, batch_size=64)  # starts training

    # generate probability predictions for every sample in the test set
    y_pred = model.predict(X_test)


    # calculate the performance metrics on the test set
    if 'auroc' in metrics_to_calculate:
        metric_values_per_fold['auroc_micro'].append(roc_auc_score(y_test, y_pred, average='micro'))

        # This is not really important as we are only interested in the micro measures.
        # Nevertheless, I basically do the macro averaging by hand so that I can skip labels that have only samples with one class
        roc_auc_per_label = []
        for label_idx in range(interaction_matrix.shape[1]):
            if len(set(y_test[:, label_idx])) >= 2:
                roc_auc_per_label.append(roc_auc_score(y_test[:, label_idx], y_pred[:, label_idx]))
        print(str(len(roc_auc_per_label)) + ' out of the ' + str(
            y_test.shape[1]) + ' total labels has more than one classes present')

        metric_values_per_fold['auroc_macro'].append(np.mean(roc_auc_per_label))

    if 'aupr' in metrics_to_calculate:
        metric_values_per_fold['aupr_micro'].append(average_precision_score(y_test, y_pred, average='micro'))

        aupr_per_label = []
        for label_idx in range(interaction_matrix.shape[1]):
            if len(set(y_test[:, label_idx])) >= 2:
                aupr_per_label.append(average_precision_score(y_test[:, label_idx], y_pred[:, label_idx]))

        metric_values_per_fold['aupr_macro'].append(np.mean(aupr_per_label))

    fold_counter += 1
    print('========================================================================')
    print('')


# calculate the mean and std for every metric measured during training and validation
for metric_name in metric_values_per_fold.keys():
    print(metric_name+': '+ str(np.mean(metric_values_per_fold[metric_name])) +' ('+ str(np.std(metric_values_per_fold[metric_name])) +')')
    print('')



# auroc_micro: 0.9068833081892113 (0.036407076855024055)
# auroc_macro: 0.864190760729695 (0.04743285885207322)

# aupr_micro: 0.7420953931720107 (0.07780001631679835)
# aupr_macro: 0.6792063065285503 (0.10186137684887413)









