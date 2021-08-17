import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot

mu_A_0 = 0.0
mu_A_1 = 1.0
mu_B_0 = 3.0
mu_B_1 = 4.0
sigma_A = 1.5
sigma_B = sigma_A
ratio_A_1_to_0 = 4
ratio_B_1_to_0 = 19
number_A_ALL = 200000
number_B_ALL = number_A_ALL

predict_A_0 = mu_A_0 + sigma_A * np.random.randn(number_A_ALL//(ratio_A_1_to_0+1))
predict_A_1 = mu_A_1 + sigma_A * np.random.randn(number_A_ALL//(ratio_A_1_to_0+1)*ratio_A_1_to_0)
predict_B_0 = mu_B_0 + sigma_B * np.random.randn(number_B_ALL//(ratio_B_1_to_0+1))
predict_B_1 = mu_B_1 + sigma_B * np.random.randn(number_B_ALL//(ratio_B_1_to_0+1)*ratio_B_1_to_0)

print_predict_value_distribution = True
count_AUC = True

if print_predict_value_distribution:
    distribution_A_0 = {}
    for item in predict_A_0:
        item_cut = int(item * 100) / 100.0

        if item_cut in distribution_A_0.keys():
            distribution_A_0[item_cut] += 1
        else:
            distribution_A_0[item_cut] = 1

    distribution_A_1 = {}
    for item in predict_A_1:
        item_cut = int(item * 100) / 100.0

        if item_cut in distribution_A_1.keys():
            distribution_A_1[item_cut] += 1
        else:
            distribution_A_1[item_cut] = 1

    distribution_B_0 = {}
    for item in predict_B_0:
        item_cut = int(item * 100) / 100.0

        if item_cut in distribution_B_0.keys():
            distribution_B_0[item_cut] += 1
        else:
            distribution_B_0[item_cut] = 1

    distribution_B_1 = {}
    for item in predict_B_1:
        item_cut = int(item * 100) / 100.0

        if item_cut in distribution_B_1.keys():
            distribution_B_1[item_cut] += 1
        else:
            distribution_B_1[item_cut] = 1

    pyplot.scatter(distribution_A_0.keys(), distribution_A_0.values(), marker='.', color=(1, 0, 0))
    pyplot.scatter(distribution_A_1.keys(), distribution_A_1.values(), marker='+', color=(1, 0, 0))
    pyplot.scatter(distribution_B_0.keys(), distribution_B_0.values(), marker='.', color=(0, 0, 1))
    pyplot.scatter(distribution_B_1.keys(), distribution_B_1.values(), marker='+', color=(0, 0, 1))
    # axis labels
    pyplot.xlabel('Predict value')
    pyplot.ylabel('Distribution')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()

y_A = []
pred_A = []
y_A.extend(np.zeros(len(predict_A_0)))
pred_A.extend(predict_A_0)
y_A.extend(np.ones(len(predict_A_1)))
pred_A.extend(predict_A_1)

y_B = []
pred_B = []
y_B.extend(np.zeros(len(predict_B_0)))
pred_B.extend(predict_B_0)
y_B.extend(np.ones(len(predict_B_1)))
pred_B.extend(predict_B_1)

y_ALL = []
pred_ALL = []
y_ALL.extend(y_A)
y_ALL.extend(y_B)
pred_ALL.extend(pred_A)
pred_ALL.extend(pred_B)

if count_AUC:
    roc_auc_A = roc_auc_score(y_A, pred_A)
    print('roc_auc_A = %.3f' % roc_auc_A)

    roc_auc_B = roc_auc_score(y_B, pred_B)
    print('roc_auc_B = %.3f' % roc_auc_B)

    roc_auc_ALL = roc_auc_score(y_ALL, pred_ALL)
    print('roc_auc_ALL = %.3f' % roc_auc_ALL)
    
    AUC_diff = (2 * roc_auc_ALL - roc_auc_A - roc_auc_B) / 2
    print ('AUC_diff = %.3f' % AUC_diff)

    lr_fpr_A, lr_tpr_A, _ = roc_curve(y_A, pred_A)
    pyplot.plot(lr_fpr_A, lr_tpr_A, marker='.', color=(1, 0, 0))
    lr_fpr_B, lr_tpr_B, _ = roc_curve(y_B, pred_B)
    pyplot.plot(lr_fpr_B, lr_tpr_B, marker='.', color=(0, 1, 0))
    lr_fpr_ALL, lr_tpr_ALL, _ = roc_curve(y_ALL, pred_ALL)
    pyplot.plot(lr_fpr_ALL, lr_tpr_ALL, marker='.', color=(0, 0, 1))
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('sigma_A=%s' % sigma_A)
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
