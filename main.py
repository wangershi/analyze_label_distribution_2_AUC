import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot

mu_0 = 0.0
mu_1 = 1.0
sigma = 0.3
ratio_1_to_0 = 10
number_0 = 100000

predict_0 = mu_0 + sigma * np.random.randn(number_0)
predict_1 = mu_1 + sigma * np.random.randn(number_0*ratio_1_to_0)

print_predict_value_distribution = True
count_AUC = True
count_PRAUC = True

if print_predict_value_distribution:
    distribution_0 = {}
    for item in predict_0:
        item_cut = int(item * 100) / 100.0

        if item_cut in distribution_0.keys():
            distribution_0[item_cut] += 1
        else:
            distribution_0[item_cut] = 1

    distribution_1 = {}
    for item in predict_1:
        item_cut = int(item * 100) / 100.0

        if item_cut in distribution_1.keys():
            distribution_1[item_cut] += 1
        else:
            distribution_1[item_cut] = 1

    pyplot.scatter(distribution_0.keys(), distribution_0.values(), marker='.', color=(1, 0, 0))
    pyplot.scatter(distribution_1.keys(), distribution_1.values(), marker='.', color=(0, 1, 0))
    # axis labels
    pyplot.xlabel('Predict value')
    pyplot.ylabel('Distribution')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()

y = []
pred = []
y.extend(np.zeros(len(predict_0)))
pred.extend(predict_0)
y.extend(np.ones(len(predict_1)))
pred.extend(predict_1)

if count_AUC:
    roc_auc = roc_auc_score(y, pred)
    print('ROC AUC=%.3f' % roc_auc)

    lr_fpr, lr_tpr, _ = roc_curve(y, pred)
    pyplot.plot(lr_fpr, lr_tpr, marker='.')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('sigma=%s' % sigma)
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()

if count_PRAUC:
    lr_precision, lr_recall, _ = precision_recall_curve(y, pred)
    lr_auc = auc(lr_recall, lr_precision)
    print('Logistic: auc=%.3f' % lr_auc)
    
    pyplot.plot(lr_recall, lr_precision, marker='.')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)
    pyplot.title('sigma=%s' % sigma)
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
