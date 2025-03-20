from __future__ import print_function


__author__ = 'avisegal'
import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import joblib
from functools import reduce
from  sklearn.ensemble import RandomForestClassifier
from  sklearn.ensemble import ExtraTreesClassifier
from  sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
import pydotplus
from six import StringIO
from IPython.display import Image, display
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import r2_score
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import operator

import logging
import sys
# from scipy.stats import itemfreq


def plot_im(im, dpi=300):
    #px,py = im.shape # depending of your matplotlib.rc you mayhave to use py,px instead
    px,py = im[:,:,0].shape # if image has a (x,y,z) shape
    size = (py/np.float(dpi), px/np.float(dpi)) # note the np.float()


    fig = plt.figure(figsize=size, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    # Customize the axis
    # remove top and right spines
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    # turn off ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    ax.imshow(im, aspect='auto')
    plt.show()

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
#    with open("Decision_Tree.dot", 'w') as f:
#        dot_data = export_graphviz(tree, out_file=f,filled=True, rounded=True,proportion=True, precision=2,
#                        feature_names=feature_names)
#    command = ["dot", "-Tpng", "Decision_Tree.dot", "-o", "Decision_Tree.png"]
#    try:
#        subprocess.check_call(command)
#    except:
#        exit("Could not run dot, ie graphviz, to "
#             "produce visualization")

    dot_data = export_graphviz(tree, out_file=None,filled=True, rounded=True,proportion=True, precision=2,
                    feature_names=feature_names, class_names=("Safe","Risk!"))
    graph = pydotplus.graph_from_dot_data(dot_data)
    nodes = graph.get_node_list()

    for node in nodes:
        print (node.get_label())
        if node.get_label():
            values = [float(ii) for ii in node.get_label().split('value = [')[1].split(']')[0].split(',')]
            values = [int(255 * v) for v in values]
            color = '#{:02x}{:02x}{:02x}'.format(values[1],values[0],0)
            node.set_fillcolor(color)
    graph.write_png('Decision_tree.png')


def get_code(tree, feature_names, target_names,
             spacer_base="    "):
    """Produce psuedo-code for decision tree.

    Args
    ----
    tree -- scikit-leant DescisionTree.
    feature_names -- list of feature names.
    target_names -- list of target (class) names.
    spacer_base -- used for spacing code (default: "    ").

    Notes
    -----
    based on http://stackoverflow.com/a/30104792.
    """
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node, depth):
        spacer = spacer_base * depth
        if (threshold[node] != -2):
            print(spacer + "if ( " + features[node] + " <= " + \
                  str(threshold[node]) + " ) {")
            if left[node] != -1:
                    recurse(left, right, threshold, features,
                            left[node], depth+1)
            print(spacer + "}\n" + spacer +"else {")
            if right[node] != -1:
                    recurse(left, right, threshold, features,
                            right[node], depth+1)
            print(spacer + "}")
        else:
            target = value[node]
            for i, v in zip(np.nonzero(target)[1],
                            target[np.nonzero(target)]):
                target_name = target_names[i]
                target_count = int(v)
                print(spacer + "return " + str(target_name) + \
                      " ( " + str(target_count) + " examples )")

    recurse(left, right, threshold, features, 0, 0)

# MAIM Starts Here

labelstr = "suicattempt"
root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

dir = "data/"
fin = "../data/DataSet_Combined_SI_SNI_Baseline_FE.csv"

logging.info("Loading  Data")
origindf= pd.read_csv(fin)
logging.info("Data Loaded")
print("* origindf.head()", origindf.head(), sep="\n", end="\n\n")
print("* original label types:", origindf[labelstr].unique(), sep="\n")

logging.info("Generating sub tree for analysis")
df = origindf[['age', 'gender', 'sexori', 'raceall', 'trauma_sum', \
               'cesd_score', 'harddrug_life','school','degree','job',\
               'sex', 'concurrent', 'exchange', 'children',\
               'weapon','fight', 'fighthurt', 'ipv', 'ptsd_score', 'alcfirst', \
               'potfirst', 'staycurrent', 'homelage', 'time_homeless_month', 'jail', 'jailkid',\
               'gettherapy', 'sum_alter', 'prop_family', 'prop_home_friends',\
               'prop_street_friends', 'prop_unknown_alter', 'sum_talk_once_week', 'sum_alter3close', \
               'prop_family_harddrug', 'prop_friends_harddrug', 'prop_friends_home_harddrug', 'prop_friends_street_harddrug',\
               'prop_alter_all_harddrug', 'prop_enc_badbehave', 'prop_alter_homeless', 'prop_family_emosup', 'prop_friends_emosup',
               'prop_friends_home_emosup', 'prop_friends_street_emosup', 'prop_alter_all_emosup', 'prop_family_othersupport', 'prop_friends_othersupport',\
               'prop_friends_home_othersupport', 'prop_friends_street_othersupport', 'prop_alter_all_othersupport','sum_alter_staff',\
               'prop_object_badbehave', 'prop_enc_goodbehave', 'prop_alter_school_job', 'sum_alter_borrow',\
               labelstr]].copy()
print("* df.head()", df.head(), sep="\n", end="\n\n")
print ('(rows, columns):' + str(df.shape))
print()
logging.info("Cleaning sub tree from NaN")
dfn = df.dropna().copy()
print("* dfn.head()", df.head(), sep="\n", end="\n\n")
print ('(rows, columns):' + str(dfn.shape))
train_test_cutoff=int(round(dfn.shape[0]*.75))
print ("train test cutoff: " + str((train_test_cutoff)))

features = list(dfn.columns[:-1])
label= list(dfn.columns[-1])
targets = dfn[labelstr].unique()
print("* features:", features, sep="\n")
print ("labels count:")
print (dfn[labelstr].value_counts())

dfm = dfn.values
X_train, y_train = dfm[0:train_test_cutoff,:-1], dfm[0:train_test_cutoff,-1]
X_test, y_test = dfm[train_test_cutoff:,:-1], dfm[train_test_cutoff:,-1]
y = dfm[:,-1]
#print (y_train)

# Data preparation Ended

logging.info("Fitting Model")
cw = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
print ("Class Weighting: " + str(cw))
cwt={0:cw[0], 1:cw[1]}

clf = DecisionTreeClassifier(criterion='gini',min_samples_leaf=10, min_samples_split=30, max_depth=4, class_weight=cwt, min_impurity_decrease=0.01)
clf.fit(X_train, y_train)

logging.info("Fitting Ended")

visualize_tree(clf, features)
print ()
#get_code(clf, features, targets)

logging.info("Predicting...")
y_predicted = clf.predict(X_test)
y_predicted_prob=clf.predict_proba(X_test)
logging.info("Prediction Ended")

#print (y_predicted_prob)

# Print metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_predicted))

# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
sorted_idx= sorted_idx[-20:]
i=np.searchsorted(feature_importance,0) # find zero location
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
X_feature_names=np.asarray(features)
plt.yticks(pos, X_feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')

print("Feature Importance %:")
for id in sorted_idx:
    if (feature_importance[id]>0):
        print(X_feature_names[id]+": "+str(feature_importance[id]))

n_classes = 2
y_test_bin=label_binarize(y_test, classes=range(n_classes+1)) # +1 to ovecome sklearn issue with binarizing of 2 classes
print ("-----------------------------------------------------")
print ("Accuracy:" + str(accuracy_score(y_test,y_predicted)))
print("AUC: " + str(roc_auc_score(y_test_bin[:,:-1], y_predicted_prob, average='micro')))
print ("Classification Report:")
print(classification_report(y_test, y_predicted))
print ("Note: \nSensitivity == Recall of the Positive Class.\nSpecificity == Recall of the Negative Class.\n")
print ("-----------------------------------------------------")

f = open('Metrics.txt','w')
f. write("-----------------------------------------------------\n")
f.write("Features: ")
f.write(str(features))
f.write("\nLabel: ")
print (label)
lbl=reduce(operator.add, label)
f.write(lbl)
f. write ("\n\nAccuracy:" + str(accuracy_score(y_test,y_predicted))+'\n')
f. write("AUC: " + str(roc_auc_score(y_test_bin[:,:-1], y_predicted_prob, average='micro'))+'\n')
f. write ("Classification Report:\n")
f. write(classification_report(y_test, y_predicted)+'\n')
f. write ("Note: \nSensitivity == Recall of the Positive Class.\nSpecificity == Recall of the Negative Class.\n")
f. write ("-----------------------------------------------------\n")
f.close()
plt.savefig("Feature Importance.png")
plt.show()


# plot ROC curve

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
y_score = y_predicted_prob
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin[:,:-1].ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
plt.figure()

lw = 2
plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating Characteristic (Micro) Curve')
plt.legend(loc="lower right")
plt.savefig("Roc Curve.png")
plt.show()

mpl.rcParams['figure.dpi']= 300
img = mpimg.imread('Decision_Tree.png')
plot_im(img)
plt.show()

#print ()
#get_code(clf, features, targets)


