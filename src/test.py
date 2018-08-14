from sklearn import tree
from pdb import set_trace
clf = tree.DecisionTreeClassifier()
X = [[0, 0], [2, 2],[-1,-3],[1,3]]
y = ['yes','no','yes','no']

clf.fit(X,y)
pos_at = list(clf.classes_).index('yes')
train_prob = clf.predict_proba(X)
set_trace()