import numpy as np
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation
from pdb import set_trace


label_prop_model = LabelPropagation()
iris = datasets.load_iris()
rng = np.random.RandomState(42)
random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
labels = np.copy(iris.target)
labels[random_unlabeled_points] = -1
label_prop_model.fit(iris.data, labels)
set_trace()
