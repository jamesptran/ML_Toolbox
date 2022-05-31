from flask import Blueprint
from flask import Blueprint, request, g, jsonify, abort, current_app
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap
from flask import Response
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io

import pickle
import numpy as np
import pandas as pd

mltask_controller = Blueprint('ml_task', __name__)

@mltask_controller.route('/train', methods=['POST'])

def train():
	req_data = request.get_json()
	model = req_data['model_name']
	ml_dict  = {'K Nearest Neighbor': KNeighborsClassifier(3),\
				'Random Forest': SVC(gamma=2, C=1),\
				'Support Vector Machine':  RandomForestClassifier(max_depth=5, \
										   n_estimators=10,\
										   max_features=1) }
	df_train = pd.read_csv('train.csv')
	X_train = df_train[[col for col in df_train.columns if col.startswith('feature')]].to_numpy()
	y_train = df_train['label'].to_numpy()
	X_train = StandardScaler().fit_transform(X_train)
	clf = ml_dict[model]
	clf.fit(X_train, y_train)
	with open(f"{model}.pkl", 'wb') as f:
		pickle.dump(clf,f)

	resp = jsonify(success=True)
	resp.status_code = 200

	return resp

@mltask_controller.route('/test', methods=['POST'])

def test():
	req_data = request.get_json()
	model = req_data['model_name']
	with open(f"{model}.pkl", "rb") as f:
		clf = pickle.load(f)

	df_test = pd.read_csv('test.csv')
	df_train = pd.read_csv('train.csv')
	classes = list(df_train['label'].unique())
	X_test = df_test[[col for col in df_test.columns if col.startswith('feature')]].to_numpy()
	y_true = df_test['label'].to_numpy()
	X_test = StandardScaler().fit_transform(X_test)
	y_pred = clf.predict(X_test)
	clf_report = classification_report(y_true, y_pred, target_names = ['Class ' + str(i) for i in range(1, len(classes) + 1)])

	resp = jsonify(success=True, text = clf_report)
	resp.status_code = 200

	return resp



@mltask_controller.route('/visualize', methods=['GET'])

def visualize():
	# req_data = request.get_json()
	# model = req_data['model_name']
	# plt.rcParams["figure.figsize"] = [7.50, 3.50]
	# plt.rcParams["figure.autolayout"] = True
	# fig = Figure()
	# with open(f"{model}.pkl", "rb") as f:
	# 	clf = pickle.load(f)
	# classifiers = [clf]
	# df_test = pd.read_csv('test.csv')
	# df_train = pd.read_csv('train.csv')
	# X_test = df_test[[col for col in df_test.columns if col.startswith('feature')]].to_numpy()
	# X_train = df_train[[col for col in df_train.columns if col.startswith('feature')]].to_numpy()
	# print("X_train.shape", X_train.shape)
	# print("X_test.shape", X_test.shape)
	# X = np.concatenate((X_train, X_test), axis=0)
	# X = StandardScaler().fit_transform(X)
	# print("X.shape", X.shape)
	# X_test = StandardScaler().fit_transform(X_test)
	# y_test = df_test['label'].to_numpy()
	# X_train = StandardScaler().fit_transform(X_train)
	# y_train = df_train['label'].to_numpy()
	# x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
	# y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
	# cm = plt.cm.RdBu
	# cm_bright = ListedColormap(["#FF0000", "#0000FF"])
	# ax = fig.add_subplot(1, 2, 1)
	# ax.set_title("Input data")
	# ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
	# # Plot the testing points
	# ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k")
	# ax.set_xlim(x_min, x_max)
	# ax.set_ylim(y_min, y_max)
	# ax.set_xticks(())
	# ax.set_yticks(())
	# ax = plt.subplot(1, 2, 2)
	# score = clf.score(X_test, y_test)
	# DecisionBoundaryDisplay.from_estimator(clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5)

	# # Plot the training points
	# ax.scatter(
	# 	X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
	# )
	# # Plot the testing points
	# ax.scatter(
	# 	X_test[:, 0],
	# 	X_test[:, 1],
	# 	c=y_test,
	# 	cmap=cm_bright,
	# 	edgecolors="k",
	# 	alpha=0.6,
	# )

	# ax.set_xlim(x_min, x_max)
	# ax.set_ylim(y_min, y_max)
	# ax.set_xticks(())
	# ax.set_yticks(())
	# ax.set_title(model)
	# ax.text(
	# 	x_max - 0.3,
	# 	y_min + 0.3,
	# 	("%.2f" % score).lstrip("0"),
	# 	size=15,
	# 	horizontalalignment="right",
	# )
	# output = io.BytesIO()
	# FigureCanvas(fig).print_png(output)
	req_data = request.get_json()
	model = req_data['model_name']
	with open(f"{model}.pkl", "rb") as f:
		clf = pickle.load(f)

	print("test")
	fig = Figure()
	axis = fig.add_subplot(1, 1, 1)
	xs = np.random.rand(100)
	ys = np.random.rand(100)
	axis.plot(xs, ys)
	output = io.BytesIO()
	FigureCanvas(fig).print_png(output)
	return Response(output.getvalue(), mimetype='image/png')
