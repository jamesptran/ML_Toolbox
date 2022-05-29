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
import pickle
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
