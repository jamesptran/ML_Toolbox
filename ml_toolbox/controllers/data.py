from flask import Blueprint, request, g, jsonify, abort, current_app
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

data_controller = Blueprint('data', __name__)

@data_controller.route('/download', methods=['POST'])

def download():

	req_data = request.get_json()
	data_tok = req_data['data_folder']
	gauth = GoogleAuth()
	drive = GoogleDrive(gauth)
	data_list = drive.ListFile({'q': f"{data_tok} in parents and trashed=false"}).GetList()
	for file in data_list:
		print('title: %s, id: %s'% (file['title'],file['id']))
		file.GetContentFile(file['title'])
	resp = jsonify(success=True)
	resp.status_code = 200
	return resp


@data_controller.route('/upload', methods=['POST'])
def upload():

	resp = jsonify(success=True)
	resp.status_code = 200
	return resp
