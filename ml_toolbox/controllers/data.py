from flask import Blueprint
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

auth_controller = Blueprint('data', __name__)

@auth_controller.route('/download', methods=['POST'])

def download():

	req_data = request.get_json()
	train_tok = req_data['train_folder']
    	gauth = GoogleAuth()      
    	drive = GoogleDrive(gauth) 
	test_tok = req_data['test_folder']
	train_list = drive.ListFile({'q': f"{train_tok} in parents and trashed=false"}).GetList()
	test_list = drive.ListFile({'q': f"{test_tok} in parents and trashed=false"}).GetList()
	for file in train_list:
    	print('title: %s, id: %s'% (file['title'],file['id']))
    	file.GetContentFile(file['title'])

    for file in test_list:
    	print('title: %s, id: %s'% (file['title'],file['id']))
    	file.GetContentFile(file['title'])

    resp = jsonify(success=True)
	resp.status_code = 200
	return resp


@auth_controller.route('/upload', methods=['POST'])
def upload():
	
    resp = jsonify(success=True)
	resp.status_code = 200
	return resp
