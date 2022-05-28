from flask import Blueprint
from flask import Blueprint, request, g, jsonify, abort, current_app

mltask_controller = Blueprint('ml_task', __name__)

@mltask_controller.route('/train', methods=['POST'])

def train():
	req_data = request.get_json()
	print(req_data)
	resp = jsonify(success=True)
	resp.status_code = 200
	return resp