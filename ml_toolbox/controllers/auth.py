from flask import Blueprint, request, g, jsonify, abort, current_app
from mongoengine import NotUniqueError
import hashlib
import hmac
from flask_security.utils import login_user
from flask_security import auth_token_required, roles_required, current_user


auth_controller = Blueprint('auth', __name__)

@auth_controller.route('/register', methods=['POST'])
def register():
    req_data = request.get_json()

    email = req_data['email']
    password = req_data['ID_token']

    u = User(email=email, password=password, active=True, roles=[userRole])
    try:
        u.save()
    except NotUniqueError:
        abort(HTTPStatus.CONFLICT)

    return jsonify(u)



@auth_controller.route('/login', methods=['POST'])
def login():
    req_data = request.get_json()

    email = req_data['email']
    password = req_data['password']

    user = User.objects.get(email=email)
    if(user.password != password):
        abort(HTTPStatus.BAD_REQUEST)

    login_user(user)

    response = {
        'authentication_token': user.get_auth_token(),
        'id': str(user.id),
        'roles': user.getRolesName()
    }

    return jsonify(response)