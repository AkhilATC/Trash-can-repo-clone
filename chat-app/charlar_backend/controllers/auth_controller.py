from factory import mongo_db
from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token

auth_module = Blueprint("auth_module", __name__, url_prefix="/auth")


@auth_module.route("/", methods=['GET'])
def auth_app_info():
    """
    discription: ping route for chat controller
    :return:
    """
    return jsonify({
        "app-name": r"Charlar",
        "contoller": "Auth",
        "info": "api -testing"
    }), 200


@auth_module.route("/sign_up", methods=['POST'])
def sign_up():
    """

    :return:
    """
    payload = request.json
    if all([payload.get(x,None) for x in ['name', 'first_name', 'last_name', 'password']]):
        is_exists = list(mongo_db.db.users.find({'name': payload["name"]}))
        if is_exists:
            return jsonify({"message": "user already exists , try another email or password"}),404
        mongo_db.db.users.insert_one(payload)
        return jsonify({'message': 'New user added successfully'}), 201
    return jsonify({'message': 'Required info\'s missing'}), 404


@auth_module.route("login", methods=['POST'])
def log_in():
    """

    :return:
    """
    username = request.json.get("username", None)
    password = request.json.get("password", None)
    print(username)
    is_authenticate = mongo_db.db.users.find_one({'name': username, 'password': password})
    if not is_authenticate:
        return jsonify({'message': 'user-name or password is incorrect'}), 401
    access_token = create_access_token(identity=username)
    return jsonify({'user': username, 'token': access_token}), 200
