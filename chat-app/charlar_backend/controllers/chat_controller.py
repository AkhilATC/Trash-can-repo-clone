from flask import Blueprint, jsonify, request, Response
from flask_jwt_extended import get_jwt_identity
from flask_jwt_extended import jwt_required
from datetime import datetime
from factory import mongo_db
from json import dumps
from bson.objectid import ObjectId


chat_module = Blueprint("chat_module", __name__, url_prefix="/chat")


@jwt_required()
@chat_module.route("/", methods=['GET'])
def chat_app_info():
    """
    discription: ping route for chat controller
    :return:
    """
    return jsonify({
        "app-name": "Charlar",
        "contoller": "chat",
        "info": __name__
    }), 200


@chat_module.route("/publish_my_note", methods=['POST'])
@jwt_required()
def publish_post():
    """

    :return:
    """
    try:
        post_info = request.json
        if not all([post_info[x] for x in post_info]):
            return jsonify({'message': 'mandatory fields missing'}), 409
        current_user = get_jwt_identity()
        time_now = datetime.utcnow()
        post_info.update({'wrote_by': current_user, 'created_on':time_now})
        mongo_db.db.chats.insert_one(post_info)
        return jsonify({'message': f"{current_user}'s post saved successfully"}), 201
    except Exception as e:
        print(e)
        return jsonify({'message':'Failed'}), 400


@chat_module.route("/fetch_posts", methods=['GET'])
@jwt_required()
def fetch_posts():
    """

    :return:
    """
    try:
        aggs_query = [
            {
                "$match": {
                    "thought": {
                        "$exists": True
                    }
                }
            },
            {
                "$project": {
                    "thought": 1,
                    "wrote_by": 1,
                    "created_on": 1,
                    "title": 1,
                    "is_self_thought": {
                        "$cond": [
                            {
                                "$eq": [
                                    "$wrote_by",
                                     get_jwt_identity() or ""
                                ]
                            },
                            True,
                            False
                        ]
                    }
                }
            }
        ]
        data = list(mongo_db.db.chats.aggregate(aggs_query))
        return Response(dumps(data, default=str), mimetype='application/json')
    except Exception as e:
        print(e)
        return jsonify({'message': 'Failed'}), 500


@chat_module.route("/delete/<id>", methods=['DELETE'])
@jwt_required()
def delete(id):
    """

    :return:
    """
    try:
        record = mongo_db.db.chats.find_one({"_id": ObjectId(id)})
        if record:
            mongo_db.db.chats.delete_one({"_id": ObjectId(id)})
        return jsonify({"message": "Deleted"}), 200
    except Exception as e:
        return jsonify({'message': "failed"}), 500
