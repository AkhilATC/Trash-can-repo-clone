from flask import Blueprint, current_app, request, jsonify
import models

elk_module = Blueprint('elk_module', __name__, url_prefix="/elk")

index_name = "avengers"


@elk_module.route("/", methods=['GET'])
def fetch():
    """

    :return:
    """
    con = models.get_elk_connection_as_es()
    print(current_app.config['ELK_URI'])
    print(con)

    f = con.indices.exists(index="avengers")
    print(f)
    if not f:
        index_data_pattern = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "members": {
                    "properties": {
                        "description": {
                            "type": "text"
                        },
                        "name": {
                            "type": "keyword"
                        },
                        "is_hold_mojiner": {
                            "type": "boolean"
                        },

                }
                }
            }
        }
        e = con.indices.create(index='avengers', ignore=400, body=index_data_pattern)
        print(e)

    return f"avengers index created successfully" if not f else "already exists"


@elk_module.route('/save_doc', methods=['POST'])
def save_avengers():
    con = models.get_elk_connection_as_es()
    try:
        payload = request.json
        if payload:
            res = con.index(index=index_name,  body=payload)
            print(res)
        return jsonify({'message': 'cool'})
    except Exception as e:
        return jsonify({'message':'Failed to upsert'}),400