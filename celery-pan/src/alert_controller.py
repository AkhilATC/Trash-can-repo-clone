from flask import Blueprint, request, jsonify
from src.tasks.demo_tasks import add_two

alert_interface = Blueprint('alert_channel', __name__, url_prefix="/alerts")


@alert_interface.route('/', methods=['GET'])
def fetch_annual_data():
    print("----->>>>>>---")
    task_id = add_two.delay(10, 20)
    print(task_id)
    return jsonify({"message": "cool","task": task_id}), 200