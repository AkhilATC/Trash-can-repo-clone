from flask import Blueprint, request, jsonify
import pandas as pd
from controller.models import Data
from factory import db

data_layer = Blueprint('data_layer', __name__, url_prefix="/data")


@data_layer.route('/', methods=['GET'])
def fetch_annual_data():
    """

    :return:
    """
    well_id = request.args.get('well')
    q_id = request.args.get('q_id')
    # pre procssing stage

    file_ = r'data/input.xls'
    df = pd.read_excel(file_)
    df.columns = [c.replace(' ', '_') for c in df.columns]
    critera = []
    if well_id:
        critera.append('API_WELL__NUMBER')
    if q_id:
        critera.append('QUARTER_1,2,3,4')
    if critera:
        if 'API_WELL__NUMBER' in critera:
            filtered_data = df.loc[df['API_WELL__NUMBER'] == int(well_id)]
        if 'QUARTER_1,2,3,4' in critera:
            print("inside")
            filtered_data = filtered_data.loc[df['QUARTER_1,2,3,4'] == int(q_id)]

        oil_sum = int(filtered_data['OIL'].sum()) if filtered_data['OIL'].sum() else 0
        gas_sum = int(filtered_data['GAS'].sum()) if filtered_data['GAS'].sum() else 0
        brine_sum = int(filtered_data['BRINE'].sum()) if filtered_data['BRINE'].sum() else 0

        # check key exists or not

        if not bool(Data.query.filter_by(id=int(well_id)).first()):
            db_info = Data(id=int(well_id), oil=oil_sum, gas=gas_sum, brine=brine_sum)
            db.session.add(db_info)
            db.session.commit()

        return jsonify({
                'oil': oil_sum,
                'gas': gas_sum,
                'brine': brine_sum
            })
    return jsonify({'message': "can't fetch data invalid inputs"}), 400


@data_layer.route('/get_db_info', methods=['GET'])
def fetch_db():
    """

    :return:
    """
    try:
        all_nodes = Data.query.all()
        result = [d.__dict__ for d in all_nodes]
        result = [{
            'id': x['id'],
            'OIL': x['oil'],
            'GAS': x['gas'],
            'BRINE': x['brine']} for x in result]
        return jsonify(result), 200
    except Exception as e:
        print(e)
        return jsonify({'message':'Failed to fetch records'}), 400