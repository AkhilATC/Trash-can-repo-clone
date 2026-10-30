from factory import make_app  # make_celery
from flask_cors import CORS

# app = make_app()
# CORS(app, resources={r"/*": {"origins": "*"}})
# app.run(host="0.0.0.0", port=5500, debug=True)
# # celery = make_celery(app)
if __name__ == '__main__':
    app = make_app()
    CORS(app, resources={r"/*": {"origins": "*"}})
    app.run(host="0.0.0.0", port=5500, debug=True)
else:
    app = make_app()
    CORS(app, resources={r"/*": {"origins": "*"}})
