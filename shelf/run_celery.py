from factory import make_app, make_celery
import redis
import os

app = make_app()
app.app_context().push()
app.config["REDIS_PASSWORD"] = os.environ.get('REDIS_PASSWORD')
celery = make_celery(app)
