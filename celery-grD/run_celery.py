from factory import make_app, make_celery


app = make_app()
app.app_context().push()

celery = make_celery(app)