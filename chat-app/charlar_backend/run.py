from factory import setup_flask_app

app = setup_flask_app()
app.run(host="0.0.0.0", port="5900", debug=True)