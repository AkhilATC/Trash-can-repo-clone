from factory import make_app

app = make_app()
app.run(host="0.0.0.0", port=5500, debug=True)