from factory import create_app

if __name__ == '__main__':
    f = create_app()
    f.run(host="0.0.0.0", port="7001", debug=True)