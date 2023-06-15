
def get_elk_connection_as_es():

    from factory import es
    if es is None:
        print(es)
        raise Exception("Failed to connect es")
    return es

