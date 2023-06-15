from factory import celery


@celery.task()
def add_two(x, y):
    print("cool::: GOT CALL FROM CONTROLLER ")
    print(x+y)
    return x + y
