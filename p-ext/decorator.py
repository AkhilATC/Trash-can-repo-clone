
# simple decorator function
def simple_decorator(func):
	
	def wraper_fn():
		print("INSIDE WRAPER FUNCTION")
		func()
		print("AFTER WRAPER FUNCTION")
	return wraper_fn

@simple_decorator
def c():
	print("KI")


c()


