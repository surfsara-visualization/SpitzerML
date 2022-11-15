class Test():
	def __init__(self, a):
		print('a', a)

	def __call__(self,b):
		print('b', b)


t = Test(1)
t(2)