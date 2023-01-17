import language

while True:
		text = input('bos > ')
		result, error = language.run('<stdin>', text)

		if error: print(error.as_string())
		elif result: print(result)