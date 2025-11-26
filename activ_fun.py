import numpy as np
import matplotlib.pyplot as plt

def softplus(x):
	return np.log(1+np.exp(x))

# d/dx softplus(x)
def softplus_prime(x):
	#Funfact: Softplus_prime=sigmoid
	
	ex=np.exp(x)
	
	return ex/(1+ex)

def sigmoid(x):
	return 1/(1+np.exp(-x))

# d/dx sigmoid(x)
def sigmoid_prime(x):
	ex=np.exp(-x)
	
	return ex/((1+ex)**2)

if __name__ == "__main__":
	
	# Definitionsbereich
	x = np.linspace(-5, 5, 400)
	
	# Funktion
	y = sigmoid(x)
	plt.plot(x, y, label="f(x) = sigmoid")
	
	y=sigmoid_prime(x)
	plt.plot(x, y, label="f'(x) = d/dx sigmoid")
	
	y=softplus(x)
	plt.plot(x, y, label="g(x) = softplus")
	
	y=softplus_prime(x)
	plt.plot(x, y, label="g'(x) = d/dx softplus")
	
	plt.xlabel("x")
	plt.ylabel("f(x)")
	plt.title("")
	plt.legend()
	
	plt.show()
