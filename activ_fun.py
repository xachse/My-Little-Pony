import numpy as np
import matplotlib.pyplot as plt
import sklearn

class Network():

    def __init__(self, sizes=[64, 64, 32, 3]):
        '''
        Diese Funktion initilaiisert zufällig, gleichverteilte Gewichte und Biases für das Netzwerk"
        mit gegebenen Anzahlen der Neuronen pro Schicht. In unserem Fall sind dies 28^2 Input-Neuronen,
        64 in der zweiten Layer, 32 in der dritten und schließlich 3 in der letzten Layer.
        Anschließend werden die Test- und Trainingsdaten importiert und vorbereitet.
        '''
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = []
        for i in range(1, len(sizes)):
            self.biases.append(np.random.randn(sizes[i], 1))
        self.weights = []
        for i in range(len(sizes) - 1):
            self.weights.append(np.random.randn(sizes[i + 1], sizes[i])) 

        # Nun werden die Test- und Trainingsdaten importiert

        digits = sklearn.load_digits()
        X = digits.data  # Matrix mit 1797 Zeilen und 64 Spalten (8x8 Bilder)    
        y = digits.target  # Labels (Zahlen von 0 bis 9)
        mask = np.isin(y, [1, 5, 7])
        X = X[mask]
        y = y[mask]
        label_map = {1: 0, 5: 1, 7: 2}
        y = np.array([label_map[label] for label in y])  # Labels werden umgewandelt zu 0, 1, 2
        X_train, X_test, y_train, y_test = sklearn.train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # Split 
        scaler = sklearn.StandardScaler()  # Skalierung 
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        X_train = [x.reshape(64, 1) for x in X_train] # zu Spaltenvektoren umformen
        X_test  = [x.reshape(64, 1) for x in X_test]

        training_data = list(zip(X_train, y_train)) # Liste von Tupeln (x, y), wobei x der Input (ein Spaltenvektor) und y (die zugehörige Zahl zu x) die korrekte Ausgabe ist.
        test_data = list(zip(X_test, y_test))

    def sigmoid(self, z):
        '''
        Die Sigmoid-Aktivierungsfunktion, die jede gewichtete Summe in einen Wert zwischen 0 und 1 umwandelt.
        '''
        return 1.0/(1.0+np.exp(-z))
    

    def sigmoid_prime(self, z):
        '''
        Die Ableitung der Sigmoid-Funktion.
        '''
        return self.sigmoid(z)*(1-self.sigmoid(z))


    def one_hot_encode(self, j):
        '''
        Diese Funktion wandelt eine Zahl j in einen One-Hot-Vektor um.
        '''
        e = np.zeros((self.sizes[-1], 1))
        e[j] = 1.0
        return e
    

    def feedforward(self, a):
        '''
        Diese Funktion wertet das Netzwerk aus, abhängig vom Input 'a' (ein Spaltenvektor).
        '''
        for i in range(len(self.sizes) - 1):
                a = self.sigmoid(self.weights[i]@a + self.biases[i])
        return a
    

    def evaluate(self, test_data):
        '''
        Diese Funktion wertet die Leistung des Netzwerks anhand der Testdaten aus. test_data ist eine Liste von Tupeln
        (x, y), wobei x der Input (ein Spaltenvektor) und y (die zugehörige Zahl zu x) die korrekte Ausgabe ist.
        Die argmax-Funktion gibt den Index des höchsten Wertes im Ausgabevektor zurück, also dass was das Netzwerk
        denkt die richtige Zahl zum Input x sei. Die Funktion zählt, wie oft das Netzwerk richtig lag und gibt 
        diese Zahl zurück.
        '''
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)



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
