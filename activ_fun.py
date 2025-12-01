import numpy as np
import sklearn

class Network():

    def __init__(self, dataset='mnist', activation_mode="sigmoid"):
        '''
        Diese Funktion initilaiisert zufällig, gleichverteilte Gewichte und Biases für das Netzwerk
        mit gegebenen Anzahlen der Neuronen pro Schicht. In unserem Fall sind dies 64/28^2 Input-Neuronen,
        64 in der zweiten Layer, 32 in der dritten und schließlich 3 in der letzten Layer.
        Anschließend werden die Test- und Trainingsdaten importiert und vorbereitet.
        '''
        self.biases = []
        self.weights = []

        self.activation_mode = activation_mode

        if dataset == "digits":
            self.sizes = [64, 64, 32, 3]
            """self.biases = [np.random.randn(self.sizes[i + 1], 1) for i in range(3)]
            self.weights = [np.random.randn(self.sizes[i + 1], self.sizes[i]) for i in range(3)]"""
            self.biases = [np.zeros((y, 1)) for y in self.sizes[1:]]  # im Fall von vollem MNIST-Traning mit wurzel(1/x) initialisieren -> verringert Varianz
            self.weights = [np.random.randn(y, x) * np.sqrt(1 / x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
            self.load_digits()
        elif dataset == "mnist":
            self.sizes = [784, 64, 32, 3]
            self.biases = [np.zeros((y, 1)) for y in self.sizes[1:]]  # im Fall von vollem MNIST-Traning mit wurzel(1/x) initialisieren -> verringert Varianz
            self.weights = [np.random.randn(y, x) * np.sqrt(1 / x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
            self.load_mnist()
        else:
            raise ValueError("Unbekannter Datensatz")


    def load_digits(self):
        digits = sklearn.datasets.load_digits()
        X = digits.data  # Matrix mit 1797 Zeilen (Anzahl der Bilder, nach Dokumentation) und 64 Spalten (8x8 Bilder)    
        y = digits.target  # Labels (Zahlen von 0 bis 9)
        mask = np.isin(y, [1, 5, 7])
        X = X[mask]
        y = y[mask]
        label_map = {1: 0, 5: 1, 7: 2}
        y = np.array([label_map[label] for label in y])  # Labels werden umgewandelt zu 0, 1, 2
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # Split 
        scaler = sklearn.preprocessing.StandardScaler()  # Skalierung 
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        X_train = [x.reshape(64, 1) for x in X_train] # zu Spaltenvektoren umformen
        X_test  = [x.reshape(64, 1) for x in X_test]

        training_data = list(zip(X_train, y_train)) # Liste von Tupeln (x, y), wobei x der Input (ein Spaltenvektor) und y (die zugehörige Zahl zu x) die korrekte Ausgabe ist.
        test_data = list(zip(X_test, y_test))

        self.training_data = training_data
        self.test_data = test_data

    
    def load_mnist(self):
        mnist = sklearn.datasets.fetch_openml("mnist_784", version=1, as_frame=False)
        X = mnist["data"]
        y = mnist["target"].astype(int)

        mask = np.isin(y, [1,5,7])
        X = X[mask]
        y = y[mask]

        label_map = {1:0, 5:1, 7:2}
        y = np.array([label_map[k] for k in y])

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = [x.reshape(784,1) for x in X_train]
        X_test = [x.reshape(784,1) for x in X_test]

        self.training_data = list(zip(X_train, y_train))
        self.test_data = list(zip(X_test, y_test))


    # ---------------------------------------------------------
    # Aktivierungsfunktionen abhängig vom Mode
    # ---------------------------------------------------------

    def activation(self, x):
        if self.activation_mode == "sigmoid":
            return self.sigmoid(x)
        elif self.activation_mode == "softplus":
            return self.softplus(x)
        else:
            raise ValueError("Unbekannter activation_mode")

    def activation_prime(self, x):
        if self.activation_mode == "sigmoid":
            return self.sigmoid_prime(x)
        elif self.activation_mode == "softplus":
            return self.sigmoid(x)
        else:
            raise ValueError("Unbekannter activation_mode")

    def output_activation(self, x):
        if self.activation_mode == "sigmoid":
            return self.sigmoid(x)
        elif self.activation_mode == "softplus":
            return x  
        else:
            raise ValueError("Unbekannter activation_mode")

    def output_activation_prime(self, x):
        if self.activation_mode == "sigmoid":
            return self.sigmoid_prime(x)
        elif self.activation_mode == "softplus":
            return np.ones_like(x)  
        else:
            raise ValueError("Unbekannter activation_mode")

    
    def one_hot_encode(self, j):
        '''
        Diese Funktion wandelt eine Zahl j in einen One-Hot-Vektor um.
        '''
        e = np.zeros((self.sizes[-1], 1))
        e[j] = 1.0
        return e
    

    def forward(self, a):
        '''
        Diese Funktion wertet das Netzwerk aus, abhängig vom Input 'a' (ein Spaltenvektor) und gibt die
        Aktivierungen aller Layer sowie die z-Werte (W@a + b) zurück.
        '''
        activations = [a]  # Liste aller a
        zs = []             # Liste aller z = W @ a + b // werden für Backpropagation benötigt

        for i in range(len(self.weights)):
            z = self.weights[i] @ a + self.biases[i]
            zs.append(z)
            if i == len(self.weights) - 1:
                a = self.output_activation(z)
            else:
                a = self.activation(z)
            activations.append(a)

        return activations, zs
    

    def evaluate(self, test_data):
        '''
        Diese Funktion wertet die Leistung des Netzwerks anhand der Testdaten aus. test_data ist eine Liste von Tupeln
        (x, y), wobei x der Input (ein Spaltenvektor) und y (die zugehörige Zahl zu x) die korrekte Ausgabe ist.
        Die argmax-Funktion gibt den Index des höchsten Wertes im Ausgabevektor zurück, also dass was das Netzwerk
        denkt die richtige Zahl zum Input x sei. Die Funktion zählt, wie oft das Netzwerk richtig lag und gibt 
        diese Zahl zurück.
        '''
        test_results = []
        for (x, y) in test_data:
            activations, _ = self.forward(x)
            pred = np.argmax(activations[-1])  # letzte Aktivierung (Output-Layer)
            test_results.append((pred, y))
        return sum(int(x == y) for (x, y) in test_results), test_results
    

    # Stochastic Gradient Descent (SGD) Methode von Michael Nielsen's Buch "Neural Networks and Deep Learning"

    def SGD(self, training_data, epochs, mini_batch_size, eta, loss="mse"):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        n = len(training_data)
        for j in range(epochs):
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, loss)

            for x,y in training_data:
                activations, _ = self.forward(x)
                if loss == "mse":
                    current_loss = self.mse_loss(activations[-1], self.one_hot_encode(y))
                elif loss == "ce":
                    current_loss = self.cross_entropy_loss(activations[-1], y)
                else:
                    raise ValueError("Loss muss 'mse' oder 'ce' sein")

            progress=round(j/epochs*100,1)
            
            print(f"Training progress: {progress} %    ", end="\r")


    def update_mini_batch(self, mini_batch, eta, loss="mse"):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y, loss)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
    
    # Stochastic Gradient Descent (SGD) Methode von Michael Nielsen's Buch "Neural Networks and Deep Learning"


    def backprop(self, x, y_true, loss="mse"):
        """
        Berechnet die Gradienten der Gewichte und Biases für EIN Trainingsbeispiel (x, y_true).
        Rückgabe:
        nabla_w: Liste gleicher Form wie self.weights
        nabla_b: Liste gleicher Form wie self.biases
        """

        #  Forward: Aktivierungen und z-Werte speichern 
        activations, zs = self.forward(x)   

        #  Speicher vorbereiten 
        nabla_w = [np.zeros_like(w) for w in self.weights]
        nabla_b = [np.zeros_like(b) for b in self.biases]

        #  One-Hot Label 
        y_vec = self.one_hot_encode(y_true)

        #  Backprop für die Output Layer 

        if loss == "mse":
            delta = (activations[-1] - y_vec) * self.output_activation_prime(zs[-1])
        elif loss == "ce": 
            delta = activations[-1] - y_vec            
        else:
            raise ValueError("Loss muss 'mse' oder 'ce' sein")

        nabla_b[-1] = delta
        nabla_w[-1] = delta @ activations[-2].T

        #  Backprop für alle Hidden Layers
        for l in range(2, len(self.sizes)):

            delta = (self.weights[-l + 1].T @ delta) * self.activation_prime(zs[-l])

            nabla_b[-l] = delta
            nabla_w[-l] = delta @ activations[-l-1].T

        return nabla_w, nabla_b
    

    def update_params(self, nabla_w, nabla_b, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * nabla_w[i] / len(self.training_data)
            self.biases[i]  -= lr * nabla_b[i] / len(self.training_data)
    

    def train_full_batch(self, training_data, epochs=50, lr=0.1, loss="mse"):

        for epoch in range(epochs):

            sum_nabla_w = [np.zeros_like(w) for w in self.weights]
            sum_nabla_b = [np.zeros_like(b) for b in self.biases]

            for x, y in training_data:  # berechne Gradienten für jedes Trainingspunkt 
                nabla_w, nabla_b = self.backprop(x, y, loss)

                for i in range(len(self.weights)):  # Gradienten aufsummieren, Mittelwert bilden ist nicht nötig 
                    sum_nabla_w[i] += nabla_w[i]
                    sum_nabla_b[i] += nabla_b[i]

                activations, _ = self.forward(x)
                if loss == "mse":
                    current_loss = self.mse_loss(activations[-1], self.one_hot_encode(y))
                elif loss == "ce":
                    current_loss = self.cross_entropy_loss(activations[-1], y)
                else:
                    raise ValueError("Loss muss 'mse' oder 'ce' sein")

            self.update_params(sum_nabla_w, sum_nabla_b, lr)

            progress=round(epoch/epochs*100,1)

            print(f"Training progress: {progress} %    ", end="\r")


    # Loss-Funktionen, so wie die Aktivierungsfunktionen und deren Ableitungen

    def mse_loss(self, y_pred, y_true):
        return 0.5 * np.sum((y_pred - y_true)**2)
    
    def cross_entropy_loss(self, y_pred, y_true):
        logits_shift = y_pred - np.max(y_pred)
        exp_logits = np.exp(logits_shift)
        softmax = exp_logits / np.sum(exp_logits)

        return -np.log(softmax[y_true])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))
    
    def sigmoid_prime(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def softplus(self, x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    def softplus_prime(self, x):
        return self.sigmoid(x)
    
    def print_confusion_matrix(self, classes=[1,5,7]):
	
        ergebnis=self.evaluate(self.test_data)[1]
        n=len(classes)
        m=len(ergebnis)
        
        confusion_matrix=np.zeros((n, n), dtype=int)
        
        for (x,y) in ergebnis:
            confusion_matrix[y,x]+=1
        
        string="Real\t|Predicted\n\t"
        
        for i in range(n):
            string+="|"+str(classes[i])+"\t"
        string+="\n-"
        
        for i in range(n):
            string+="-------+"
        string+="-------\n"
        
        
        for i in range(n):
            string+=str(classes[i])+"\t"
            for j in range(n):
                string+="|"+str(confusion_matrix[i,j])+"\t"
            string+="\n\t"
            for j in range(n):
                percentage=round(confusion_matrix[i,j]/m*100, 1)
                
                string+="|"+str(percentage)+" %\t"
            string+="\n-"
            for i in range(n):
                string+="-------+"
            string+="-------\n"
        
        return string 
    


if __name__ == "__main__":
    print("=== Einstellungen für das Neuronale Netz ===")

    # ------------------------------
    # 1. Datensatz auswählen
    # ------------------------------
    while True:
        print("\n1) MNIST (28x28) - entspricht 784 Input-Neuronen und etwa 16.000 Trainingsbeispielen.")
        print("2) Digits (8x8) - entspricht 64 Input-Neuronen und etwa 400 Trainingsbeispielen.")
        ds_choice = input("Auf welchem Datensatz soll trainiert werden? (1/2): ")

        if ds_choice == "1":
            dataset = "mnist"
            break
        elif ds_choice == "2":
            dataset = "digits"
            break
        else:
            print("Ungültige Eingabe! Bitte 1 oder 2 eingeben.")

    if dataset == "digits":
        while True:
            print("\nWelchen Trainingsmodus möchtest du verwenden?")
            print("1) Full Batch Gradient Descent")
            print("2) Mini Batch (SGD)")
            mode_choice = input("Bitte wählen (1/2): ")

            if mode_choice == "1":
                train_mode = "full_batch"
                break
            elif mode_choice == "2":
                train_mode = "mini_batch"
                break
            else:
                print("Ungültige Eingabe! Bitte 1 oder 2 eingeben.")
    else:
        # MNIST → immer mini-batch
        train_mode = "mini_batch"
        print("\nHinweis: MNIST ist zu groß für Full Batch Gradient Descent, daher wird Stochastic Gradient Descent automatisch verwendet.")

    # ------------------------------
    # 2. Aktivierungsfunktion wählen
    # ------------------------------
    while True:
        print("\nWelche Aktivierungsfunktion soll verwendet werden?")
        print("1) Sigmoid (dann wird automatisch MSE als Loss verwendet, weil Cross-Entropy )")
        print("2) Softplus")
        act_choice = input("Bitte wählen (1/2): ")

        if act_choice == "1":
            activation = "sigmoid"
            loss = "mse"  
            break

        elif act_choice == "2":
            activation = "softplus"

            # ------------------------------
            # 3. Loss wählen
            # ------------------------------
            while True:
                print("\nWelcher Loss soll verwendet werden?")
                print("1) MSE")
                print("2) Cross-Entropy")
                loss_choice = input("Bitte wählen (1/2): ")

                if loss_choice == "1":
                    loss = "mse"
                    break
                elif loss_choice == "2":
                    loss = "ce"
                    break
                else:
                    print("Ungültige Eingabe, bitte 1 oder 2 eingeben.")

            break

        else:
            print("Ungültige Eingabe! Bitte 1 oder 2 eingeben.")

    print("\n=== Zusammenfassung der Auswahl ===")
    print("Datensatz:        ", dataset)
    print("Trainingsmodus:   ", train_mode)
    print("Aktivierung:      ", activation)
    print("Lossfunktion:     ", loss)
    print("=====================================\n")

    net = Network(dataset=dataset, activation_mode=activation) 

    if train_mode == "full_batch" and activation == "softplus":
        net.train_full_batch(net.training_data, epochs=50, lr=0.02)
    elif train_mode == "full_batch" and activation == "sigmoid":
        net.train_full_batch(net.training_data, epochs=50, lr=1.4)
    else:
        net.SGD(net.training_data,
                epochs=30,
                mini_batch_size=30,
                eta=0.1)

    accuracy = net.evaluate(net.test_data) 
    percentage = accuracy[0] / len(net.test_data) * 100
    print(f"Genauigkeit nach SGD: {accuracy[0]} von {len(net.test_data)} Testbeispielen korrekt klassifiziert ({percentage:.2f}%).")
    print(net.print_confusion_matrix(classes=[0,1,2]))


