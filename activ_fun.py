import numpy as np
import sklearn
import os       #Stellt ein, dass das Terminal farbe unterstützt
import sys      #um dem Programm Argumente zu übergeben

os.system("")

class Network():

    def __init__(self, dataset='mnist', activation_mode="sigmoid"):
        '''
        Diese Funktion verwendet die Glorot-Initiliaiserung, um zufällige, gleichverteilte Gewichte mit geringer Varianz für das Netzwerk
        zu erstellen, mit gegebenen Anzahlen der Neuronen pro Schicht. In unserem Fall sind dies 64/28^2 Input-Neuronen,
        64 in der zweiten Layer, 32 in der dritten und schließlich 3 in der letzten Layer.
        Anschließend werden die Test- und Trainingsdaten importiert und vorbereitet.
        '''
        self.biases = []
        self.weights = []

        self.activation_mode = activation_mode

        self.loss_training=[]
        self.loss_test=[]

        if dataset == "digits":
            self.sizes = [64, 64, 32, 3]
            self.biases = [np.zeros((y, 1)) for y in self.sizes[1:]]  
            self.weights = [np.random.randn(y, x) * np.sqrt(1 / x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
            self.load_digits()
        elif dataset == "mnist":
            self.sizes = [784, 64, 32, 3]
            self.biases = [np.zeros((y, 1)) for y in self.sizes[1:]]  
            self.weights = [np.random.randn(y, x) * np.sqrt(1 / x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
            self.load_mnist()


    def load_digits(self):
        """
        Diese Funktion lädt den Digits-Datensatz aus sklearn.datasets, filtert die Bilder der Ziffern 1, 5 und 7 heraus
        (1, 5 und 7 werden zu 0, 1 und 2), teilt die Daten in Trainings- und Testdaten auf und wandelt die Daten in Vektoren um.
        """
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
        """
        Diese Funktion lädt den MNIST-Datensatz aus sklearn.datasets, filtert die Bilder der Ziffern 1, 5 und 7 heraus
        (1, 5 und 7 werden zu 0, 1 und 2), teilt die Daten in Trainings- und Testdaten auf und wandelt die Daten in Vektoren um.
        """
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
    # Aktivierungsfunktionen abhängig von Einstellung
    # ---------------------------------------------------------

    def activation(self, x):
        """
        Diese Funktion gibt die Aktivierungsfunktion zurück. 
        """
        if self.activation_mode == "sigmoid":
            return self.sigmoid(x)
        elif self.activation_mode == "softplus":
            return self.softplus(x)

    def activation_prime(self, x):
        """
        Diese Funktion gibt die Ableitung der Aktivierungsfunktion zurück.
        Dies wird für Backpropagation benötigt.
        """
        if self.activation_mode == "sigmoid":
            return self.sigmoid_prime(x)
        elif self.activation_mode == "softplus":
            return self.sigmoid(x)

    def output_activation(self, x):
        """
        Diese Funktion gibt die Aktivierungsfunktion für den Output-Layer zurück. 
        Denn im Falle von softplus soll im Output-Layer die Identitätsfunktion verwendet werden.
        """
        if self.activation_mode == "sigmoid":
            return self.sigmoid(x)
        elif self.activation_mode == "softplus":
            return x  

    def output_activation_prime(self, x):
        """
        Diese Funktion gibt die Ableitung der Aktivierungsfunktion für den Output-Layer zurück.
        Dies wird für Backpropagation benötigt.
        """
        if self.activation_mode == "sigmoid":
            return self.sigmoid_prime(x)
        elif self.activation_mode == "softplus":
            return np.ones_like(x)  

    
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
        Aktivierungen aller Layer sowie die z-Werte (W @ a + b) zurück.
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
    def SGD(self, training_data, test_data, epochs, mini_batch_size, eta, loss="mse"):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        self.loss_training=np.zeros(epochs)
        self.loss_test=np.zeros(epochs)
        
        n = len(training_data)
        for j in range(epochs):
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, loss)

            for x, y in self.training_data:  # berechne Gradienten für jedes Trainingspunkt 
                nabla_w, nabla_b = self.backprop(x, y, loss)

                for i in range(len(self.weights)):  # Gradienten aufsummieren
                    sum_nabla_w[i] += nabla_w[i]
                    sum_nabla_b[i] += nabla_b[i]

            self.update_params(sum_nabla_w, sum_nabla_b, lr)
            
            sum_loss = 0
            
            for x,y in self.training_data:
                activations, _ = self.forward(x)
                if loss == "mse":
                    current_loss = self.mse_loss(activations[-1], self.one_hot_encode(y))
                elif loss == "ce":
                    current_loss = self.cross_entropy_loss(activations[-1], y)
                sum_loss += current_loss
            
            self.loss_training[epoch]=sum_loss / len(training_data)
            
            sum_loss = 0
            
            for x,y in self.test_data:
                activations, _ = self.forward(x)
                if loss == "mse":
                    current_loss = self.mse_loss(activations[-1], self.one_hot_encode(y))
                elif loss == "ce":
                    current_loss = self.cross_entropy_loss(activations[-1], y)
                sum_loss += current_loss
                     
            self.loss_test[epoch]=sum_loss / len(self.test_data)
            
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
        Berechnet die Gradienten der Gewichte und Biases für ein Trainingsbeispiel (x, y_true) und gibt 
        die Listen nabla_w und nabla_b zurück.
        nabla_w: Partielle Ableitungen der Gewichte in Liste gleicher Form wie self.weights.
        nabla_b: Partielle Ableitungen der Biases in Liste gleicher Form wie self.biases.
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

        nabla_b[-1] = delta
        nabla_w[-1] = delta @ activations[-2].T

        #  Backprop für alle Hidden Layers
        for l in range(2, len(self.sizes)):

            delta = (self.weights[-l + 1].T @ delta) * self.activation_prime(zs[-l])

            nabla_b[-l] = delta
            nabla_w[-l] = delta @ activations[-l-1].T

        return nabla_w, nabla_b
    
    def update_params(self, nabla_w, nabla_b, lr):
        """
        Diese Funktion aktualisiert die Gewichte und Biases des Netzwerks basierend auf den übergebenen Gradienten
        und der Lernrate lr.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= lr * nabla_w[i] / len(self.training_data)
            self.biases[i]  -= lr * nabla_b[i] / len(self.training_data)
    
    def train_full_batch(self, epochs=50, lr=0.1, loss="mse"):
        """
        Diese Funktion trainiert das Netzwerk mit Full Batch Gradient Descent. 
        In jeder Epoche werden die Gradienten für alle Trainingsdaten berechnet und aufsummiert. 
        """
        
        self.loss_training=np.zeros(epochs)
        self.loss_test=np.zeros(epochs)
        
        for epoch in range(epochs):

            sum_nabla_w = [np.zeros_like(w) for w in self.weights]
            sum_nabla_b = [np.zeros_like(b) for b in self.biases]

            for x, y in self.training_data:  # berechne Gradienten für jedes Trainingspunkt 
                nabla_w, nabla_b = self.backprop(x, y, loss)

                for i in range(len(self.weights)):  # Gradienten aufsummieren
                    sum_nabla_w[i] += nabla_w[i]
                    sum_nabla_b[i] += nabla_b[i]

            self.update_params(sum_nabla_w, sum_nabla_b, lr)
            
            sum_loss = 0
            
            for x,y in self.training_data:
                activations, _ = self.forward(x)
                if loss == "mse":
                    current_loss = self.mse_loss(activations[-1], self.one_hot_encode(y))
                elif loss == "ce":
                    current_loss = self.cross_entropy_loss(activations[-1], y)
                sum_loss += current_loss
            
            self.loss_training[epoch]=sum_loss / len(self.training_data)
            
            sum_loss = 0
            
            for x,y in self.test_data:
                activations, _ = self.forward(x)
                if loss == "mse":
                    current_loss = self.mse_loss(activations[-1], self.one_hot_encode(y))
                elif loss == "ce":
                    current_loss = self.cross_entropy_loss(activations[-1], y)
                sum_loss += current_loss
            
            self.loss_test[epoch]=sum_loss / len(self.test_data)
            
            progress=round(epoch/epochs*100,1)

            print(f"Training progress: {progress} %    ", end="\r")

    # Loss-Funktionen, sowie die Aktivierungsfunktionen und deren Ableitungen:
    def mse_loss(self, y_pred, y_true):
        return 0.5 * np.sum((y_pred - y_true)**2)
    
    def cross_entropy_loss(logits, y_true):
        logits_shift = logits - np.max(logits)
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
    
    # ------------------------------
    # output to the terminal
    # ------------------------------
    
    def print_confusion_matrix(self, classes=[1,5,7]):
        ergebnis=self.evaluate(self.test_data)[1]
        n=len(classes)
        
        #initialize confusion matrix
        #and list of cardinality of every class
        confusion_matrix=np.zeros((n, n), dtype=int)
        targets_number=np.zeros(n, dtype=int)
        
        #count number of targets and prediction in every class
        for (x,y) in ergebnis:
            confusion_matrix[y,x]+=1
            targets_number[y]+=1
        
        #uper left corner of confusion matrix
        string="Real\t|Predicted\n\t"
        
        #upper line of matrix
        for i in range(n):
            string+="|"+str(classes[i])+"\t"
        string+="\n\u2500"
        
        #horizontal separator line between two fields of confusion matrix
        temp="\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        
        #horizontal separator line between first and second lines of the confusion matrix
        for i in range(n):
            string+=temp+"\u253C"
        string+=temp+"\n"
        
        for i in range(n):
            #print upper half of the most left field of line i
            string+=str(classes[i])+"\t"
            
            #print upper half of every field of the confusion matrix
            #(right of the most left field)
            for j in range(n):
                string+="|"+str(confusion_matrix[i,j])+"\t"
            
            #print lower half of the most left field
            string+="\n\t"
            
            #print lower half of every field of the confusion matrix
            #(right of the most left field)
            for j in range(n):
                percentage=round(confusion_matrix[i,j]/targets_number[i]*100, 1)
                
                if percentage!=100:
                    string+="|"+str(percentage)+" %\t"
                else:
                    string+="|"+str(percentage)+" %"
            string+="\n\u2500"
            
            #separator line between two lines of the confusion matrix
            for j in range(n):
                if i!=n-1:
                    string+=temp+"\u253C"
                else:
                    string+=temp+"\u2534"
            string+=temp+"\n"
        
        #output confusion matrix
        print(string)
    
    def plot_loss(self):
        #define labels for the legend
        string_1="Loss training"
        string_2="Loss test"
        
        #werte_2=self.loss_training
        epochs=len(self.loss_training)

        #Dimensions of the plot
        hoehe=21
        breite=epochs
        
        #upper frame
        frame="_"
        for i in range(breite):
            frame+="_"
        frame+="_\n"
        
        for i in range(0,hoehe):
            frame+="|"      #left frame
            
            for j in range(breite):
                temp_3=1/(2*hoehe-2)
                
                temp_1=self.loss_training[j]-(1-i/(hoehe-1))
                temp_2=self.loss_test[j]-(1-i/(hoehe-1))
                
                temp_1a=self.loss_training[j]-(1-(i+0.5)/(hoehe-1))
                temp_2a=self.loss_test[j]-(1-(i+0.5)/(hoehe-1))
                
                temp_1b=self.loss_training[j]-(1-(i-0.5)/(hoehe-1))
                temp_2b=self.loss_test[j]-(1-(i-0.5)/(hoehe-1))
                
                #print functions
                if abs(temp_2)<=1/2*temp_3:
                    #if test_loss's value is near middle of box
                    frame+="\033[31m\u2588\033[39m"
                elif abs(temp_2a)<=1/2*temp_3:
                    #if test_loss's value is near the middle betwenn this box and the box down
                    if abs(temp_1)<=1/2*temp_3 or abs(temp_1b)<=1/2*temp_3:
                        #if training_loss's value is near the middle of this box
                        #or training_loss's value is near the middle between this bos and the box above
                        frame+="\033[31m\033[44m\u2584\033[39m\033[49m"
                    else:
                        #training_loss's value is far away
                        frame+="\033[31m\u2584\033[39m"
                elif abs(temp_2b)<=1/2*temp_3:
                    #if test_loss's value is near the middle betwenn thos box and the box above
                    if abs(temp_1)<=1/2*temp_3 or abs(temp_1a)<=1/2*temp_3:
                        #if training_loss's value is near the middle of this box
                        #or training_loss's value is near the middle between this bos and the box down
                        frame+="\033[31m\033[44m\u2580\033[39m\033[49m"
                    else:
                        #training_loss's value is far away
                        frame+="\033[31m\u2580\033[39m"
                elif abs(temp_1)<=1/2*temp_3:
                    #if training_loss's value is near the middle of the box 
                    frame+="\033[34m\u2588\033[39m"
                elif abs(temp_1a)<=1/2*temp_3:
                    #if training_loss's value is near the middle of this box and the box down
                    frame+="\033[34m\u2584\033[39m"
                elif abs(temp_1b)<=1/2*temp_3:
                    #if training_loss's value is near the middle of this box and the upper box
                    frame+="\033[34m\u2580\033[39m"
                else:
                    if i==hoehe-1:
                        frame+="_"    #lower frame
                    else:
                        frame+=" "    #nothing here
            
            #frame and ticks right
            if (i+1)%2:#==hoehe-1:
                frame+=f"\u251c\u2500 {round(1-i/(hoehe-1),1)}\n"    #long ticks
            else:
                frame+="\u251c\n"    #short ticks
        
        frame+=" "
        i=0
        
        #ticks on lower frame
        while i<=epochs-5:
            frame+="|''''"
            i+=5
        
        frame+="|"
        i+=1

        #this loop get executed, if epochs%5!=0
        while i<=epochs-1:
            frame+="'"
            i+=1
        
        frame+="\n "
        i=0
        
        while i<=epochs:
            frame+=f"{i}"
            number_of_digits=int(len(str(i)))

            #compensation, if the number is longer as one digit
            for j in range(5-number_of_digits):
                frame+=" "
            
            i+=5
        
        #print x-axis label
        frame+=f"\n\033[{epochs//2-3}C EPOCHS\n"
        
        #print legend
        frame+=f"\033[31m \u2588\u2588\u2588 \033[39m{string_1}\n\033[34m \u2588\u2588\u2588 \033[39m{string_2}\n"
        
        #print y-axis label
        frame+=f"\033[{epochs+9}C\033[{hoehe//2+8}AL\033[1B\033[1DO\033[1B\033[1DS\033[1B\033[1DS\n"
        frame+=f"\033[{hoehe//2+6}B\n"

        #plott
        print(frame)

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("=== Einstellungen für das Neuronale Netz ===")
        
        # ------------------------------
        # 1. Datensatz auswählen
        # ------------------------------
        while True:
            print("\n1) MNIST (28x28) - entspricht 784 Input-Neuronen und etwa 16.000 Trainingsbeispielen (hier wird zusätzlich das Paket pandas benötigt).")
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
        
        while True:
            print(f"\nWelchen Wert (>0) soll die Lernrate haben? Ist der Wert \u2264 0 wird der Standardwert verwendet.")
            eingabe = input("Bitte als Dezimalzahl angeben: ")
            try:
                learningrate = float(eingabe)
                break
            except ValueError:
                print("Das war keine gültige Dezimalzahl. Bitte erneut versuchen.")
    else:
        if sys.argv[1] == "1":
            dataset = "mnist"
        elif sys.argv[1] == "2":
            dataset = "digits"
        else:
            print("Ungültige Eingabe! Bitte 1 oder 2 eingeben.")
        
        if sys.argv[2] == "1":
            train_mode = "full_batch"
        elif sys.argv[2] == "2":
            train_mode = "mini_batch"
        else:
            print("Ungültige Eingabe! Bitte 1 oder 2 eingeben.")
        
        if sys.argv[3] == "2":
            activation = "softplus"
        else:
            activation = "sigmoid"
        
        if sys.argv[4] == "1":
            loss = "mse"
        elif sys.argv[4] == "2":
            loss = "ce"
        else:
            print("Ungültige Eingabe, bitte 1 oder 2 eingeben.")
                
        learningrate=float(sys.argv[5])
    
    if learningrate<=0:
        print("Eingabe für die Lernrate ist kleiner gleich 0! Es wird der Standardwert verwendet.")
        
        if train_mode == "full_batch" and activation == "softplus":
            learningrate=0.03
        elif train_mode == "full_batch" and activation == "sigmoid":
            learningrate=1.4
        else:
            learningrate=0.1
    
    print("\n=== Zusammenfassung der Auswahl ===")
    print("Datensatz:        ", dataset)
    print("Trainingsmodus:   ", train_mode)
    print("Aktivierung:      ", activation)
    print("Lossfunktion:     ", loss)
    print("Lernrate:         ", learningrate)
    print("=====================================\n")
    
    net = Network(dataset=dataset, activation_mode=activation) 

    if train_mode == "full_batch" and activation == "softplus":
        net.train_full_batch(epochs=50, lr=learningrate) #optimal lr=0.03
    elif train_mode == "full_batch" and activation == "sigmoid":
        net.train_full_batch(epochs=50, lr=learningrate) #optimal lr=1.4
    else:
        net.SGD(net.training_data,
                net.test_data,
                epochs=30,
                mini_batch_size=30,
                eta=learningrate)    #optimal lr=0.1

    accuracy = net.evaluate(net.test_data) 
    percentage = accuracy[0] / len(net.test_data) * 100
    print(f"Genauigkeit nach SGD: {accuracy[0]} von {len(net.test_data)} Testbeispielen korrekt klassifiziert ({percentage:.2f}%).")
    net.print_confusion_matrix(classes=[1,5,7])
    net.plot_loss()
    
    os.makedirs("loss", exist_ok=True)
    
    filename = os.path.join("loss", f"{dataset}_{train_mode}_{activation}_{loss}_{learningrate}.npz")
    
    np.savez(filename, array_a=net.loss_training, array_b=net.loss_test)
    
    print(f"Losses for training and testing are saved in: {filename}\n")
    """
    geladen = np.load(filename)
    
    print(geladen["array_a"])
    print(geladen["array_b"])
    """
