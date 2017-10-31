
class Feed_fwd_nn:
    n_hidden_layers = 0
    n_hidden_diamensions = []
    n_input = 0
    n_output = 0
    weights = []
    biases = []
    diamensions = []
    activations = []
    delta_weights = []
    delta_biases = []
    
    
    def __init__(self, n_input, n_output, n_hidden_diamensions):
        np.random.seed(0)
        self.n_hidden_layers = len(n_hidden_diamensions)
        self.n_hidden_diamensions = n_hidden_diamensions
        self.n_input = n_input
        self.n_output = n_output
        
        diamensions = []
        diamensions.append(n_input)
        diamensions.extend(n_hidden_diamensions)
        diamensions.append(n_output)
        
        self.diamensions = diamensions
        
        for i in range(len(self.diamensions)-1):
            weight = np.random.randn(self.diamensions[i], self.diamensions[i+1]) / np.sqrt(self.diamensions[i])
            bias = np.zeros((1, self.diamensions[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)
            
        self.delta_weights = [np.zeros(w.shape) for w in self.weights]
        self.delta_biases = [np.zeros(b.shape) for b in self.biases]
    
    
    def forward_propagate(self, x):
        
        activations = []
        z_list = []
        a = np.array(x)
        activations.append(a)
        ## forward propagation 
        for i in range(len(self.diamensions)-1):
            z = a.dot(self.weights[i]) + self.biases[i]
            z_list.append(z)
            a = np.tanh(z)
            activations.append(a)
        
        exp_scores = np.exp(a)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        print activations
        return z_list, activations
        
    
    def predict(self, x):
        
        a = np.array(x)
        self.activations.append(a)
        ## forward propagation 
        for i in range(len(self.diamensions)-1):
            z = a.dot(self.weights[i]) + self.biases[i]
            a = np.tanh(z)
            self.activations.append(a)
            
        #print self.activations
        exp_scores = np.exp(a)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)
        
    
    
    # learning rate, 
    def fit(self, X_train, y_train, num_iterations = 20000, learning_rate = 0.01, regularization = 0.01):
        
        
        for i in range(num_iterations):
            
            z, activations = self.forward_propagate(X_train)
            
            delta = self.cost_prime(activations[-1], y_train) * self.sigmoid_prime(z[-1])
            self.delta_biases[-1] = delta
            
            self.delta_weights[-1] = np.dot(delta, activations[-1].transpose())
            
            for j in xrange(2, len(self.diamensions)):
                z_value = z[-j]
                sp = self.sigmoid_prime(z_value)
                
                delta = np.dot(delta, self.weights[-j+1].transpose()) * sp
                self.delta_biases[-j] = delta

                self.delta_weights[-j] = np.dot(delta, activations[-j].transpose())
                
                self.biases[-j] += self.delta_biases[-j]
                self.weights[-j] += self.delta_weights[-j]
                
            if num_iterations % 1000 == 0:
                print "iteration : " + str(i)
    
    
    def cost_prime(self, calculated, observed):
        return observed - calculated
        
    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def score(self):
        pass
