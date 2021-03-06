class K_Means:
    n = 0
    def __init__(self, n):
        self.n = n
        
    def fit(self, train_x, n_iterations = 100):
        iterations = n_iterations
        self.count = train_x.shape[0]
        self.features = train_x.shape[1]
        self.cluseter_centers = np.random.randn(self.n, self.features)
        
        current_assignment = np.zeros((self.count, 1), dtype=np.int) - 1
        
        self.train = np.concatenate((train_x, current_assignment), axis=1)
        
        prev_assignment = current_assignment
        
        distances = []
        for i in range(self.cluseter_centers.shape[0]):
            distance = np.linalg.norm(train_x - self.cluseter_centers[i], axis=1)
            distances.append(distance)
        distance_array = np.array(distances)
        current_assignment = np.argmax(distance_array, axis=0)
        
        self.train[:,-1] = current_assignment
        
        
        while (iterations > 0) and not (np.prod(current_assignment == prev_assignment)):
            
            iterations -= 1
            
            for i in range(self.cluseter_centers.shape[0]):
                index = np.where(self.train[:,-1] == i)
                data = self.train[np.where(self.train[:,-1] == i)][:,:-1]
                data_count = self.train[np.where(self.train[:,-1] == i)][:,:-1].shape[0]
                if data_count == 0:
                    data = np.zeros((1, self.features))
                    self.cluseter_centers[i] = data
                else:
                    self.cluseter_centers[i] = np.sum(data, axis=0)/ data_count
            
             
            distances = []
            for i in range(self.cluseter_centers.shape[0]):
                distance = np.linalg.norm(train_x - self.cluseter_centers[i], axis=1)
                distances.append(distance)
            distance_array = np.array(distances)
            current_assignment = np.argmax(distance_array, axis=0)
        
            self.train[:,-1] = current_assignment
            prev_assignment = current_assignment
            
        self.assignment = current_assignment
    
    def view(self):
        pass
