import numpy as np

class NN:
     
    def __init__(self, inputs, hidden, outputs, act_type=0):
        #initializing the network's structure
        self.act_type = act_type
        self.inputs = inputs
        self.hidden = hidden 
        self.outputs = outputs
        self.structure = [inputs] + hidden + [outputs]
        
        #initialize weights matrices at random
        self.weights = []
        for i in range(len(self.structure) - 1):
            mat = np.random.rand(self.structure[i], self.structure[i+1]) #random weights matrix for every 2 neighboring layers
            self.weights.append(mat)

        #initialize all neurons to 0
        self.neurons = []
        for layer in self.structure:
            self.neurons.append(np.zeros(layer))

        #initialize all partial derivatives to 0
        self.partial_dt = []
        for mat in self.weights:
            dt = np.zeros(mat.shape)
            self.partial_dt.append(dt)

        
    #activation functions
    def A(self, vec):
        if self.act_type == 0: #sigmoid 
            for i in range(len(vec)):
                vec[i] = 1/(1+np.exp(-vec[i]))
                
        elif self.act_type == 1: #RELU 
            for i in range(len(vec)):
                vec[i] = max(0,vec[i])

        return vec        
    
    #derivatives of activation functions
    def dtA(self, vec):
        if self.act_type == 0: #sigmoid 
            for i in range(len(vec)):
                vec[i] = (1/(1+np.exp(-vec[i])))*(1 - (1/(1+np.exp(-vec[i]))))

        elif self.act_type == 1: #RELU 
            for i in range(len(vec)):
                vec[i] = 1 if vec[i] > 0 else 1
    
        return vec

    #calculate result of the network using forward propagation method
    def calc(self, input_vec):
        output_vec = input_vec #first layer is always the input layer
        self.neurons[0] = input_vec #set 1st neuron layer
        #propagate forward
        for i in range(len(self.weights)):
            output_vec = np.matmul(output_vec, self.weights[i]) #calculate next layer
            output_vec = self.A(output_vec) #pass through activation function
            self.neurons[i+1] = output_vec #set (i+1)'th neuron layer
            
        return output_vec
    
    #calculate partial derivatives using backward propagation method
    def calc_dt(self, error, k=1):
        if k > len(self.partial_dt):
            return error

        i = len(self.partial_dt) - k
        raw = [self.neurons[i], error * self.dtA(self.neurons[i+1])]
        aligned = [raw[0].reshape(raw[0].shape[0], -1), (raw[1].reshape(raw[1].shape[0], -1)).T]
        self.partial_dt[i] = np.matmul(aligned[0], aligned[1])
        error = np.matmul(raw[1] , (self.weights[i]).T)
        self.calc_dt(error, k+1)

    #use partial derivatives to minimize the error function
    def gradient_descent(self, step):
        for i in range(len(self.partial_dt)):
            self.weights[i] += self.partial_dt[i]*step

    #train the network using labaled data and gradient descent
    def train(self, data, labels, epochs):
        for i in range(epochs):
            epoch_accuracy = 0
            for d,l in zip(data,labels):
                result = self.calc(d)
                error = l - result
                epoch_accuracy += abs(l - result)
                self.calc_dt(error)
                self.gradient_descent(.00001)
            print("epoch number: {} total error is {}".format(i, epoch_accuracy/len(data)))

    #single case prediction
    def predict(self, input, label):
        result = self.calc(input)
        print("the model predicted the result: {}, answer is: {}".format(result, label))