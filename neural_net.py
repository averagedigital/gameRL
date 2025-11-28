import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Weights and biases
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)
    
    def predict(self, inputs):
        # Input layer -> Hidden layer
        h1 = np.dot(inputs, self.w1) + self.b1
        h1 = np.tanh(h1)  # Activation function
        
        # Hidden layer -> Output layer
        out = np.dot(h1, self.w2) + self.b2
        out = np.tanh(out)  # Tanh for output to get range [-1, 1] for motor/torque
        
        return out

    def get_weights(self):
        return [self.w1, self.b1, self.w2, self.b2]

    def set_weights(self, weights):
        self.w1, self.b1, self.w2, self.b2 = weights

    def copy(self):
        new_net = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        new_net.w1 = self.w1.copy()
        new_net.b1 = self.b1.copy()
        new_net.w2 = self.w2.copy()
        new_net.b2 = self.b2.copy()
        return new_net
    
    def mutate(self, rate=0.1, scale=0.5):
        if np.random.random() < rate:
            self.w1 += np.random.randn(*self.w1.shape) * scale
        if np.random.random() < rate:
            self.b1 += np.random.randn(*self.b1.shape) * scale
        if np.random.random() < rate:
            self.w2 += np.random.randn(*self.w2.shape) * scale
        if np.random.random() < rate:
            self.b2 += np.random.randn(*self.b2.shape) * scale

