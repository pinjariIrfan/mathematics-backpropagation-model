"""
Simple Neural Network with Backpropagation
Educational implementation demonstrating the algorithm
"""
import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights randomly
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        
        # Initialize biases
        self.bias_hidden = np.random.randn(hidden_size)
        self.bias_output = np.random.randn(output_size)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def forward_propagation(self, inputs):
        """Forward pass through the network"""
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        
        return self.final_output
    
    def backward_propagation(self, inputs, expected_output, actual_output, learning_rate=0.1):
        """Backpropagation algorithm implementation"""
        # Calculate output layer error
        output_error = expected_output - actual_output
        output_delta = output_error * self.sigmoid_derivative(actual_output)
        
        # Calculate hidden layer error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.weights_input_hidden += inputs.T.dot(hidden_delta) * learning_rate
        
        self.bias_output += np.sum(output_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta) * learning_rate
    
    def train(self, training_inputs, training_outputs, epochs=10000):
        """Train the neural network"""
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward_propagation(training_inputs)
            
            # Backward propagation
            self.backward_propagation(training_inputs, training_outputs, output)
            
            if epoch % 1000 == 0:
                loss = np.mean(np.square(training_outputs - output))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Example usage
if __name__ == "__main__":
    print("🧮 Mathematics Backpropagation Model")
    print("=" * 40)
    
    # Example: XOR problem
    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_outputs = np.array([[0], [1], [1], [0]])
    
    # Create and train neural network
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    
    print("Training neural network...")
    nn.train(training_inputs, training_outputs, epochs=5000)
    
    # Test the trained network
    print("\nTesting trained network:")
    for i in range(len(training_inputs)):
        prediction = nn.forward_propagation(training_inputs[i])
        print(f"Input: {training_inputs[i]} -> Prediction: {prediction[0]:.4f}")
    
    print("=" * 40)
    print("✅ Backpropagation implementation completed!")