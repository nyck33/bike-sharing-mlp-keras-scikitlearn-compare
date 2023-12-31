import numpy as np
import pandas as pd
import os


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
    
    def save_weights(self, filename):
        ''' Save the weights of the neural network to a file. '''
        weights = {
            'input_to_hidden': self.weights_input_to_hidden,
            'hidden_to_output': self.weights_hidden_to_output
        }
        np.savez(filename, **weights)
    
    def load_latest_weights(self, directory='saved_weights'):
        """ Load the latest weights from the specified directory. """
        if not os.path.exists(directory) or not os.listdir(directory):
            print("Directory not found or is empty. No weights loaded.")
            return

        # List all .npz files in the directory
        files = [f for f in os.listdir(directory) if f.endswith('.npz') and f.startswith('weights_iter_')]
        
        # Extract the iteration numbers from the file names
        iter_numbers = [int(f.split('_')[-1].split('.')[0]) for f in files]

        # Find the file with the highest iteration number
        latest_file = files[np.argmax(iter_numbers)]
        latest_path = os.path.join(directory, latest_file)
        
        # Load weights from this file
        data = np.load(latest_path)
        self.weights_input_to_hidden = data['input_to_hidden']
        self.weights_hidden_to_output = data['hidden_to_output']
        print(f"Loaded weights from {latest_path}")
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #print("X shape, X dtype: ", X.shape, X.dtype)
        if X.dtype == 'O':
            X = X.astype('float64')

        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        # Convert all inputs to float64 if they are not already
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        # Calculate the output error - Output layer error is the difference between desired target and actual output.
        error = y - final_outputs
        
        # Calculate the output error term - For regression, this is just the error as the derivative of the activation function is 1.
        output_error_term = error
        
        # Calculate the hidden layer's contribution to the error - This is the error propagated to the hidden layer.
        hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)
        
        # Calculate the hidden layer error term - This is the error of the hidden layer scaled by the derivative of the activation function (sigmoid in this case).
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        
        # Reshape X to ensure it's a 2D array for the weight update step
        X = X.reshape(-1, self.input_nodes)
        
        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X.T
        
        # Weight step (hidden to output) - hidden_outputs needs to be a 2D array
        hidden_outputs = hidden_outputs.reshape(-1, self.hidden_nodes)
        delta_weights_h_o += output_error_term * hidden_outputs.T

        return delta_weights_i_h, delta_weights_h_o
    
    
    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: DataFrame or 2D array where each row is one data record, each column is a feature
        '''
        
        # Convert DataFrame to NumPy array if it's a DataFrame
        if isinstance(features, pd.DataFrame):
            features = features.to_numpy()

        # Ensure features is a 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)  # Reshape 1D array to 2D array with one row
        elif features.ndim > 2:
            raise ValueError("Features must be a 1D or 2D array.")

        # Convert to float64 if not already
        if features.dtype != np.float64:
            features = features.astype(np.float64)

        # Hidden layer
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer
        
        # Output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)  # signals into final output layer
        final_outputs = final_inputs  # For regression, output is linear
        
        return final_outputs



#########################################################
# Set your hyperparameters here
##########################################################
iterations = 10000
learning_rate = 0.01
hidden_nodes = 32
output_nodes = 1
