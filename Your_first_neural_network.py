#!/usr/bin/env python
# coding: utf-8

# # Your first neural network
# 
# In this project, you'll build your first neural network and use it to predict daily bike rental ridership. We've provided some of the code, but left the implementation of the neural network up to you (for the most part). After you've submitted this project, feel free to explore the data and the model more.
# 
# 

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Load and prepare the data
# 
# A critical step in working with neural networks is preparing the data correctly. Variables on different scales make it difficult for the network to efficiently learn the correct weights. Below, we've written the code to load and prepare the data. You'll learn more about this soon!

# In[2]:


data_path = 'Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)


# In[3]:


rides.head()


# ## Checking out the data
# 
# This dataset has the number of riders for each hour of each day from January 1 2011 to December 31 2012. The number of riders is split between casual and registered, summed up in the `cnt` column. You can see the first few rows of the data above.
# 
# Below is a plot showing the number of bike riders over the first 10 days or so in the data set. (Some days don't have exactly 24 entries in the data set, so it's not exactly 10 days.) You can see the hourly rentals here. This data is pretty complicated! The weekends have lower over all ridership and there are spikes when people are biking to and from work during the week. Looking at the data above, we also have information about temperature, humidity, and windspeed, all of these likely affecting the number of riders. You'll be trying to capture all this with your model.

# In[4]:


rides[:24*10].plot(x='dteday', y='cnt')


# ### Dummy variables
# Here we have some categorical variables like season, weather, month. To include these in our model, we'll need to make binary dummy variables. This is simple to do with Pandas thanks to `get_dummies()`.

# In[5]:


dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
data.head()


# ### Scaling target variables
# To make training the network easier, we'll standardize each of the continuous variables. That is, we'll shift and scale the variables such that they have zero mean and a standard deviation of 1.
# 
# The scaling factors are saved so we can go backwards when we use the network for predictions.

# In[6]:


quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std


# ### Splitting the data into training, testing, and validation sets
# 
# We'll save the data for the last approximately 21 days to use as a test set after we've trained the network. We'll use this set to make predictions and compare them with the actual number of riders.

# In[7]:

# In[8]:


# Save data for approximately the last 21 days
test_data = data[-21*24:]

# Now remove the test data from the data set 
data = data[:-21*24]

# Separate the data
#  into features and targets
# Separate the data into features and targets
target_field = 'cnt'
# drop casual and registered
# Separate the data into features and targets
# Separate the data into features and targets
fields_to_drop = ['casual', 'registered', 'cnt']
features = data.drop(fields_to_drop, axis=1)
targets = data[['cnt']]  # Creates a DataFrame with a single column 'cnt'

test_features = test_data.drop(fields_to_drop, axis=1)
test_targets = test_data[['cnt']]  # Same for test data


'''
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
'''


# In[9]:


# check the shape of features and targets for both training and test sets
print('features.shape:', features.shape, 'targets.shape:', targets.shape)
print('test_features.shape:', test_features.shape, 'test_targets.shape:', test_targets.shape)

# print the head of features and targets for both training and test sets
#print('features.head():\n', features.head())
#print('targets.head():\n', targets.head())
#print('test_features.head():\n', test_features.head())
#print('test_targets.head():\n', test_targets.head())

# print col names for all data sets
print('features.columns:\n', features.columns)
print('targets.columns:\n', targets.columns)
print('test_features.columns:\n', test_features.columns)
print('test_targets.columns:\n', test_targets.columns)




# We'll split the data into two sets, one for training and one for validating as the network is being trained. Since this is time series data, we'll train on historical data, then try to predict on future data (the validation set).

# In[10]:


# print col names for all data sets
print('features.columns:\n', features.columns)
print('targets.columns:\n', targets.columns)
print('test_features.columns:\n', test_features.columns)
print('test_targets.columns:\n', test_targets.columns)

# Hold out the last 60 days or so of the remaining data as a validation set
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]


# ## Time to build the network
# 
# Below you'll build your network. We've built out the structure. You'll implement both the forward pass and backwards pass through the network. You'll also set the hyperparameters: the learning rate, the number of hidden units, and the number of training passes.
# 
# <img src="assets/neural_network.png" width=300px>
# 
# The network has two layers, a hidden layer and an output layer. The hidden layer will use the sigmoid function for activations. The output layer has only one node and is used for the regression, the output of the node is the same as the input of the node. That is, the activation function is $f(x)=x$. A function that takes the input signal and generates an output signal, but takes into account the threshold, is called an activation function. We work through each layer of our network calculating the outputs for each neuron. All of the outputs from one layer become inputs to the neurons on the next layer. This process is called *forward propagation*.
# 
# We use the weights to propagate signals forward from the input to the output layers in a neural network. We use the weights to also propagate error backwards from the output back into the network to update our weights. This is called *backpropagation*.
# 
# > **Hint:** You'll need the derivative of the output activation function ($f(x) = x$) for the backpropagation implementation. If you aren't familiar with calculus, this function is equivalent to the equation $y = x$. What is the slope of that equation? That is the derivative of $f(x)$.
# 
# Below, you have these tasks:
# 1. Implement the sigmoid function to use as the activation function. Set `self.activation_function` in `__init__` to your sigmoid function.
# 2. Implement the forward pass in the `train` method.
# 3. Implement the backpropagation algorithm in the `train` method, including calculating the output error.
# 4. Implement the forward pass in the `run` method.
#   

# In[11]:


#############
# In the my_answers.py file, fill out the TODO sections as specified
#############

from my_answers import NeuralNetwork


# In[12]:


def MSE(y, Y):
    return np.mean((y-Y)**2)


# ## Unit tests
# 
# Run these unit tests to check the correctness of your network implementation. This will help you be sure your network was implemented correctly befor you starting trying to train it. These tests must all be successful to pass the project.

# In[13]:


import unittest

inputs = np.array([[0.5, -0.2, 0.1]])
targets = np.array([[0.4]])
test_w_i_h = np.array([[0.1, -0.2],
                       [0.4, 0.5],
                       [-0.3, 0.2]])
test_w_h_o = np.array([[0.3],
                       [-0.1]])

class TestMethods(unittest.TestCase):
    
    ##########
    # Unit tests for data loading
    ##########
    
    def test_data_path(self):
        # Test that file path to dataset has been unaltered
        self.assertTrue(data_path.lower() == 'bike-sharing-dataset/hour.csv')
        
    def test_data_loaded(self):
        # Test that data frame loaded
        self.assertTrue(isinstance(rides, pd.DataFrame))
    
    ##########
    # Unit tests for network functionality
    ##########

    def test_activation(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        # Test that the activation function is a sigmoid
        self.assertTrue(np.all(network.activation_function(0.5) == 1/(1+np.exp(-0.5))))

    def test_train(self):
        # Test that weights are updated correctly on training
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()
        
        network.train(inputs, targets)
        self.assertTrue(np.allclose(network.weights_hidden_to_output, 
                                    np.array([[ 0.37275328], 
                                              [-0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[ 0.10562014, -0.20185996], 
                                              [0.39775194, 0.50074398], 
                                              [-0.29887597, 0.19962801]])))

    def test_run(self):
        # Test correctness of run method
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))

suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
unittest.TextTestRunner().run(suite)


# ## Training the network
# 
# Here you'll set the hyperparameters for the network. The strategy here is to find hyperparameters such that the error on the training set is low, but you're not overfitting to the data. If you train the network too long or have too many hidden nodes, it can become overly specific to the training set and will fail to generalize to the validation set. That is, the loss on the validation set will start increasing as the training set loss drops.
# 
# You'll also be using a method know as Stochastic Gradient Descent (SGD) to train the network. The idea is that for each training pass, you grab a random sample of the data instead of using the whole data set. You use many more training passes than with normal gradient descent, but each pass is much faster. This ends up training the network more efficiently. You'll learn more about SGD later.
# 
# ### Choose the number of iterations
# This is the number of batches of samples from the training data we'll use to train the network. The more iterations you use, the better the model will fit the data. However, this process can have sharply diminishing returns and can waste computational resources if you use too many iterations.  You want to find a number here where the network has a low training loss, and the validation loss is at a minimum. The ideal number of iterations would be a level that stops shortly after the validation loss is no longer decreasing.
# 
# ### Choose the learning rate
# This scales the size of weight updates. If this is too big, the weights tend to explode and the network fails to fit the data. Normally a good choice to start at is 0.1; however, if you effectively divide the learning rate by n_records, try starting out with a learning rate of 1. In either case, if the network has problems fitting the data, try reducing the learning rate. Note that the lower the learning rate, the smaller the steps are in the weight updates and the longer it takes for the neural network to converge.
# 
# ### Choose the number of hidden nodes
# In a model where all the weights are optimized, the more hidden nodes you have, the more accurate the predictions of the model will be.  (A fully optimized model could have weights of zero, after all.) However, the more hidden nodes you have, the harder it will be to optimize the weights of the model, and the more likely it will be that suboptimal weights will lead to overfitting. With overfitting, the model will memorize the training data instead of learning the true pattern, and won't generalize well to unseen data.  
# 
# Try a few different numbers and see how it affects the performance. You can look at the losses dictionary for a metric of the network performance. If the number of hidden units is too low, then the model won't have enough space to learn and if it is too high there are too many options for the direction that the learning can take. The trick here is to find the right balance in number of hidden units you choose.  You'll generally find that the best number of hidden nodes to use ends up being between the number of input and output nodes.

# In[14]:


import sys
import os

# Import hyperparameters
from my_answers import iterations, learning_rate, hidden_nodes, output_nodes

# Initialize the network
N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)
network.load_latest_weights()


# In[15]:


# Initialize dictionary to store loss values
losses = {'train':[], 'validation':[]}

# Early stopping parameters
early_stopping = True
patience = 100  # Increased patience
min_loss_threshold = 0.3  # Threshold for acceptable loss
start_early_stopping_at_iter = 500  # Start checking early stopping after this many iterations
best_val_loss = float('inf')
patience_counter = 0

save_weights_period = 500
# Create a directory for saving weights if it doesn't exist
weights_dir = 'saved_weights'
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

for ii in range(iterations):
    # Select a random batch of records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    X, y = train_features.iloc[batch].values, train_targets.iloc[batch]['cnt']
    
    # Perform training on the batch
    network.train(X, y)
    
    # Evaluate the model on the entire training set
    train_predictions = network.run(train_features)
    train_loss = MSE(train_predictions.T, train_targets['cnt'].values)
    
    # Evaluate the model on the entire validation set
    val_predictions = network.run(val_features)
    val_loss = MSE(val_predictions.T, val_targets['cnt'].values)
    
    # Print out the training progress
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()
    
    # Store the losses for plotting or analysis
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)

    # Save weights periodically
    if ii % save_weights_period == 0:
        filename = os.path.join(weights_dir, f'weights_iter_{ii}.npz')
        network.save_weights(filename)

    # Early stopping checks
    if early_stopping and ii > start_early_stopping_at_iter:
        if train_loss < min_loss_threshold and val_loss < min_loss_threshold:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                filename = os.path.join(weights_dir, 'weights_early_stopping.npz')
                network.save_weights(filename)
                print("\nEarly stopping triggered at iteration %d" % ii)
                break


# In[ ]:


plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
_ = plt.ylim()


# ## Check out your predictions
# 
# Here, use the test data to view how well your network is modeling the data. If something is completely wrong here, make sure each step in your network is implemented correctly.

# In[25]:


fig, ax = plt.subplots(figsize=(8,4))

mean, std = scaled_features['cnt']
predictions = network.run(test_features).T*std + mean
ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets['cnt']*std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(rides.iloc[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)


# ## OPTIONAL: Thinking about your results(this question will not be evaluated in the rubric).
#  
# Answer these questions about your results. How well does the model predict the data? Where does it fail? Why does it fail where it does?
# 
# > **Note:** You can edit the text in this cell by double clicking on it. When you want to render the text, press control + enter
# 
# #### Your answer below

# ## Submitting:
# Open up the 'jwt' file in the first-neural-network directory (which also contains this notebook) for submission instructions

# # try using Keras for better results

# In[16]:


# print col names for all data sets
#print('features.columns:\n', features.columns)
#print('targets.columns:\n', targets.columns)
#print('test_features.columns:\n', test_features.columns)
#print('test_targets.columns:\n', test_targets.columns)

# Hold out the last 60 days or so of the remaining data as a validation set
#train_features, train_targets = features[:-60*24], targets[:-60*24]
#val_features, val_targets = features[-60*24:], targets[-60*24:]

# print out dtypes of the above data sets
print('train_features.dtypes:\n', train_features.dtypes)
print('train_targets.dtypes:\n', train_targets.dtypes)
print('val_features.dtypes:\n', val_features.dtypes)
print('val_targets.dtypes:\n', val_targets.dtypes)

# check if they above datasets are valid inputs for Keras model
print('train_features.shape:', train_features.shape, 'train_targets.shape:', train_targets.shape)
print('val_features.shape:', val_features.shape, 'val_targets.shape:', val_targets.shape)

# check if above datasets are all numpy arrays
print('train_features type:', type(train_features))
print('train_targets type:', type(train_targets))
print('val_features type:', type(val_features))
print('val_targets type:', type(val_targets))


# In[21]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


float32 = "float32"
# Create the model
model = Sequential()
model.add(Dense(64, input_dim=train_features.shape[1], activation='relu')) # Adding a hidden layer
model.add(Dense(1, activation='linear'))

adam_optimizer = Adam(learning_rate=0.001)
# Compile the model
model.compile(optimizer=adam_optimizer, loss='mean_squared_error')

# Train the model
#model.fit(train_features.to_numpy().astype(float32), train_targets.to_numpy().astype(float32), epochs=10, batch_size=32)

# Evaluate the model
#loss = model.evaluate(val_features.to_numpy().astype(float32), val_targets.to_numpy().astype(float32))
#print('Test loss:', loss)


# In[22]:


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[23]:


from tensorflow.keras.callbacks import EarlyStopping

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model with validation split
history = model.fit(
    train_features.to_numpy().astype(float32), 
    train_targets.to_numpy().astype(float32),
    epochs=100,
    batch_size=32,
    validation_data=(val_features.to_numpy().astype('float32'), val_targets.to_numpy().astype('float32')),
    callbacks=[early_stopping]
)

# Evaluate the model
loss = model.evaluate(test_features.to_numpy().astype(float32), test_targets.to_numpy().astype(float32))
print('Test loss:', loss)

# Optionally, plot the training history
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[30]:


# Ensure that test_features is a NumPy array with the correct data type
test_features_array = test_features.to_numpy().astype('float32')

# Get predictions
predictions_scaled = model.predict(test_features_array)

# Rescale predictions back to original distribution
mean = scaled_features['cnt'][0]
std = scaled_features['cnt'][1]
predictions = predictions_scaled * std + mean

# Convert predictions and actual values to their original scale
actual = (test_targets['cnt']*std + mean).values

# Now you can proceed to plot or save the predictions as needed


# Assuming you have already defined `predictions` and `actual` as numpy arrays

# Plotting
fig, ax = plt.subplots(figsize=(8,4))

ax.plot(predictions, label='Prediction')
ax.plot(actual, label='Data')  # No need to index 'cnt' as `actual` is a numpy array
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(rides.iloc[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)

plt.show()


# In[ ]:





# In[ ]:





# In[24]:


import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Is GPU available:", tf.test.is_gpu_available())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[ ]:





# In[ ]:




