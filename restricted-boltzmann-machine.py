import pandas as pd
import numpy as np

class RBM():
    def __init__(self, num_visible, num_hidden, T):
        self.num_hidden = num_hidden # 24
        self.num_visible = num_visible # 15 (12 features + 3 classes)
        
        # random network initialization 
        self.weights = np.random.rand(num_visible, num_hidden)
        self.hidden_bias = np.random.rand(num_hidden)
        self.visible_bias = np.random.rand(num_visible)
        self.T=T

    def train(self, data, classes, max_epochs = 1000, batch_size=128, learning_rate = 0.001):
        num_examples = data.shape[0]
        # randomize the dataset
        # append a column of 3 zeros to the data, as at the beginning we dont know the class
        data = np.append(data, np.zeros((num_examples, 3)), axis=1)
        # epoch - a full iteration over all the examples
        for epoch in range(max_epochs):
            for i in range(0, num_examples, batch_size):
                batch = data[i:i+batch_size]
                cl = classes[i:i+batch_size] # the known classes from the datasets
                self.train_internal(batch, cl, learning_rate)


    def train_internal(self, v, classes, learning_rate = 0.001):
        # v is the data
        h0 = self.get_hidden(v) # get the hidden vector of v (h is length of 24)

        hidden = h0
        v1 = self.get_visible(hidden) # Get recostructed v
        h1 = self.get_hidden(v1)

        # remove the last 3 columns of zeros from the data
        v = v[:, :-3]
        # and replace them with the original classes, for the updating of the weights
        v = np.append(v, classes, axis=1)
        positive_associations = np.dot(v.T, h0)
        negative_associations = np.dot(v1.T, h1)

        # updating weights and biases of the model
        self.weights += learning_rate * (positive_associations - negative_associations)
        self.visible_bias += learning_rate * np.mean(v - v1, axis=0) #mean for working with batches
        self.hidden_bias += learning_rate * np.mean(h0 - h1, axis=0)

    def get_hidden(self, visible_vector):
        hidden_activations = np.dot(visible_vector, self.weights) + self.hidden_bias
        hidden_probabilities = self.sigmoid(hidden_activations)
        return (hidden_probabilities > np.random.rand(*hidden_probabilities.shape)).astype('int')

    def get_visible(self, hidden_probabilities):
        visible_activations = np.dot(hidden_probabilities, self.weights.T) + self.visible_bias
        visible_probabilities = self.sigmoid(visible_activations)
        return (visible_probabilities > np.random.rand(*visible_probabilities.shape)).astype('int')

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x/ self.T))

    def get_hidden_prob(self, visible_probabilities):
        # Takes a visible vector and returns the hidden vector
        hidden_activations = np.dot(visible_probabilities, self.weights) + self.hidden_bias
        hidden_probabilities = self.sigmoid(hidden_activations)
        return hidden_probabilities

    def get_visible_prob(self, hidden_probabilities):
        # Takes a hidden vector and returns the visible vector
        visible_activations = np.dot(hidden_probabilities, self.weights.T) + self.visible_bias
        visible_probabilities = self.sigmoid(visible_activations)
        return visible_probabilities

    def classify(self, data):
        # Takes an examples with 12 one-hot features and 3 zeros after them and returns the class of the example
        hidden = self.get_hidden(data)
        visible = self.get_visible_prob(hidden)
        # keep only the last 3 visible weights as those are for the classes
        return np.argmax(visible[-3:])

if __name__ == '__main__':
    np.random.seed(0)

    # read the dataset
    df = pd.read_csv("archive/Iris.csv")
    df.dropna(inplace=True)

    # find maximum petal length, petal width, sepal length, sepal width
    max_pental_length = df['PetalLengthCm'].max()
    max_pental_width = df['PetalWidthCm'].max()
    max_sepal_length = df['SepalLengthCm'].max()
    max_sepal_width = df['SepalWidthCm'].max()

    # create numpy feature vector for each example. Each example has 12 features, for each feature
    # split the value into low, medium and high value based on the maximum value of that feature
    # and put 1 in the corresponding bin and 0 in the other bins while saving the class of each example
    data = []
    classes = []
    for index, row in df.iterrows():
        feature = []
        if row['PetalLengthCm'] < max_pental_length/3:
            feature.extend([1, 0, 0])
        elif row['PetalLengthCm'] < 2*max_pental_length/3:
            feature.extend([0, 1, 0])
        else:
            feature.extend([0, 0, 1])

        if row['PetalWidthCm'] < max_pental_width/3:
            feature.extend([1, 0, 0])
        elif row['PetalWidthCm'] < 2*max_pental_width/3:
            feature.extend([0, 1, 0])
        else:
            feature.extend([0, 0, 1])

        if row['SepalLengthCm'] < max_sepal_length/3:
            feature.extend([1, 0, 0])
        elif row['SepalLengthCm'] < 2*max_sepal_length/3:
            feature.extend([0, 1, 0])
        else:
            feature.extend([0, 0, 1])

        if row['SepalWidthCm'] < max_sepal_width/3:
            feature.extend([1, 0, 0])
        elif row['SepalWidthCm'] < 2*max_sepal_width/3:
            feature.extend([0, 1, 0])
        else:
            feature.extend([0, 0, 1])

        data.append(feature)
        if row['Species'] == 'Iris-setosa':
            classes.append([1, 0, 0])
        elif row['Species'] == 'Iris-versicolor':
            classes.append([0, 0, 1])
        else:
            classes.append([0, 1, 0])

    data = np.array(data)
    # create an RBM model with 15 visible units and 24 hidden units
    rbm = RBM(num_visible=15, num_hidden=24, T=1)

    # evaluate the dataset
    correct = 0
    # data is a vector of 12 one hot features, we append 3 zeros for inference
    data_test = np.append(data, np.zeros((data.shape[0], 3)), axis=1)

    # evaluate the model on dataset before training
    for example, c in zip(data_test, classes):
        # check if the class of the example is the same as the known class
        if np.argmax(c) == rbm.classify(example):
            correct += 1
    print(f'Accuracy before training: {correct/len(data) * 100:.2f}%')

    # call the training algorithm of the model
    rbm.train(data, classes=classes, max_epochs=1000)

    # evaluate the model on dataset after training
    correct = 0
    for example, c in zip(data_test, classes):
        if np.argmax(c) == rbm.classify(example):
            correct += 1

    print(f'Accuracy after training: {correct/len(data) * 100:.2f}%')