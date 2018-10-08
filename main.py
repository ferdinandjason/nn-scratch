from nn.neuralnetwork import NeuralNetwork
from nn.dataset import Dataset
from nn.layer import Layer


if __name__ == '__main__':
    dataset = Dataset('iris.csv')
    train,test = dataset.get_data(0.8)

    nn = NeuralNetwork()
    nn.add(Layer(4))
    nn.add(Layer(16,activation='sigmoid'))
    nn.add(Layer(3,activation='sigmoid'))

    nn.learn(train,test,100,0.1)