from torch import layer_norm
import torch.nn as nn
import torch.nn.functional as F
# from utils import init_weights


# def init_weights(net: nn.Module, init_type='zero_constant'):
#     #####################################################################################
#     # TODO: A function that initializes the weights in the entire nn.Module recursively.#
#     # When you get an instance of your nn.Module model later, pass this function        #
#     # to torch.nn.Module.apply. For more explanation visit:                             #
#     # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch #
#     # Note: initialize both weights and biases of the entire model                      #
#     #####################################################################################
#     valid_initializations = ['zero_constant', 'uniform']
#     if init_type not in valid_initializations:
#         print("INVALID TYPE OF WEIGHT INITIALIZATION!")
#         return

#     for layer in net.children():
#         if isinstance(layer, nn.Linear):
#             nn.init.zeros_(layer.weight.data)
#             nn.init.zeros_(layer.bias.data)


class MLP(nn.Module):
    def __init__(self, units: list, hidden_layer_activation='relu', init_type='uniform'):
        super(MLP, self).__init__()
        self.units = units
        self.n_layers = len(units)  # including input and output layers
        valid_activations = {'relu': nn.ReLU(),
                             'tanh': nn.Tanh(),
                             'sigmoid': nn.Sigmoid()}
        self.activation = valid_activations[hidden_layer_activation]
        #####################################################################################
        # TODO: Implement the model architecture with respect to the units: list            #
        # use nn.Sequential() to stack layers in a for loop                                 #
        # It can be summarized as: ***[LINEAR -> ACTIVATION]*(L-1) -> LINEAR -> SOFTMAX***  #
        # Use nn.Linear() as fully connected layers                                         #
        #####################################################################################

        self.mlp = nn.Sequential()

        # stacked_layers.add_module('linear0', nn.Linear(256, 256))
        for i in range(self.n_layers - 2):
            self.mlp.add_module('linear' + str(i+1),
                                nn.Linear(self.units[i], self.units[i+1]))
            self.mlp.add_module('activation' + str(i+1), self.activation)

            if i == self.n_layers - 3:
                self.mlp.add_module('linear' + str(i+2),
                                    nn.Linear(self.units[i+1], self.units[i+2]))

        # init_weights(self.mlp, init_type='zero_constant')

        # self.mlp = nn.ModuleList()
        # self.mlp.append(stacked_layers)
        #####################################################################################
        #                                 END OF YOUR CODE                                  #
        #####################################################################################

    def forward(self, X):
        #####################################################################################
        # TODO: Forward propagate the input                                                 #
        # ~ 2 lines of code#
        # First propagate the input and then apply a softmax layer                          #
        #####################################################################################
        out = self.mlp(X)
        #####################################################################################
        #                                 END OF YOUR CODE                                  #
        #####################################################################################
        return out
