import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def reset_parameters(layers):
    for layer in layers:
        layer.weight.data.uniform_(-3e-3,3e-3)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_layers=[400,300]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_layers (list): Number of nodes in hidden layers
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Define input and output values for the hidden layers
        dims = [state_size] + fc_layers + [action_size]
        # Create the hidden layers
        self.fc_layers = nn.ModuleList(
            [nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        # Initialize the hidden layer weights
        reset_parameters(self.fc_layers)

        print('Actor network built:', self.fc_layers)

    def forward(self, x):
        """Build an actor (policy) network that maps states -> actions."""
        # Pass the input through all the layers apllying ReLU activation, but the last
        for layer in self.fc_layers[:-1]:
            x = F.relu(layer(x))
        # Pass the result through the output layer apllying hyperbolic tangent function
        x = torch.tanh(self.fc_layers[-1](x))
        # Return the better action for the input state
        return x


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc_layers=[400,300]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_layers (list): Number of nodes in hidden layers
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Append the output size to the layers's dimensions
        dims = fc_layers + [1]
        # Create a list of layers
        layers_list = []
        layers_list.append(nn.Linear(state_size, dims[0]))
        # The second layer receives the the first layer output + action
        layers_list.append(nn.Linear(dims[0] + action_size, dims[1]))
        # Build the next layers, if that is the case
        for dim_in, dim_out in zip(dims[1:-1], dims[2:]):
            layers_list.append(nn.Linear(dim_in, dim_out))
        # Store the layers as a ModuleList
        self.fc_layers = nn.ModuleList(layers_list)
        # Initialize the hidden layer weights
        reset_parameters(self.fc_layers)
        # Add batch normalization to the first hidden layer
        self.bn = nn.BatchNorm1d(dims[0])

        print('Critic network built:', self.fc_layers)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Pass the states into the first layer
        x = self.fc_layers[0](state)
        x = self.bn(x)
        x = F.relu(x)
        # Concatenate the first layer output with the action
        x = torch.cat((x, action), dim=1)
        # Pass the input through all the layers apllying ReLU activation, but the last
        for layer in self.fc_layers[1:-1]:
            x = F.relu(layer(x))
        # Pass the result through the output layer apllying sigmoid activation
        x = torch.sigmoid(self.fc_layers[-1](x))
        # Return the Q-Value for the input state-action
        return x