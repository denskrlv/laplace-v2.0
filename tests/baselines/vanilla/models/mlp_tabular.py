import torch.nn as nn


class MLPTabular(nn.Module):
    def __init__(self, num_features, num_classes=2, hidden_dims=[100, 50]):
        """
        A simple MLP for tabular data.

        Parameters
        ----------
        num_features : int
            The number of input features after preprocessing.
        num_classes : int, default=2
            The number of output classes.
        hidden_dims : list[int], default=[100, 50]
            A list of integers specifying the number of neurons in each hidden layer.
        """
        super().__init__()

        layers = []
        input_dim = num_features
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)