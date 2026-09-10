"""A fully connected network whose hidden layer widths are given explicitly.

QSPRpred's ``STFullyConnected`` hard-codes its topology as ``neurons_h1`` followed by
one or two layers of ``neurons_hx``, which cannot express the 4000-2000-1000 network
used as the DNN baseline in Lenselink et al. (2017). This subclass takes the widths as
a list instead.

``DNNModel`` accepts the estimator class through its ``alg`` argument, initialize it
as follows:

    DNNModel(..., alg=HiddenSizesFullyConnected,
             parameters={"hidden_sizes": [4000, 2000, 1000], ...})
"""

import torch
from qsprpred.extra.gpu.models.neural_network import STFullyConnected
from torch import nn
from torch.nn import functional as f


class HiddenSizesFullyConnected(STFullyConnected):
    """``STFullyConnected`` with an arbitrary number of hidden layers.

    Attributes:
        hidden_sizes (list[int]): width of each hidden layer, e.g. [4000, 2000, 1000].
            When None, falls back to the parent's neurons_h1/neurons_hx/extra_layer
            topology, so this class is a drop-in replacement.
    """

    def __init__(
        self,
        n_dim,
        n_class,
        device,
        gpus,
        n_epochs=100,
        lr=None,
        batch_size=256,
        patience=50,
        tol=0,
        is_reg=True,
        neurons_h1=256,
        neurons_hx=128,
        extra_layer=False,
        hidden_sizes=None,
        dropout_frac=0.25,
        random_state=None,
    ):
        self.__dict__["hidden_sizes"] = list(hidden_sizes) if hidden_sizes else None
        super().__init__(
            n_dim,
            n_class,
            device,
            gpus,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
            tol=tol,
            is_reg=is_reg,
            neurons_h1=neurons_h1,
            neurons_hx=neurons_hx,
            extra_layer=extra_layer,
            dropout_frac=dropout_frac,
            random_state=random_state,
        )

    def initModel(self):
        """Build the hidden stack from ``hidden_sizes``, or the parent topology."""
        sizes = self.__dict__.get("hidden_sizes")
        if not sizes:
            sizes = [self.neurons_h1, self.neurons_hx]
            if self.extra_layer:
                sizes.append(self.neurons_hx)
            self.__dict__["hidden_sizes"] = sizes
        dims = [self.n_dim] + list(sizes)
        self.dropout = nn.Dropout(self.dropout_frac)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.output_layer = nn.Linear(dims[-1], self.n_class)
        if self.is_reg:
            self.criterion = nn.MSELoss()
        elif self.n_class == 1:
            self.criterion = nn.BCELoss()
            self.activation = nn.Sigmoid()
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.activation = nn.Softmax(dim=1)

    def forward(self, X, is_train=False) -> torch.Tensor:
        y = X
        for layer in self.hidden_layers:
            y = f.relu(layer(y))
            if is_train:
                y = self.dropout(y)
        if self.is_reg:
            return self.output_layer(y)
        return self.activation(self.output_layer(y))
