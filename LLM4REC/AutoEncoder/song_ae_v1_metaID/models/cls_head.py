from torch import nn


class ClsHead(nn.Module):
    def __init__(self, num_class, input_dim, num_layers=1):
        super(ClsHead, self).__init__()
        self.num_class = num_class
        self.input_dim = input_dim
        self.num_layers = num_layers

        self.ln_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        h_dims = list()
        for idx in range(2, num_layers + 2):
            h_dims.append(input_dim * idx)
        for num_in, num_out in zip(([input_dim] + h_dims)[:-1], h_dims):
            self.ln_layers.append(nn.Linear(num_in, num_out))
            self.norm_layers.append(nn.LayerNorm(num_out))
        self.out_layer = nn.Linear(h_dims[-1], num_class)

        self.act = nn.ReLU()

        self._init()

    def _init(self):
        for m in self.ln_layers:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        nn.init.xavier_uniform_(self.out_layer.weight)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x):
        for l_ln, l_norm in zip(self.ln_layers, self.norm_layers):
            x = l_ln(x)
            x = l_norm(x)
            x = self.act(x)
        logits = self.out_layer(x)
        return logits
