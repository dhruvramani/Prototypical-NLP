import torch

def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    source : https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ProtoConvNet(torch.nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, x_dim, hid_dim, z_dim):
        super(ProtoConvNet, self).__init__()
        
        self.encoder = torch.nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, X):
        x = self.encoder(x)
        return x.view(x.size(0), -1)