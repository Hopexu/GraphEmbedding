from ..utils import *
def encode_layer(layers):
    length = len(layers)
    if length <= 1:
        return None
    encode_layers = [nn.Linear(layers[0],layers[1])]
    for i in range(2,layers):
        encode_layers.append(nn.Tanh)
        encode_layers.append(nn.Linear(layers[i-1],layers[i]))
    return ModuleList(encode_layers)

def decode_layer(layers):
    length = len(layers)
    if length <= 1:
        return None
    decode_layers = [nn.Linear(layers[length - 1],layers[length - 2])]

    for i in range(length - 2, 0,-1):
        decode_layers.append(nn.Tanh)
        decode_layers.append(nn.Linear[i],layers[i - 1])
    decode_layers.append(nn.Sigmoid)
    return ModuleList(decode_layers)