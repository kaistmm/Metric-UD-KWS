import torch
import torch.nn as nn

## activation layer
class Swish(nn.Module):
    """Swish is a smooth, non-monotonic function that 
    consistently matches or outperforms ReLU on 
    deep networks applied to a variety of challenging 
    domains such as Image classification and Machine translation.
    """
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, inputs):
        return inputs * inputs.sigmoid()
    