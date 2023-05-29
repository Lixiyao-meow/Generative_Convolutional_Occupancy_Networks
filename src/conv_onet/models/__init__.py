import torch
import torch.nn as nn
from torch import distributions as dist
from src.conv_onet.models import decoder

# Decoder dictionary
decoder_dict = {
    'simple_local': decoder.LocalDecoder,
    'simple_local_crop': decoder.PatchLocalDecoder,
    'simple_local_point': decoder.LocalPointDecoder
}


class ConvolutionalOccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, decoder, encoder=None, device=None):
        super().__init__()
        
        self._device = device
        self.encode = encoder.to(device)
        self.decode = decoder.to(device)

    def forward(self, p, inputs, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        c = self.encoder(inputs)
        p_r = self.decoder(p, c, **kwargs)
        return p_r

    def encoder(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''
        return self.encode(inputs)

    def decoder(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decode(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r
