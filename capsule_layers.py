"""
Code for Capsule Layers, described in:
'Dynamic Routing Between Capsules', Sabour, Frosst, Hinton
https://arxiv.org/abs/1710.09829
"""

import torch as T
import torch.nn.functional as F

# ------------------------------------------------------------------------------


class PrimaryCapsules(T.nn.Module):
    """ Primary Capsules Layer.
    Essentially convolution with a squash() non-linearity.

    :param in_channels: Number of input channels
    :param out_channels: Number of output channels
    :param capsule_size: Dimension of layer capsules
    :param kernel_size: Convolution kernel size
    :param stride: Convolution stride size
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 capsule_size,
                 kernel_size=3,
                 stride=1
                 ):

        super(PrimaryCapsules, self).__init__()

        self.capsule_size = capsule_size
        self.out_channels = out_channels
        self.conv = T.nn.Conv2d(in_channels=in_channels,
                                out_channels=capsule_size * out_channels,
                                kernel_size=kernel_size,
                                stride=stride)
        pass


    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size()[:3] + (self.capsule_size, self.out_channels))
        x = squash(x)
        return x

# ------------------------------------------------------------------------------


class RoutingCapsules(T.nn.Module):
    """ Routing Capsules Layer

    :param in_capsules_num: Number of incoming capsules
    :param in_capsule_size: Dimension of incoming capsules
    :param capsules_num: Number of capsules in layer
    :param capsule_size: Dimension of capsules in layer
    :param num_iterations: Number of routing iteration
    :param w_init_sigma: Variance of weights initialization
    """

    def __init__(self,
                 in_capsules_num,
                 in_capsule_size,
                 capsules_num,
                 capsule_size,
                 num_iterations=None,
                 w_init_sigma=None,
                 ):

        super(RoutingCapsules, self).__init__()

        # set defaults:
        self.num_iterations = 3 if num_iterations is None else num_iterations
        w_init_sigma = 0.1 if w_init_sigma is None else w_init_sigma

        # initialize:
        self.capsules_num = capsules_num
        self.capsule_size = capsule_size
        self.in_capsules_num = in_capsules_num
        self.in_capsule_size = in_capsule_size

        # W[i, j] is a matrix that transforms vector from (incoming) capsule i,
        # to a vector in (this layer's) capsule j:
        self.W = T.nn.Parameter(w_init_sigma * T.randn(size=(
            self.capsules_num, self.in_capsules_num,
            self.capsule_size, self.in_capsule_size)), requires_grad=True)

        pass

    def forward(self, u):
        """
        :param u: size =
            either: (batch, H, W, input_caps_size, input_chans)
            or:     (batch, num_input_caps, input_caps_size)
        :return: size = (batch, num_caps, caps_size)
        """

        # compute: u_hat{j|i} is the projection of the i-th incoming capsule,
        # to the j-th capsule, according to the transformation matrix W[i->j]:
        u = u.reshape(-1, 1, self.in_capsules_num, self.in_capsule_size, 1)
        uhat = T.matmul(self.W.unsqueeze(0), u).squeeze()
        uhat_nograd = uhat.detach() # size = (batch, num input caps, caps_size)

        # determine routing coefficients:
        b = T.zeros(u.size(0), self.capsules_num, self.in_capsules_num, 1)
        for _ in range(self.num_iterations - 1):
            c = F.softmax(b, dim=1)
            v = squash((c * uhat_nograd).sum(2))  # (batch, num caps, caps size)
            b += (v.unsqueeze(2) * uhat_nograd).sum(3).unsqueeze(-1)

        # compute output:
        c = F.softmax(b, dim=1)
        v = squash((c * uhat).sum(2))  # (batch, num caps, caps size)

        return v


# ------------------------------------------------------------------------------

def squash(s):
    """ Shrinks vector to maintain norm between 0 & 1.
    Scale factor = |s| / (1+|s|)
    """
    norm2 = T.sum(s ** 2, dim=-1, keepdim=True)
    v = (T.sqrt(norm2) / (1 + norm2)) * s
    return v

