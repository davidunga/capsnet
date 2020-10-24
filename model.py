import torch as T
import torch.nn.functional as F
from capsule_layers import PrimaryCapsules, RoutingCapsules

# ------------------------------------------------------------------------------


class CapsEncoder(T.nn.Module):
    """ The encoder (classifier) part of CapsNet, from:
        'Dynamic Routing Between Capsules', Sabour, Frosst, Hinton
        https://arxiv.org/abs/1710.09829
    """

    def __init__(self, num_classes=10):
        super(CapsEncoder, self).__init__()

        self.name = 'CapsEncoder'

        self.conv1 = T.nn.Conv2d(in_channels=1,
                                 out_channels=256,
                                 stride=1,
                                 kernel_size=9)

        self.primary_caps = PrimaryCapsules(
            in_channels=256,
            out_channels=32,
            capsule_size=8,
            kernel_size=9,
            stride=2
        )

        self.digit_caps = RoutingCapsules(
            in_capsules_num=6 * 6 * 32,
            in_capsule_size=8,
            capsule_size=16,
            capsules_num=num_classes
        )

        pass

    def forward(self, x):
        """
        :param x: (batch, H, W, channel)
        """

        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = F.relu(x)
        x = x.permute(0, 2, 3, 1)

        x = self.primary_caps(x)
        x = self.digit_caps(x)
        x = T.norm(x, dim=-1)

        return x

# ------------------------------------------------------------------------------


class MarginLoss(T.nn.Module):

    def __init__(self, m_minus=.1, m_plus=.9, lmbda=.5):
        super(MarginLoss, self).__init__()
        assert 0 < m_minus < m_plus < 1
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lmbda = lmbda
        pass

    def forward(self, preds, targets):
        loss_p = targets * F.relu(self.m_plus - preds) ** 2
        loss_m = (1 - targets) * self.lmbda * F.relu(preds - self.m_minus) ** 2
        loss = (loss_p + loss_m).mean()
        return loss

# ------------------------------------------------------------------------------

def test():
    from data import load

    print("Testing capsnest model...")

    num_classes = 3
    imgs, labels = load('train', num_samples=50, num_classes=num_classes)
    print(f"Loaded {len(imgs)} imgs.")
    
    print("Initializing model")
    model = CapsEncoder(num_classes=num_classes)
    print("Running images through model")

    preds = model.forward(imgs)

    print(f"Done. Input size={imgs.size()}, Output size={preds.size()}")

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    test()
    pass
