import torch as T
import os


def load_checkpoint(fname):
    """ Load model and optimizer state """

    xx = T.load(fname)
    model = xx['model']
    model.load_state_dict(xx['state_dict'])
    optimizer_state = xx['optimizer_state']
    meta = xx['meta']
    return model, optimizer_state, meta


def save_checkpoint(fname, model, optimizer, meta=None):
    """ Save model and optimizer state """

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    T.save({'model': model, 'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(), 'meta': meta}, fname)

