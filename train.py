import os
from model import CapsEncoder, MarginLoss
from data import load
import torch as T
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from checkpoints import save_checkpoint
from datetime import datetime

# ------------------------------------------------------------------------------

T.manual_seed(0)

# ------------------------------------------------------------------------------


def train(model, imgs, labels, batch_size=32, epochs=10, val_split=0.1,
          logging_step=10, tb=True):
    """
    Train capsules model.

    - Saves checkpoint to ./checkpoints/<model_name>.pth
    - Writes Tensorboard Summary to ./tb/

    :param model: Model to train
    :param imgs: Tensor, size: (#samples, imgH, imgW, #chans)
    :param labels: Tensor of onehot labels, size: (#samples, #classes)
    :param batch_size: Batch size
    :param epochs: Number of epochs
    :param val_split: Portion of data to use for validation
    :param logging_step: Batch steps for writing checkpoint & Tensorboard
    :param tb: Boolean. Write Tensorboard summary?
    """

    assert imgs.size(0) == labels.size(0)
    assert labels.dim() == 2
    assert imgs.dim() == 4

    model.train()
    opt = T.optim.Adam(model.parameters(), lr=1e-3)
    lossfnc = MarginLoss()

    # init checkpoint and tensorboard
    checkpt_fname = './checkpoints/' + model.name + '.pth'
    best_loss_so_far = None

    if tb:
        tb_writer = SummaryWriter(log_dir='./tb/' + model.name)
        os.makedirs(os.path.dirname(tb_writer.log_dir), exist_ok=True)

    # train/validation split
    sample_idxs = T.randperm(len(imgs))
    n_train = int(len(imgs) * (1 - val_split))
    train_idxs = sample_idxs[:n_train]
    val_idxs = sample_idxs[n_train:]

    print(f"Training {model.name}")
    print(f"Train data: {len(train_idxs)}, Validation data: {len(val_idxs)}")

    num_batches = len(train_idxs) // batch_size

    step = 0
    for epoch in range(epochs):

        # shuffle sample order
        sample_idxs = train_idxs[T.randperm(len(train_idxs))]

        for batch in range(num_batches):

            # indices of current batch's samples
            idxs = sample_idxs[batch * batch_size: (batch + 1) * batch_size]

            # ------------
            # actual training:

            opt.zero_grad()
            preds = model.forward(imgs[idxs])
            loss = lossfnc(preds, labels[idxs])
            loss.backward()
            opt.step()

            # ------------
            # logging:

            print('[Epoch: {:2d}, Batch: {:5d} ({:2d}%)] loss: {:2.5f}'.format(
                epoch + 1, batch + 1, int(100 * (batch + 1) / num_batches),
                loss.item()), end='')

            if step % logging_step == 0:

                loss = loss.item()
                score = _score(labels[idxs], preds)
                print(' score: {:2.2f} '.format(score), end='')

                if tb:
                    tb_writer.add_scalars('Loss', {'train': loss}, step)
                    tb_writer.add_scalars('Score', {'train': score}, step)

                if len(val_idxs) > 0:
                    preds = model.forward(imgs[val_idxs])
                    loss = lossfnc(preds, labels[val_idxs]).item()
                    score = _score(labels[val_idxs], preds)
                    print('Val loss: {:2.4f} score: {:2.2f} '.format(
                        loss, score), end='')

                    if tb:
                        tb_writer.add_scalars('Loss', {'val': loss}, step)
                        tb_writer.add_scalars('Score', {'val': score}, step)

                # note that at this point [loss] is either the training loss
                # or that validation loss, depending on if validation exists.
                if best_loss_so_far is None or loss < best_loss_so_far:
                    best_loss_so_far = loss
                    meta = {'epoch': epoch, 'batch': batch, 'loss': loss}
                    save_checkpoint(checkpt_fname, model, opt, meta)
                    print('- saved checkpoint', end='')

            print('')
            step += 1

            # ------------

    return model


def _score(ytrue, ypred):
    return (T.argmax(ytrue, dim=1) == T.argmax(ypred, dim=1)).to(float).mean()


def run():
    num_classes = 3
    imgs, labels = load('train', num_samples=10000, num_classes=num_classes)
    print(f"Loaded {len(imgs)} imgs. {num_classes} unique labels.")
    model = CapsEncoder(num_classes=num_classes)
    model.name += datetime.now().strftime("%Y%m%d-%H%M%S")
    train(model, imgs, labels, batch_size=16, val_split=0.01)

if __name__ == "__main__":
    run()
