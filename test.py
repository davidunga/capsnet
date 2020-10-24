import matplotlib.pyplot as plt
import torch as T
from data import load
from checkpoints import load_checkpoint, save_checkpoint


def test(model, imgs):
    """
    Show model's predictions
    """

    for i in range(imgs.size(0)):
        pred = model.forward(imgs[i:i+1])
        pred_label = int(T.argmax(pred))

        plt.imshow(imgs[i][..., 0], cmap='gray')
        plt.title('Prediction=' + str(pred_label))
        plt.draw()
        try:
            plt.waitforbuttonpress()
        except:
            break


if __name__ == "__main__":
    """
    Load trained model and show predictions
    """
    model, _, _ = load_checkpoint('checkpoints/CapsEncoder_mnist.pth')
    imgs, _ = load('test', num_samples=500)
    test(model, imgs)
