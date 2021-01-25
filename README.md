# Capsule Layers

A pytorch implementation of Primary and Routing Capsule layers, and the encoder (classifier) part of CapsNet, as described in: 'Dynamic Routing Between Capsules' by Sabour, Frosst, Hinton.
https://arxiv.org/abs/1710.09829.

Convolution layers produce heatmaps that signal the presence of certain features in the image.
Capsule Layers extend this notion to include not only the _presence_ of a feature, but also its instantiations properties, e.g. orientation, size, hue, etc.

Therefore, each feature is associated with a vector rather than a scalar.
Such a vector is called a Capsule.
In Capsules the presence of a feature is encoded in the length of the vector, i.e. there's no dedicated 'presence' component (although this was changed in later versions.)

This structure allows to encode an object (=high level feature / class) as a set of (lower level) features, with their relative instantiations.

Such encoding means the network is robust to changes in point of view, lighting conditions, etc.
At inference, each capsule 'votes' for objects that it might be part of.
In doing so it also predicts the instantiations of the objects.
Thus, an object's presence (and instantiation) is encoded in the agreement of votes it recives.
The voting is done via an iterative 2-step rout-by-agreement mechanism (repeated 3 times):
- Each object's instantiation is computed as a weighted sum of incoming capsules' instantiations.
- The weight of each capsule is re-adusted, reinforecing connections between agreeing capsules and objectss.

Ultimately, an objects' presnce is encoded by the norm of its capsule vector.

For more in-depth material:

https://pechyonkin.me/capsules-1

https://blog.paperspace.com/capsule-networks

## Getting Started

The data module provided here is for the mnist csv dataset, which can be downloaded here: [https://www.kaggle.com/oddrationale/mnist-in-csv].
After download, make sure [DATA_DIR] in data.py, points to the folder which contains the csv files. 

Running test.py will load a pre-trained model, and display its predictions for mnist digits.
Running train.py will train a new model on the mnist data, according to your configurations.
