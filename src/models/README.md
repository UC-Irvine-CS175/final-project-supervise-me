# Congratulations! :confetti_ball:
While exploring the hw repository you have happened to stop in the `supervised` sample code directory.

## PyTorch's `nn.Module`
`lenet_scratch.py` has been kept in the repository as an example of logging loss and accuracy using Weights and Biases and showing you the full capabilities of using PyTorch and PyTorch Lightning.
LeNet-5 is a classic convolutional neural network (CNN) architecture that was introduced by Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner in 1998. It was primarily designed for handwritten digit recognition tasks, such as recognizing digits in the MNIST dataset. LeNet-5 played a crucial role in popularizing deep learning and paved the way for many modern CNN architectures.

![LeNet5](../../../documentation/images/lenet.png)

Although modeling from scratch is generally more expensive in terms of computational resources and time, it is good to know how to do it if you need to. Notice that we must inherit from the PyTorch `nn.Module` to define the Classifier. In this derived class we must implement the `__init__` function and the `forward` function. The constructor allows use to define the layer architecture comprising of convolutional, pooling, and fully connected layers. The `forward` method chains the layers together with activation functions retrieved from PyTorch's `torch.nn.functional` library.

## PyTorch Lightning's `pl.LightningModule`
To remove the overhead of boilerplate code that often pollutes the main functions of many a naive machine learning researcher, PyTorch Lightning provides an organized way of approaching the training and validation loop. By inheriting from `pl.LightningModule` you can create a class that calls the model defined using `nn.Module` with additional Accuracy metrics in the `__init__`. In addition to `__init__` you must also implement the following member functions:

- `forward`
- `training_step`
- `validation_step`
- `test_step`
- `configure_optimizers`

For more information on how to use PyTorch Lightning with PyTorch as well as helpful tutorials, see:

- [PyTorch Lightning: Basic Skills](https://lightning.ai/docs/pytorch/latest/levels/core_skills.html)
- [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/)
