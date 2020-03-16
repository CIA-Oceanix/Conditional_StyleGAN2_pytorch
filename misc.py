import numpy as np

import torch
from torch.autograd import grad as torch_grad


def gradient_penalty(images, output, labels, weight=10):
    """
    Compute the gradient of the output, relatively to the images.

    :param images: input of the network.
    :param output: output of the network.
    :param labels: labels of the input, unused.
    :param weight: factor by wich multiply the penalty.
    :return:
    """
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size()).cuda(),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def image_noise(n, im_size):
    """
    Create random noise of shape (n, im_size, im_size, 1) following the uniform distribution between 0 and 1.

    :param n: first shape of the noise, size of the batch for which we generate the noise.
    :type n: int
    :param im_size: second and third shape of the noise, size of the picture to generate.
    :type im_size: int
    :return: a tensor of shape (n, im_size, im_size, 1) following the uniform distribution between 0 and 1.
    """
    return torch.empty(n, im_size, im_size, 1).uniform_(0., 1.).cuda()


def noise(batch_size, latent_dim):
    """
    Create a random noise of shape (batch_size, latent_dim) following the standard normal distribution.

    :param batch_size: first shape of the noise, size of the batch for we which generate the noise.
    :type batch_size: int
    :param latent_dim: second shape of the noise, size of the latent space of the network
    :type latent_dim: int
    :return: a tensor of shape (batch_size, latent_dim) following the standard normal distribution.
    :rtype: torch.Tensor
    """
    return torch.randn(batch_size, latent_dim).cuda()


def noise_list(n, layers, latent_dim):
    return [(noise(n, latent_dim), layers)]


def mixed_list(n, layers, latent_dim):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim) + noise_list(n, layers - tt, latent_dim)


def latent_to_w(style_vectorizer, standard_random_vectors, labels):
    """
    Convert the standard_random_vectors to the latent_vector, conditionnaly to the labels.

    :param style_vectorizer: the nn.Module converting the random vectors into the latent ones.
    :type style_vectorizer: StyleGAN2.StyleVectorizer
    :param standard_random_vectors: the tensor given by the standard normal distribution
    :type standard_random_vectors: torch.Tensor
    :param labels: the conditions of the vectors
    :type labels: torch.Tensor
    :return: the latent vector
    :rtype: torch.Tensor
    """
    return [(style_vectorizer(z, labels), num_layers) for z, num_layers in standard_random_vectors]


def get_random_labels(batch_size, label_dim, label_index=None):
    """
    Generate a random one-hot-encoded vector of label.

    :param batch_size: first shape of the generated vector, size of the batch for which generate the labels.
    :type batch_size: int
    :param label_dim: second shape of the generated vector, total number of labels.
    :type label_dim: int
    :param label_index: If not False, force the output to be of a specific label (non-random).
    :type bool or int. Default is False.
    :return: a tensor of random one-hot-encoded labels of shape (n, label_dim)
    :rtype: torch.Tensor
    """
    if label_dim == 1:
        labels = np.ones((batch_size, 1))
    else:
        if isinstance(label_index, bool) and not label_index:
            labels = np.eye(label_dim)[np.random.randint(0, label_dim, size=batch_size)]
        else:
            labels = np.eye(label_dim)[label_index]
    return torch.from_numpy(labels).cuda().float()


def evaluate_in_chunks(max_batch_size, model, *args):
    """
    Evaluate the args by dividing in batchs

    :param max_batch_size: the maximimum size of the created batchs
    :type max_batch_size: int
    :param model: the model used to evaluate
    :type model: nn.Module
    :param args: the inputs to enter the model
    :type args: torch.Tensor
    :return: the outputs of the model on the given inputs
    :rtype: torch.Tensor
    """
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)


def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)


def set_requires_grad(model, boolean):
    for p in model.parameters():
        p.requires_grad = boolean


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
