import tqdm.auto as tqdm
import numpy as np
import PIL.Image

import torch


def get_hot_one_encoded(index, index_max=10):
    array = np.zeros(index_max)
    array[index] = 1
    return array


def optimizer_step(generator, optimizer, loss_function, target, parameters):
    optimizer.zero_grad()

    output = generator(*parameters)
    output = output/output.max()

    loss = loss_function(output, target)
    loss.backward()
    optimizer.step()
    return loss.item(), output


def latent_vector_optimization(generator, mapper, image_filename, label, learning_rate=1e-2, steps=1000,
                               use_tqdm=False, optimize_on=(True, False, False)):
    im = PIL.Image.open(image_filename)
    im = im.resize((generator.image_size, generator.image_size))
    
    im = np.array(im)/255
    if len(im.shape) == 3:
        im = np.transpose(im, [2, 0, 1])
    else:
        im = np.expand_dims(im, axis=0)
    im = np.expand_dims(im, axis=0)
    
    target = torch.from_numpy(im).cuda().float()

    label_array = np.expand_dims(get_hot_one_encoded(label, index_max=mapper.label_dim), axis=0)
    label = torch.from_numpy(label_array).cuda().float()

    noise = torch.from_numpy(np.random.random((1, generator.image_size, generator.image_size, 1))).cuda().float()
    latent = torch.from_numpy(np.random.random((1, generator.latent_dim))).cuda().float()
    latent = mapper(latent, label)
    latent = torch.stack([latent for _ in range(generator.num_layers)], axis=1)

    var_noise = torch.autograd.Variable(noise.data, requires_grad=True)
    var_latent = torch.autograd.Variable(latent.data, requires_grad=True)
    var_label = torch.autograd.Variable(label.data, requires_grad=True)

    optimizer = torch.optim.SGD([variable for use, variable in zip(optimize_on, [var_latent, var_noise, var_label])],
                                lr=learning_rate)

    loss_function = torch.nn.MSELoss(reduction='sum')

    steps = tqdm.tqdm(range(steps)) if use_tqdm else range(steps)
    for _ in steps:
        loss, output = optimizer_step(generator, optimizer, loss_function, target, (var_latent, var_noise, var_label))

    return (var_latent, var_noise, label), np.transpose(output.cpu().detach().numpy()[0], [1, 2, 0]), \
           loss
