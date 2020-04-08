import os
from random import random
from shutil import rmtree
from pathlib import Path
import numpy as np

import torch
from torch.utils import data
import torch.nn.functional as F

import torchvision

from CStyleGAN2_pytorch.dataset import cycle, Dataset
from CStyleGAN2_pytorch.StyleGAN2 import StyleGAN2
from CStyleGAN2_pytorch.misc import gradient_penalty, image_noise, noise_list, mixed_list, latent_to_w, \
    evaluate_in_chunks, styles_def_to_tensor, EMA

from CStyleGAN2_pytorch.config import RESULTS_DIR, MODELS_DIR, EPSILON, LOG_FILENAME, GPU_BATCH_SIZE, LEARNING_RATE, \
    PATH_LENGTH_REGULIZER_FREQUENCY, HOMOGENEOUS_LATENT_SPACE, USE_DIVERSITY_LOSS, SAVE_EVERY, EVALUATE_EVERY, CHANNELS, \
    CONDITION_ON_MAPPER, MIXED_PROBABILITY, GRADIENT_ACCUMULATE_EVERY, MOVING_AVERAGE_START, MOVING_AVERAGE_PERIOD, \
    USE_BIASES


class Trainer():
    def __init__(self, name, folder, image_size, batch_size=GPU_BATCH_SIZE, mixed_prob=MIXED_PROBABILITY,
                 lr=LEARNING_RATE, channels=CHANNELS, path_length_regulizer_frequency=PATH_LENGTH_REGULIZER_FREQUENCY,
                 homogeneous_latent_space=HOMOGENEOUS_LATENT_SPACE, use_diversity_loss=USE_DIVERSITY_LOSS,
                 save_every=SAVE_EVERY, evaluate_every=EVALUATE_EVERY, condition_on_mapper=CONDITION_ON_MAPPER,
                 gradient_accumulate_every=GRADIENT_ACCUMULATE_EVERY, moving_average_start=MOVING_AVERAGE_START,
                 moving_average_period=MOVING_AVERAGE_PERIOD, use_biases=USE_BIASES,
                 *args, **kwargs):
        self.condition_on_mapper = condition_on_mapper
        self.folder = folder
        self.label_dim = len([subfolder for subfolder in os.listdir(folder)
                              if os.path.isdir(os.path.join(folder, subfolder))])
        if not self.label_dim:
            self.label_dim = 1

        self.name = name
        self.GAN = StyleGAN2(lr=lr, image_size=image_size, label_dim=self.label_dim, channels=channels,
                             condition_on_mapper=self.condition_on_mapper, *args, **kwargs)
        self.GAN.cuda()

        self.batch_size = batch_size
        self.lr = lr
        self.mixed_prob = mixed_prob
        self.steps = 0
        self.save_every = save_every
        self.evaluate_every = evaluate_every

        self.av = None
        self.path_length_mean = 0
        self.moving_average_start = moving_average_start
        self.moving_average_period = moving_average_period

        self.dataset = Dataset(folder, image_size)
        self.loader = cycle(data.DataLoader(self.dataset, num_workers=0, batch_size=batch_size,
                                            drop_last=True, shuffle=False, pin_memory=False))
        self.gradient_accumulate_every = gradient_accumulate_every

        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = 0

        self.path_length_moving_average = EMA(0.99)
        self.path_length_regulizer_frequency = path_length_regulizer_frequency
        self.homogeneous_latent_space = homogeneous_latent_space
        self.use_diversity_loss = use_diversity_loss
        self.init_folders()

        self.labels_to_evaluate = None
        self.noise_to_evaluate = None
        self.latents_to_evaluate = None

        self.evaluate_in_chunks = evaluate_in_chunks
        self.styles_def_to_tensor = styles_def_to_tensor

    def train(self):
        self.GAN.train()
        if not self.steps:
            self.draw_reals()
        total_disc_loss = torch.tensor(0.).cuda()
        total_gen_loss = torch.tensor(0.).cuda()

        batch_size = self.batch_size

        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim if self.condition_on_mapper else self.GAN.G.latent_dim - self.label_dim
        num_layers = self.GAN.G.num_layers

        apply_gradient_penalty = self.steps % 4 == 0
        apply_path_penalty = self.steps % self.path_length_regulizer_frequency == 0

        # train discriminator

        average_path_length = self.path_length_mean
        self.GAN.D_opt.zero_grad()
        inputs = []

        for i in range(self.gradient_accumulate_every):
            image_batch, label_batch = next(self.loader)

            get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
            style = np.array(get_latents_fn(batch_size, num_layers, latent_dim))
            noise = image_noise(batch_size, image_size)
            inputs.append((style, noise, label_batch))

            w_space = latent_to_w(self.GAN.S, style, label_batch)
            w_styles = self.styles_def_to_tensor(w_space)

            generated_images = self.GAN.G(w_styles, noise, label_batch)
            fake_output = self.GAN.D(generated_images.clone().detach(), label_batch)

            image_batch = image_batch.cuda()
            image_batch.requires_grad_()
            real_output = self.GAN.D(image_batch, label_batch)
            divergence = (F.relu(1 + real_output) + F.relu(1 - fake_output))
            divergence = divergence.mean()
            disc_loss = divergence

            if apply_gradient_penalty:
                gp = gradient_penalty(image_batch, real_output, label_batch)
                self.last_gp_loss = gp.clone().detach().item()
                disc_loss = disc_loss + gp

            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.backward()

            total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

        self.d_loss = float(total_disc_loss)
        self.GAN.D_opt.step()

        # train generator

        self.GAN.G_opt.zero_grad()
        if self.use_diversity_loss:
            labels = np.array([np.eye(self.label_dim)[np.random.randint(self.label_dim)]
                               for _ in range(8 * self.label_dim)])
            self.set_evaluation_parameters(labels_to_evaluate=labels, reset=True)
            self.evaluate()
            w = self.last_latents.cpu().data.numpy()
            w_std = np.mean(np.abs(0.25 - w.std(axis=0)))
        else:
            w_std = 0

        for i in range(self.gradient_accumulate_every):
            style, noise, random_label = inputs[i]

            w_space = latent_to_w(self.GAN.S, style, random_label)
            w_styles = self.styles_def_to_tensor(w_space)

            generated_images = self.GAN.G(w_styles, noise, random_label)
            fake_output = self.GAN.D(generated_images, random_label)
            loss = fake_output.mean()
            generator_loss = loss

            if self.homogeneous_latent_space and apply_path_penalty:
                std = 0.1 / (w_styles.std(dim=0, keepdims=True) + EPSILON)
                w_styles_2 = w_styles + torch.randn(w_styles.shape).cuda() / (std + EPSILON)
                path_length_images = self.GAN.G(w_styles_2, noise, random_label)
                path_lengths = ((path_length_images - generated_images) ** 2).mean(dim=(1, 2, 3))
                average_path_length = np.mean(path_lengths.detach().cpu().numpy())

                if self.path_length_mean is not None:
                    path_length_loss = ((path_lengths - self.path_length_mean) ** 2).mean()
                    if not torch.isnan(path_length_loss):
                        generator_loss = generator_loss + path_length_loss

            generator_loss = (generator_loss + w_std) / self.gradient_accumulate_every
            generator_loss.backward()
            total_gen_loss += loss.detach().item() / self.gradient_accumulate_every

        self.g_loss = float(total_gen_loss)
        self.GAN.G_opt.step()

        # calculate moving averages

        if apply_path_penalty and not np.isnan(average_path_length):
            self.path_length_mean = self.path_length_moving_average.update_average(self.path_length_mean,
                                                                                   average_path_length)

        if self.steps == self.moving_average_start:
            self.GAN.reset_parameter_averaging()
        if self.steps % self.moving_average_period == 0 and self.steps > self.moving_average_start:
            self.GAN.EMA()

        if not self.steps % self.save_every:
            self.save(self.steps // self.save_every)

        if not self.steps % self.evaluate_every:
            self.set_evaluation_parameters()
            generated_images, average_generated_images = self.evaluate()
            self.save_images(generated_images, f'{self.steps // self.evaluate_every}.png')
            self.save_images(generated_images, 'fakes.png')
            self.save_images(average_generated_images, f'{self.steps // self.evaluate_every}-EMA.png')
        self.steps += 1
        self.av = None

    def set_evaluation_parameters(self, latents_to_evaluate=None, noise_to_evaluate=None, labels_to_evaluate=None,
                                  num_rows='labels', num_cols=8, reset=False, total=None):
        """
        Set the latent vectors, the noises and the labels to evaluate, convert them to tensor, cuda and float if needed

        :param latents_to_evaluate: the latent vector to enter (either the mapper or the generator) network.
                                    If None, they will be sampled from standart normal distribution.
        :type latents_to_evaluate: torch.Tensor or np.ndarray, optional, default at None.
        :param noise_to_evaluate: the noise to enter the generator, convert them to tensor, cuda and float if needed.
                                  If None, they will be sampled from standard normal distribution.
        :type noise_to_evaluate: torch.Tensor or np.ndarray, optional, default at None.
        :param labels_to_evaluate: the labels to enter the mapper, convert them to tensor, cuda and float if needed
                                   If None, add all the label one after another.
        :type labels_to_evaluate: torch.Tensor or np.darray, optional, default at None
        :param num_rows: number of rows in the generated mosaic.
                         Only needed to compute the size of other parameters when they are at None.
        :type num_rows: int, optional, default at 'labels' (transformed to the number of labels).
        :param num_cols: number of columns in the generated mosaic.
                         Only needed to compute the size of other parameters when they are at None.
        :type num_cols: int, optional, default at 8
        :param total: bypass the num_cols and num_rows to choose the total number of imgs
        :type total: int, optional, default is None
        """
        if num_rows == 'labels':
            num_rows = self.label_dim
        if num_cols == 'labels':
            num_cols = self.label_dim
        if total is None:
            total = num_cols * num_rows

        if latents_to_evaluate is None:
            if self.latents_to_evaluate is None or reset:
                latent_dim = self.GAN.G.latent_dim if self.condition_on_mapper else self.GAN.G.latent_dim - self.label_dim
                self.latents_to_evaluate = noise_list(total, self.GAN.G.num_layers, latent_dim)
        else:
            self.latents_to_evaluate = latents_to_evaluate

        if noise_to_evaluate is None:
            if self.noise_to_evaluate is None or reset:
                self.noise_to_evaluate = image_noise(total, self.GAN.G.image_size)
        else:
            self.noise_to_evaluate = noise_to_evaluate
        if isinstance(self.latents_to_evaluate, np.ndarray):
            self.noise_to_evaluate = torch.from_numpy(self.noise_to_evaluate).cuda().float()

        if labels_to_evaluate is None:
            if self.labels_to_evaluate is None or reset:
                self.labels_to_evaluate = np.array([np.eye(num_rows)[i % num_rows] for i in range(total)])
        else:
            self.labels_to_evaluate = labels_to_evaluate
        if isinstance(self.labels_to_evaluate, np.ndarray):
            self.labels_to_evaluate = torch.from_numpy(self.labels_to_evaluate).cuda().float()

    @torch.no_grad()
    def evaluate(self, use_mapper=True):
        self.GAN.eval()

        def generate_images(stylizer, generator, latents, noise, labels):
            if use_mapper:
                latents = latent_to_w(stylizer, latents, labels)
                latents = styles_def_to_tensor(latents)
            self.last_latents = latents  # for inspection purpose

            generated_images = self.evaluate_in_chunks(self.batch_size, generator, latents, noise, labels)
            generated_images.clamp_(0., 1.)
            return generated_images

        generated_images = generate_images(self.GAN.S, self.GAN.G,
                                           self.latents_to_evaluate, self.noise_to_evaluate, self.labels_to_evaluate)
        average_generated_images = generate_images(self.GAN.SE, self.GAN.GE,
                                                   self.latents_to_evaluate, self.noise_to_evaluate,
                                                   self.labels_to_evaluate)
        return generated_images, average_generated_images

    def save_images(self, generated_images, filename):
        torchvision.utils.save_image(generated_images, str(RESULTS_DIR / self.name / filename),
                                     nrow=self.label_dim)

    def draw_reals(self):
        nrows = 8
        reals_filename = str(RESULTS_DIR / self.name / 'reals.png')
        images = [image
                  for images, labels in [next(self.loader)
                                         for _ in range(self.label_dim * nrows // self.batch_size)]
                  for image in images]
        images = images[:len(images) // nrows * nrows]
        torchvision.utils.save_image(images, reals_filename, nrow=len(images) // nrows)
        print(f'\nMosaic of real images created at {reals_filename}\n')

    def print_log(self, batch_id):
        if batch_id == 0:
            with open(LOG_FILENAME, 'w') as file:
                file.write('G;D;GP;PL\n')
        else:
            with open(LOG_FILENAME, 'a') as file:
                file.write(f'{self.g_loss:.2f};{self.d_loss:.2f};{self.last_gp_loss:.2f};{self.path_length_mean:.2f}\n')

    def model_name(self, num, root=MODELS_DIR):
        if isinstance(root, str):
            root = Path(root)
        return str(root / self.name / f'model_{num}.pt')

    def init_folders(self):
        (RESULTS_DIR / self.name).mkdir(parents=True, exist_ok=True)
        (MODELS_DIR / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(RESULTS_DIR / self.name)
        rmtree(MODELS_DIR / self.name)
        self.init_folders()

    def save(self, num):
        torch.save(self.GAN.state_dict(), self.model_name(num))

    def load(self, num=-1, root=MODELS_DIR):
        if isinstance(root, str):
            root = Path(root)
        name = num
        if num == -1:
            file_paths = [p for p in Path(root / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'Continuing from previous epoch - {name}')
        self.steps = name * self.save_every
        self.GAN.load_state_dict(torch.load(self.model_name(name, root=root)))
