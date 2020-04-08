import os
import fire
from tqdm import tqdm
import json
from pathlib import Path

from CStyleGAN2_pytorch.trainer import Trainer

from CStyleGAN2_pytorch.config import FOLDER, NAME, NEW, LOAD_FROM, GPU, IMAGE_SIZE, CHANNELS, GPU_BATCH_SIZE, \
    GRADIENT_BATCH_SIZE, NETWORK_CAPACITY, NUM_TRAIN_STEPS, LEARNING_RATE, \
    PATH_LENGTH_REGULIZER_FREQUENCY, HOMOGENEOUS_LATENT_SPACE, USE_DIVERSITY_LOSS, SAVE_EVERY, \
    EVALUATE_EVERY, CONDITION_ON_MAPPER, MODELS_DIR, USE_BIASES


# nohup python3.6 run.py "/homelocal/gpu1/pyve/acolin/data/chen/train" --image_size=512 --batch_size=1 --gpu=0 --name=TenGeoP-SARwv_512 > nohup_gpu0.out &
# nohup python3.6 run.py "/homelocal/gpu1/pyve/acolin/data/chen/train" --image_size=256 --batch_size=2 --gpu=1 --name=TenGeoP-SARwv_256 > nohup_gpu1.out &
# nohup python3.6 run.py "/homelocal/gpu1/pyve/acolin/data/chen/train" --image_size=128 --batch_size=8 --gpu=2 --name=TenGeoP-SARwv_128 > nohup_gpu2.out &
# python run.py "E:\datasets\GochiUsa_128\train" --image_size=32 --batch_size=2 --name=GochiUsa_32
# python run.py "E:\datasets\GochiUsa_128\train" --image_size=32 --batch_size=2 --name=GochiUsa_32 --condition_on_mapper=False
# python run.py "E:\datasets\mnist\train" --image_size=32 --batch_size=2 --name=mnist_condition_on_m --channels=1

#  python run.py "E:\datasets\mnist/train" --image_size=32 --name=mnist_condition_on_m --homogenenous_latent_space=False --gpu=0  --gpu_batch_size=8 --channels=1
#  python run.py "E:\datasets\mnist/train" --image_size=32 --name=mnist_condition_on_m_homogeneous --gpu=0   --gpu_batch_size=8 --channels=1
#  python run.py "E:\datasets\mnist/train" --image_size=32 --name=mnist_condition_on_g --condition_on_mapper=False --homogenenous_latent_space=False --gpu=0 --gpu_batch_size=8 --channels=1

#  python run.py "E:\datasets\GochiUsa_128/train" --image_size=64 --name=gochiusa64_condition_on_m --homogenenous_latent_space=False --gpu=0  --gpu_batch_size=4 --channels=3
#  python run.py "E:\datasets\GochiUsa_128/train" --image_size=64 --name=gochiusa64_condition_on_m_homogeneous --gpu=0   --gpu_batch_size=4 --channels=3
#  python run.py "E:\datasets\GochiUsa_128/train" --image_size=64 --name=gochiusa64_condition_on_g --condition_on_mapper=False --homogenenous_latent_space=False --gpu=0 --gpu_batch_size=4 --channels=3

#  nohup python3.6 run.py "E:\datasets\mnist\train" --image_size=32 --batch_size=2 --name=mnist_condition_on_g --condition_on_mapper=False --homogenenous_latent_space=False --gpu=0 --batch_size=32 --channels=1 > nohup_gpu2.out &


def train_from_folder(folder=FOLDER, name=NAME, new=NEW, load_from=LOAD_FROM, image_size=IMAGE_SIZE,
                      gpu_batch_size=GPU_BATCH_SIZE, gradient_batch_size=GRADIENT_BATCH_SIZE,
                      network_capacity=NETWORK_CAPACITY, num_train_steps=NUM_TRAIN_STEPS,
                      learning_rate=LEARNING_RATE, gpu=GPU, channels=CHANNELS,
                      path_length_regulizer_frequency=PATH_LENGTH_REGULIZER_FREQUENCY,
                      homogeneous_latent_space=HOMOGENEOUS_LATENT_SPACE,
                      use_diversity_loss=USE_DIVERSITY_LOSS,
                      save_every=SAVE_EVERY,
                      evaluate_every=EVALUATE_EVERY,
                      condition_on_mapper=CONDITION_ON_MAPPER,
                      use_biases=USE_BIASES):
    """
    Train the conditional stylegan model on the data contained in a folder.

    :param folder: the path to the folder containing either pictures or subfolder with pictures.
    :type folder: str, optional
    :param name: name of the model. The results (pictures and models) will be saved under this name.
    :type name: str, optional
    :param new: True to overwrite the previous results with the same name, else False.
    :type new: bool, optional
    :param load_from: index of the model to import if new is False.
    :type load_from: int, optional
    :param image_size: size of the picture to generate.
    :type image_size: int, optional
    :param gpu_batch_size: size of the batch to enter the GPU.
    :type gpu_batch_size: str, optional
    :param gradient_batch_size: size of the batch on which we compute the gradient.
    :type gradient_batch_size: int, optional
    :param network_capacity: basis for the number of filters.
    :type network_capacity: int, optional
    :param num_train_steps: number of steps to train.
    :type num_train_steps: int, optional
    :param learning_rate: learning rate for the training.
    :type learning_rate: float, optional
    :param gpu: name of the GPU to use, usually '0'.
    :param gpu: int, optional
    :param channels: number of channels of the input images.
    :type channels: str, optional
    :param path_length_regulizer_frequency: frequency of the path length regulizer.
    :type path_length_regulizer_frequency: int
    :param homogeneous_latent_space: choose if the latent space homogeneous or not.
    :type homogeneous_latent_space: bool, optional
    :param use_diversity_loss: penalize the generator by the lack of std for w.
    :type use_diversity_loss: bool, optional
    :param save_every: number of (gradient) batch after which we save the network
    :type save_every: int, optional
    :param evaluate_every: number of (gradient) batch after which we evaluate the network
    :type evaluate_every: int, optional
    :param condition_on_mapper: wether to use the conditions in the mapper or the generator
    :type condition_on_mapper: bool, optional
    :param use_biases: wether to use biases in the mapper or not
    :type use_biases: bool, optional
    :return:
    """
    gradient_accumulate_every = gradient_batch_size // gpu_batch_size
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    json_path = Path(MODELS_DIR / name / 'config.json')
    os.makedirs(Path(MODELS_DIR / name), exist_ok=True)
    if not new and os.path.exists(json_path):
        with open(json_path, 'r') as file:
            config = json.load(file)
    else:
        config = {'name': name,
                  'folder': folder,
                  'batch_size': gpu_batch_size,
                  'gradient_accumulate_every': gradient_accumulate_every,
                  'image_size': image_size,
                  'network_capacity': network_capacity,
                  'lr': learning_rate,
                  'channels': channels,
                  'path_length_regulizer_frequency': path_length_regulizer_frequency,
                  'homogeneous_latent_space': homogeneous_latent_space,
                  'use_diversity_loss': use_diversity_loss,
                  'save_every': save_every,
                  'evaluate_every': evaluate_every,
                  'condition_on_mapper': condition_on_mapper,
                  'use_biases': use_biases
                  }
    model = Trainer(**config)

    if not new:
        model.load(load_from)
    else:
        model.clear()
    with open(json_path, 'w') as file:
        json.dump(config, file, indent=4, sort_keys=True)

    for batch_id in tqdm(range(num_train_steps - model.steps), ncols=60):
        model.train()
        if batch_id % 50 == 0:
            model.print_log(batch_id)


if __name__ == "__main__":
    fire.Fire(train_from_folder)
