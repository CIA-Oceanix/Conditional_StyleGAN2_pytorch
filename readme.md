# A conditional PyTorch StyleGAN2

This repository is an heavily modified/refactored version of [lucidrains](https://github.com/lucidrains)'s [stylegan2-pytorch](https://github.com/lucidrains/stylegan2-pytorch).

The original version of [StyleGAN2](https://github.com/NVlabs/stylegan2) can be found [here](https://arxiv.org/abs/1912.04958).

Check [our pdf](https://github.com/CIA-Oceanix/Conditional_StyleGAN2_pytorch/blob/master/A_Conditional_Pytorch_StyleGan_V2.pdf) for details about the conditioning.

## Abstract

We release a PyTorch implementation of the second version of the StyleGan2 architecture.  With the addition of a conditional component to the latent vector, we enable the possibility to create pictures from a categorization dataset, by choosing the class of the generated sample.

## Table of Content

- [Abstract](#Abstract)
- [Table of Content](#Table-of-Content)
- [Requirements](#Requirements)
- [How to Use](#How-to-Use)
    - [Train the Model](#Train-the-Model)
    - [File system configuration](#File-system-configuration)
    - [Use the trained model](#Use-the-trained-model)

## Requirements

The script has been tested with Python 3.6.9 and the following packages:

- os
- glob
- shutil
- json
- random
- math
- pathlib (2.3.4)
- numpy (1.17.0)
- PIL (1.1.7) 
- tqdm (4.32.1)
- fire (0.2.1)
- torch (1.2.0+cu92)
- torchvision (0.4.0+cu92)
 
## How to Use


### Train the Model

Running `python run.py --help` returns the following result:

```
NAME    
    run.py - Train the conditional stylegan model on the data contained in a folder.

SYNOPSIS
    run.py <flags>

DESCRIPTION 
    Train the conditional stylegan model on the data contained in a folder.
    
FLAGS
    --folder=FOLDER
        the path to the folder containing either pictures or subfolder with pictures.
    --name=NAME
        name of the model. The results (pictures and models) will be saved under this name.
    --new=NEW
        True to overwrite the previous results with the same name, else False.
    --load_from=LOAD_FROM
        index of the model to import if new is False.
    --image_size=IMAGE_SIZE
        size of the picture to generate.
    --gpu_batch_size=GPU_BATCH_SIZE
        size of the batch to enter the GPU.
    --gradient_batch_size=GRADIENT_BATCH_SIZE
        size of the batch on which we compute the gradient.
    --network_capacity=NETWORK_CAPACITY
        basis for the number of filters.
    --num_train_steps=NUM_TRAIN_STEPS
        number of steps to train.
    --learning_rate=LEARNING_RATE
        learning rate for the training.
    --gpu=GPU
        name of the GPU to use, usually '0'.
    --channels=CHANNELS
        number of channels of the input images.
    --path_length_regulizer_frequency=PATH_LENGTH_REGULIZER_FREQUENCY
        frequency of the path length regulizer.
    --homogeneous_latent_space=HOMOGENEOUS_LATENT_SPACE
        choose if the latent space homogeneous or not.
    --use_diversity_loss=USE_DIVERSITY_LOSS
        penalize the generator by the lack of std for w.
    --save_every=SAVE_EVERY
        number of (gradient) batch after which we save the network.
    --evaluate_every=EVALUATE_EVERY
        number of (gradient) batch after which we evaluate the network.
    --condition_on_mapper=CONDITION_ON_MAPPER
        whether to use the conditions in the mapper or the generator.
    --use_biases=USE_BIASES
        whether to use biases in the mapper or not.
    --label_epsilon=LABEL_EPSILON
        epsilon for the discriminator.
```

### File system configuration

#### Requirements for the dataset

The dataset is read by listing all the subfolders in the provided folder argument. These subfolders will be the labels. 
The picture must be directly inside these folders and not at higher depth.

Suppose that you run the line `python run.py "E:\\mnist" --image_size=32 --channels=1 --name=mnist`:
- E:
    - mnist
        - *0*
            - *1.png*
            - *2.png*
            - [...]
        - *1*
        - [...]
        
The files/folders in italic do not require particular naming.
                
If, for example by using a dataset with 10k or 100k files per label, you have to add a level of depth, you will need to modify the dataset.Dataset class appropriately.

#### Location of the outputs

By default, the outputs folders will be located in the same folder *run.py*. There will two kinds of outputs:
- **the configuration and the checkpoints**, named *./{MODELS_DIR}/{NAME}/config.json* and *./{MODELS_DIR}/{NAME}/model_{i}.pt* where {MODELS_DIR} (by default *models*) is the variable of the same name defined in *config.py*, *{NAME}* the argument in the command line and *{i}* the index of the checkpoint, which is the batch index divided by the *SAVE_EVERY* variable defined in *config.py*.
- **the generated samples**, named *./{RESULTS_DIR}/{NAME}/{i}.png* and  *./{RESULTS_DIR}/{NAME}/{i}-EMA.png* where *{RESULTS_DIR} (by default *results*) is the variable of the same name in *config.py*, *{NAME}* the argument in the command line and *{i}* is the index of the samples, which is the batch index divided by the *EVALUATE_EVERY* variable defined in *config.py*.


### Use the trained model

When the models are trained, you can impor the weights and the checkpoint quite easily

```python
root = 'models'  # what you used as MODEL_DIR
name = 'mnist'  # what you used as name

with open(os.path.join(root, name, 'config.json'), 'r') as file:
    config = json.load(file)
model = Trainer(**config)
model.load(-1, root=root)  # the first argument is the index of the checkpoint, -1 means the last checkpoint


model.set_evaluation_parameters(labels_to_evaluate=None, reset=True, total=64)  # you can set the latents, the noise or the labels
generated_images, average_generated_images = model.evaluate()
```

If you want to play a little, you can find a pre-trained model in *./models/gochiusa64*.