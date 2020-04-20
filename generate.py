import os
import json
import fire
from CStyleGAN2_pytorch.trainer import Trainer
import torchvision

root = 'models'
name = "GochiUsa64_with_bias"

def generate(use_mapper=True,truncation_trick=1.):
    with open(os.path.join(root, name, 'config.json'), 'r') as file:
        config = json.load(file)
        config['folder'] = config['folder'].replace("/homelocal/gpu1/pyve/acolin/data",
                                                   "E:\datasets").replace('/datasets/GochiUsa_128/GochiUsa_128', 
                                                                          'E:\\datasets\\GochiUsaV2\\train')
    model = Trainer(**config)
    model.load(-1, root=root)


    model.set_evaluation_parameters(labels_to_evaluate=None, reset=True, total=64)
    generated_images, average_generated_images = model.evaluate(use_mapper=use_mapper,truncation_trick=truncation_trick)
    
    torchvision.utils.save_image(average_generated_images, 'test.png', nrow=model.label_dim)

if __name__ == "__main__":
    fire.Fire(generate)
