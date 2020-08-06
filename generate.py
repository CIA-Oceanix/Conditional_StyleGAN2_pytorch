import os
import json
import fire
from trainer import Trainer
import torchvision

root = 'models'
name = "TenGeoP-SARwv_256"

images_to_generate = 20

def generate(use_mapper=True,truncation_trick=1.):
    with open(os.path.join(root, name, 'config.json'), 'r') as file:
        config = json.load(file)
        config['folder'] = config['folder'].replace("/homelocal/gpu1/pyve/acolin/data",
                                                   "E:\datasets").replace('/datasets/GochiUsa_128/GochiUsa_128', 
                                                                          'E:\\datasets\\GochiUsaV2\\train')
    model = Trainer(**config)
    model.load(199, root=root)

    class_number = len(os.listdir(config['folder']))
    for i in range(images_to_generate):
        model.set_evaluation_parameters(labels_to_evaluate=None, reset=True, total=class_number)
        generated_images, average_generated_images = model.evaluate(use_mapper=use_mapper,truncation_trick=truncation_trick)
        
        torchvision.utils.save_image(average_generated_images, 'test.png', nrow=model.label_dim)
        
        for j, im in enumerate(average_generated_images):
            torchvision.utils.save_image(im, f'test\{i}-{j}.png', nrow=model.label_dim)

if __name__ == "__main__":
    fire.Fire(generate)
