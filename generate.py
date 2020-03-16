import os
import fire
import numpy as np
import pickle


def generate(mapper_input=None, labels=None, noises=None, use_mapper=True):
    with open('tset.pkl', 'rb') as test_file:
        model = pickle.load(test_file)

    for i in range(model.label_dim + 1):
        if i == model.label_dim:
            print(f'Overall')
            labels = None
        else:
            print(f'For label {i}')
            labels = np.array([np.eye(model.label_dim)[i] for _ in range(64)])

        model.set_evaluation_parameters(labels_to_evaluate=labels, reset=True)

        y = np.stack([x.cpu().data.numpy() for x, i in model.latents_to_evaluate])[0]
        y_std = np.mean(np.abs(y.std(axis=0)))

        generated_images, average_generated_images = model.evaluate(use_mapper=use_mapper)
        w = model.last_latents.cpu().data.numpy()
        w_std = np.mean(np.abs(w.std(axis=0)))

        x = generated_images.cpu().data.numpy()
        x_std = np.mean(np.abs(x.std(axis=0)))

        print(f"y_std: {y_std}, w_std:{w_std}, x_std:{x_std}")


if __name__ == "__main__":
    fire.Fire(generate)
