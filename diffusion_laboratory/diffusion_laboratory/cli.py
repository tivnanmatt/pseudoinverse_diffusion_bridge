# cli.py

import argparse
import torch
from .config import Config
from .sample import sample_model
from .train import train_model

def main():
    parser = argparse.ArgumentParser(description='Train a diffusion model on a dataset.')

    parser.add_argument('--device', type=str, default='cuda:0', help='The device to use for training.')
    parser.add_argument('--diffusion_model', type=str, default='PIDB', help='The diffusion model to use.')
    parser.add_argument('--imaging_modality', type=str, default='PHOTO', help='The imaging modality to use.')
    parser.add_argument('--batch_size', type=int, default=16, help='The batch size to use for training.')
    parser.add_argument('--num_epochs', type=int, default=10, help='The number of epochs to train for.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='The learning rate to use for training.')
    parser.add_argument('--warmup_steps', type=int, default=10, help='The number of warmup steps to use for training.')
    parser.add_argument('--batches_per_epoch', type=int, default=100, help='The number of batches per epoch to use for training.')
    parser.add_argument('--num_iterations', type=int, default=1000, help='The number of iterations to run.')
    parser.add_argument('--num_reverse_steps', type=int, default=100, help='The number of reverse steps to use for sampling.')
    parser.add_argument('--animation_frames_downsample', type=int, default=1, help='Number of reverse steps per animation frame.')
    parser.add_argument('--train', action='store_true', help='Whether to train the model.')
    parser.add_argument('--no-train', dest='train', action='store_false', help='Whether to not train the model.')
    parser.add_argument('--sample', action='store_true', help='Whether to sample from the model.')
    parser.add_argument('--no-sample', dest='sample', action='store_false', help='Whether to not sample from the model.')
    parser.add_argument('--load', action='store_true', help='Whether to load the model.')


    parser.set_defaults(train=False, sample=False, load=True)
    
    args = parser.parse_args()
    
    # require either train or sample to be set
    assert args.train or args.sample, 'At least one of --train or --sample must be provided'

    config = Config(
        device=torch.device(args.device),
        diffusion_model=args.diffusion_model,
        imaging_modality=args.imaging_modality,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        batches_per_epoch=args.batches_per_epoch,
        num_iterations=args.num_iterations,
        num_reverse_steps=args.num_reverse_steps,
        animation_frames_downsample=args.animation_frames_downsample,
        load=args.load,
        train=args.train,
        sample=args.sample,
    )

    for i in range(config.num_iterations):
        if config.train:
            train_model(config)
        if config.sample:
            sample_model(config)

if __name__ == '__main__':
    main()
    