# config.py

import os
import yaml
import torch
import numpy as np

from .datasets import (
    SynthRad2023Task1MRISlice_Dataset,
    ADNIPETSlice_Dataset,
    CelebA_Dataset as PHOTO_Dataset,
    SynthRad2023Task1CTSlice_Dataset,
    MICRONS_Dataset as SEM_Dataset
)

from .measurement_models import (
    LinearSystemPlusGaussianNoise,
    DownSampleMeanSystemResponse,
    ScalarNoiseCovariance,
    GaussianFilterMeanSystemResponse,
    DenseSVDMeanSystemResponse,
    IdentityMeanSystemResponse,
    UndersampledMRIMeanSystemResponse,
    MeanResponseAppliedToWhiteNoiseCovariance,
    StationaryNoiseCovariance
)

from .diffusion_models import (
    PseudoInverseDiffusionModel, 
    NullspaceDiffusionModel
)

from .networks import DiffusersUnet

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'..')
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, 'weights')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
ANIMATIONS_DIR = os.path.join(PROJECT_ROOT, 'animations')
DATA_DIR = os.path.join(PROJECT_ROOT, '../../../data/')

class Config:
    def __init__(self, device, diffusion_model, imaging_modality, batch_size, num_epochs, learning_rate, warmup_steps, batches_per_epoch, num_iterations, num_reverse_steps, load, train, sample):
        self.device = device
        self.diffusion_model = diffusion_model
        self.imaging_modality = imaging_modality
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.batches_per_epoch = batches_per_epoch
        self.num_iterations = num_iterations
        self.num_reverse_steps = num_reverse_steps
        self.load = load
        self.train = train
        self.sample = sample
        self.setup_imaging_modality()
        self.setup_diffusion_model()

    def setup_imaging_modality(self):
        if self.imaging_modality == 'MRI':
            self.dataset = SynthRad2023Task1MRISlice_Dataset(os.path.join(DATA_DIR, 'SynthRad2023/Task1/brain/'), verbose=True)
            self.mean_response = UndersampledMRIMeanSystemResponse(256)
            mask_rfft2 = self.mean_response.mask_rfft2.repeat(1, 1, 256, 1)
            mask = torch.fft.irfft2(mask_rfft2, norm='ortho')
            mask = mask.permute(0, 1, 3, 2)
            noise_kernel = 20000 * mask
            self.noise_covariance = StationaryNoiseCovariance(noise_kernel, kernel=True)
            print('Debug: self.device=', self.device)
            self.weights_path = os.path.join(WEIGHTS_DIR, f'{self.diffusion_model}_MRI.pth')
            self.figure_path = os.path.join(FIGURES_DIR, f'{self.diffusion_model}_MRI.png')
            self.animation_path = os.path.join(ANIMATIONS_DIR, f'{self.diffusion_model}_MRI.mp4')
            self.reverse_diffusion_path = os.path.join(FIGURES_DIR, f'reverse_diffusion_process_{self.diffusion_model}_MRI.png')
            self.vmin, self.vmax = 0, 400
            self.vmin_null, self.vmax_null = -200, 200
            self.dataset_max_train = 200 * 37
            self.null_space_variance = 500000.0

        elif self.imaging_modality == 'PET':
            self.dataset = ADNIPETSlice_Dataset(os.path.join(DATA_DIR, 'ADNI/ADNI/'), verbose=True)
            self.dataset_max_train = 10000
            self.mean_response = GaussianFilterMeanSystemResponse(3.0)
            self.noise_covariance = MeanResponseAppliedToWhiteNoiseCovariance(variance=1e-1, mean_system_response=self.mean_response)
            self.null_space_variance = 0.04
            self.weights_path = os.path.join(WEIGHTS_DIR, f'{self.diffusion_model}_PET.pth')
            self.figure_path = os.path.join(FIGURES_DIR, f'{self.diffusion_model}_PET.png')
            self.animation_path = os.path.join(ANIMATIONS_DIR, f'{self.diffusion_model}_PET.mp4')
            self.reverse_diffusion_path = os.path.join(FIGURES_DIR, f'reverse_diffusion_process_{self.diffusion_model}_PET.png')
            self.vmin, self.vmax = 0.0, 1.0
            self.vmin_null, self.vmax_null = -0.05, 0.05

        elif self.imaging_modality == 'PHOTO':
            self.dataset = PHOTO_Dataset(os.path.join(DATA_DIR, 'img_align_celeba/'), verbose=True)
            self.dataset_max_train = 10000
            self.mean_response = DownSampleMeanSystemResponse(4)
            self.noise_covariance = ScalarNoiseCovariance(0.01)
            self.null_space_variance = 3.0
            self.weights_path = os.path.join(WEIGHTS_DIR, f'{self.diffusion_model}_PHOTO.pth')
            self.figure_path = os.path.join(FIGURES_DIR, f'{self.diffusion_model}_PHOTO.png')
            self.animation_path = os.path.join(ANIMATIONS_DIR, f'{self.diffusion_model}_PHOTO.mp4')
            self.reverse_diffusion_path = os.path.join(FIGURES_DIR, f'reverse_diffusion_process_{self.diffusion_model}_PHOTO.png')
            self.vmin, self.vmax = 0, 1
            self.vmin_null, self.vmax_null = .45, .55

        elif self.imaging_modality == 'CT':
            self.dataset = SynthRad2023Task1CTSlice_Dataset(os.path.join(DATA_DIR, 'SynthRad2023/Task1/brain/'), verbose=True)
            self.dataset_max_train = 200 * 37
            U = torch.tensor(np.load(os.path.join(WEIGHTS_DIR, 'U.npy')), dtype=torch.float32)
            S = torch.tensor(np.load(os.path.join(WEIGHTS_DIR, 'S.npy')), dtype=torch.float32)
            V = torch.tensor(np.load(os.path.join(WEIGHTS_DIR, 'V.npy')), dtype=torch.float32)
            xShape = [1, 256, 256]
            yShape = [72, 375]
            self.mean_response = DenseSVDMeanSystemResponse(U, S, V, xShape, yShape)
            self.noise_covariance = ScalarNoiseCovariance(50000)
            self.null_space_variance = 100000.0
            self.weights_path = os.path.join(WEIGHTS_DIR, f'{self.diffusion_model}_CT.pth')
            self.figure_path = os.path.join(FIGURES_DIR, f'{self.diffusion_model}_CT.png')
            self.animation_path = os.path.join(ANIMATIONS_DIR, f'{self.diffusion_model}_CT.mp4')
            self.reverse_diffusion_path = os.path.join(FIGURES_DIR, f'reverse_diffusion_process_{self.diffusion_model}_CT.png')
            self.vmin, self.vmax = -30, 90
            self.vmin_null, self.vmax_null = -30, 90

        elif self.imaging_modality == 'SEM':
            self.dataset = SEM_Dataset(os.path.join(DATA_DIR, 'MICRONS'), verbose=True)
            self.dataset_max_train = 1000 * 20
            self.mean_response = IdentityMeanSystemResponse()
            self.noise_covariance = ScalarNoiseCovariance(500)
            self.null_space_variance = 10000.0
            self.weights_path = os.path.join(WEIGHTS_DIR, f'{self.diffusion_model}_SEM.pth')
            self.figure_path = os.path.join(FIGURES_DIR, f'{self.diffusion_model}_SEM.png')
            self.animation_path = os.path.join(ANIMATIONS_DIR, f'{self.diffusion_model}_SEM.mp4')
            self.reverse_diffusion_path = os.path.join(FIGURES_DIR, f'reverse_diffusion_process_{self.diffusion_model}_SEM.png')
            self.vmin, self.vmax = 110, 150
            self.vmin_null, self.vmax_null = -1, 1

        else:
            raise ValueError("Invalid imaging modality. Choose from 'MRI', 'PET', 'PHOTO', 'CT', 'SEM'.")

        self.measurement_likelihood = LinearSystemPlusGaussianNoise(self.mean_response, self.noise_covariance).to(self.device)

    def setup_diffusion_model(self):
        if self.diffusion_model not in ['PDB', 'NDM']:
            raise ValueError("Invalid diffusion model. Choose 'PDB' or 'NDM'.")

        if self.imaging_modality == 'PHOTO':
            self.denoiser_network = DiffusersUnet(input_channels=3, unet_out_channels=3)
        else:
            self.denoiser_network = DiffusersUnet()

        if self.diffusion_model == 'PDB':
            self.diffusion_model_instance = PseudoInverseDiffusionModel(self.measurement_likelihood, self.denoiser_network, null_space_variance=self.null_space_variance)
        elif self.diffusion_model == 'NDM':
            self.diffusion_model_instance = NullspaceDiffusionModel(self.measurement_likelihood, self.denoiser_network, null_space_variance=self.null_space_variance)

        self.diffusion_model_instance.to(self.device)

        if self.load:
            self.diffusion_model_instance.denoiser_network.load_state_dict(torch.load(self.weights_path))

    def save_to_yaml(self, path):
        with open(path, 'w') as file:
            yaml.dump(self.__dict__, file)

    @classmethod
    def load_from_yaml(cls, path):
        with open(path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return cls(**config_dict)