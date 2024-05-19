# train.py

import torch
import numpy as np
import shutil

def train_model(config):
    
    device = config.device
    diffusion_model = config.diffusion_model_instance
    dataset = config.dataset
    dataset_max_train = config.dataset_max_train
    batch_size = config.batch_size
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate
    warmup_steps = config.warmup_steps
    batches_per_epoch = config.batches_per_epoch
    weights_path = config.weights_path

    # Select a random starting index for the training batch
    ind = np.random.randint(0, dataset_max_train - batch_size * batches_per_epoch)
    data_tensor = dataset[ind:ind + batch_size * batches_per_epoch].to(device)
    tensor_dataset = torch.utils.data.TensorDataset(data_tensor)

    if config.imaging_modality == 'CT':
        def loss_fn(z_0, z_0_hat, t):
            residual = z_0 - z_0_hat
            residual = residual/torch.sqrt(t.unsqueeze(-1).unsqueeze(-1))
            idx = torch.logical_and(z_0 > -0.05, z_0 < 0.05)
            residual[idx] = 10*residual[idx]
            return torch.mean(residual**2)
    else:
        def loss_fn(z_0, z_0_hat, t):
            residual = z_0 - z_0_hat
            residual = residual/torch.sqrt(t.unsqueeze(-1).unsqueeze(-1))
            return torch.mean(residual**2)

    # Train the model
    diffusion_model.train(
        tensor_dataset, 
        batch_size=batch_size, 
        num_epochs=num_epochs, 
        learning_rate=learning_rate, 
        warmup_steps=warmup_steps, 
        batches_per_epoch=batches_per_epoch,
        loss_fn=loss_fn
    )

    # Save a backup of the weights
    torch.save(diffusion_model.denoiser_network.state_dict(), weights_path)
    # Copy the weights to a backup file to avoid issues when killing the training
    shutil.copyfile(weights_path, weights_path.replace('.pth', f'_backup.pth'))
