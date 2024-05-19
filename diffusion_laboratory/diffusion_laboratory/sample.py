# sample.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def sample_model(config):
    device = config.device
    diffusion_model = config.diffusion_model_instance
    dataset = config.dataset
    num_reverse_steps = config.num_reverse_steps
    figure_path = config.figure_path
    animation_path = config.animation_path
    reverse_diffusion_path = config.reverse_diffusion_path
    vmin = config.vmin
    vmax = config.vmax
    vmin_null = config.vmin_null
    vmax_null = config.vmax_null
    imaging_modality = config.imaging_modality
    batches_per_epoch = config.batches_per_epoch
    batch_size = config.batch_size
    dataset_max_train = config.dataset_max_train

    # Select a random starting index for the sampling batch
    ind = np.random.randint(0, dataset_max_train - batch_size * batches_per_epoch)
    data_tensor = dataset[ind:ind + batch_size * batches_per_epoch].to(device)
    tensor_dataset = torch.utils.data.TensorDataset(data_tensor)

    X = tensor_dataset[0:1][0]
    Y = diffusion_model.measurement_likelihood.sample(X)
    Z = diffusion_model.measurement_likelihood.mean_system_response.pinvA(Y)
    time_steps = torch.linspace(1, 0, num_reverse_steps + 1).to(device)**(4.0)
    z_t = diffusion_model.sample(Y, time_steps=time_steps, verbose=True, return_all_timesteps=True)

    def projection_to_range_space(z_t):
        _tmp = diffusion_model.measurement_likelihood.mean_system_response.A(z_t)
        _range_space = diffusion_model.measurement_likelihood.mean_system_response.pinvA(_tmp)
        return _range_space

    def prep_for_imshow(img, modality):
        if modality == 'PHOTO':
            return img.cpu().numpy().transpose(1, 2, 0).clip(0, 1)
        elif modality == 'CT' or modality == 'MRI':
            return img.cpu().numpy()[0].T
        else:
            return img.cpu().numpy()[0]

    X_RangeSpace = projection_to_range_space(X)
    Z_RangeSpace = projection_to_range_space(Z)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(3, 3, 1)
    ax.set_title('True Image')
    im11 = ax.imshow(prep_for_imshow(X[0], imaging_modality), cmap='gray', vmin=vmin, vmax=vmax)
    fig.colorbar(im11, ax=ax)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(3, 3, 2)
    ax.set_title('Pseudoinverse Reconstruction')
    im12 = ax.imshow(prep_for_imshow(Z_RangeSpace[0], imaging_modality), cmap='gray', vmin=vmin, vmax=vmax)
    fig.colorbar(im12, ax=ax)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(3, 3, 3)
    ax.set_title(f'{config.diffusion_model} Sample')
    im13 = ax.imshow(prep_for_imshow(z_t[0, 0], imaging_modality), cmap='gray', vmin=vmin, vmax=vmax)
    fig.colorbar(im13, ax=ax)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(3, 3, 4)
    ax.set_title('True Image (Range Space)')
    im21 = ax.imshow(prep_for_imshow(X_RangeSpace[0], imaging_modality), cmap='gray', vmin=vmin, vmax=vmax)
    fig.colorbar(im21, ax=ax)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(3, 3, 5)
    ax.set_title('Pseudoinverse (Range Space)')
    im22 = ax.imshow(prep_for_imshow(Z_RangeSpace[0], imaging_modality), cmap='gray', vmin=vmin, vmax=vmax)
    fig.colorbar(im22, ax=ax)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(3, 3, 6)
    ax.set_title(f'{config.diffusion_model} Sample (Range Space)')
    im23 = ax.imshow(prep_for_imshow(z_t[0, 0], imaging_modality), cmap='gray', vmin=vmin, vmax=vmax)
    fig.colorbar(im23, ax=ax)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(3, 3, 7)
    ax.set_title('True Image (Null Space)')
    null_space_image = X[0] - X_RangeSpace[0]
    if imaging_modality == 'PHOTO':
        null_space_image += 0.5
    im31 = ax.imshow(prep_for_imshow(null_space_image, imaging_modality), cmap='gray', vmin=vmin_null, vmax=vmax_null)
    fig.colorbar(im31, ax=ax)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(3, 3, 8)
    ax.set_title('Pseudoinverse (Null Space)')
    null_space_image = Z[0] - Z_RangeSpace[0]
    if imaging_modality == 'PHOTO':
        null_space_image += 0.5
    im32 = ax.imshow(prep_for_imshow(null_space_image, imaging_modality), cmap='gray', vmin=vmin_null, vmax=vmax_null)
    fig.colorbar(im32, ax=ax)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(3, 3, 9)
    ax.set_title(f'{config.diffusion_model} Sample (Null Space)')
    null_space_image = z_t[0, 0] - projection_to_range_space(z_t[0])[0]
    if imaging_modality == 'PHOTO':
        null_space_image += 0.5
    im33 = ax.imshow(prep_for_imshow(null_space_image, imaging_modality), cmap='gray', vmin=vmin_null, vmax=vmax_null)
    fig.colorbar(im33, ax=ax)
    ax.set_xticks([]); ax.set_yticks([])

    plt.savefig(figure_path)

    def animate(i):
        print(f'Animating Frame: {i}')
        im13.set_data(prep_for_imshow(z_t[i, 0], imaging_modality))
        ax.set_title(f'{config.diffusion_model} Sample, Time: {i}')
        _range_space = projection_to_range_space(z_t[i])[0]
        im23.set_data(prep_for_imshow(_range_space, imaging_modality))
        ax.set_title(f'{config.diffusion_model} Sample (Range Space), Time: {i}')
        _null_space = z_t[i, 0] - _range_space
        if imaging_modality == 'PHOTO':
            _null_space += 0.5
        im33.set_data(prep_for_imshow(_null_space, imaging_modality))
        ax.set_title(f'{config.diffusion_model} Sample (Null Space), Time: {i}')
        return im33,

    ani = animation.FuncAnimation(fig, animate, frames=z_t.shape[0], interval=100)
    writer = animation.writers['ffmpeg'](fps=15)
    ani.save(animation_path, writer=writer)

    # Save linear array showing the reverse diffusion process
    n_steps = 6
    step_indices = np.linspace(0, z_t.shape[0] - 1, n_steps - 1, dtype=int)
    fig, axes = plt.subplots(3, n_steps, figsize=(12, 6), gridspec_kw={'wspace': 0.03, 'hspace': 0.03})
    for j, idx in enumerate(step_indices):
        axes[0, j].imshow(prep_for_imshow(z_t[idx, 0], imaging_modality), cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])

        range_space = projection_to_range_space(z_t[idx])[0]
        axes[1, j].imshow(prep_for_imshow(range_space, imaging_modality), cmap='gray', vmin=vmin, vmax=vmax)
        axes[1, j].set_xticks([]); axes[1, j].set_yticks([])

        null_space = z_t[idx, 0] - range_space
        if imaging_modality == 'PHOTO':
            null_space += 0.5
        axes[2, j].imshow(prep_for_imshow(null_space, imaging_modality), cmap='gray', vmin=vmin_null, vmax=vmax_null)
        axes[2, j].set_xticks([]); axes[2, j].set_yticks([])

    axes[0, -1].imshow(prep_for_imshow(X[0], imaging_modality), cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, -1].set_xticks([]); axes[0, -1].set_yticks([])

    axes[1, -1].imshow(prep_for_imshow(X_RangeSpace[0], imaging_modality), cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, -1].set_xticks([]); axes[1, -1].set_yticks([])

    null_space_image = X[0] - X_RangeSpace[0]
    if imaging_modality == 'PHOTO':
        null_space_image += 0.5
    axes[2, -1].imshow(prep_for_imshow(null_space_image, imaging_modality), cmap='gray', vmin=vmin_null, vmax=vmax_null)
    axes[2, -1].set_xticks([]); axes[2, -1].set_yticks([])

    plt.savefig(reverse_diffusion_path, dpi=300)

    print('Done')