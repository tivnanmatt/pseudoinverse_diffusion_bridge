import torch
import numpy as np

from .measurement_models import LinearSystemPlusGaussianNoise
from .measurement_models import MeanSystemResponse

device_name = 'cuda:0' 
device = torch.device(device_name)

class NullspaceMeanSystemResponse(MeanSystemResponse):
    def __init__(self, measurement_likelihood):
        super(NullspaceMeanSystemResponse, self).__init__()
        self.measurement_likelihood = measurement_likelihood
    def A(self, x):
        x = self.measurement_likelihood.mean_system_response.A(x)
        x = self.measurement_likelihood.mean_system_response.pinvA(x)
        return x
    def AT(self, y):
        return self.A(y)
    def pinvATA(self, x):
        return self.A(x)

class NullspaceCovariance(torch.nn.Module):
    def __init__(self, nullspace_noise_variance, measurement_likelihood):
        super(NullspaceCovariance, self).__init__()
        if ~isinstance(nullspace_noise_variance, torch.Tensor):
            nullspace_noise_variance = torch.tensor(nullspace_noise_variance, dtype=torch.float32)
        self.nullspace_noise_variance = nullspace_noise_variance
        self.measurement_likelihood = measurement_likelihood
    def _project_to_range_space(self, x):
        x = self.measurement_likelihood.mean_system_response.A(x)
        x = self.measurement_likelihood.mean_system_response.pinvA(x)
        return x
    def Sigma(self, x):
        return self.nullspace_noise_variance*(x-self._project_to_range_space(x))
    def sqrtSigma(self, x):
        return torch.sqrt(self.nullspace_noise_variance)*(x-self._project_to_range_space(x))
    def invSigma(self, x):
        return x
    def forward(self, x):
        return self.Sigma(x)
    

class PseudoInverseDiffusionModel(torch.nn.Module):
    def __init__(self,  
                measurement_likelihood,
                denoiser_network,
                null_space_noise_covariance=None,
                null_space_variance=None):

        super(PseudoInverseDiffusionModel, self).__init__()
        assert isinstance(measurement_likelihood, LinearSystemPlusGaussianNoise)

        if null_space_noise_covariance is None:
            if null_space_variance is None:
                raise ValueError('Either null_space_noise_covariance or null_space_variance must be provided')
            null_space_noise_covariance = NullspaceCovariance(null_space_variance, measurement_likelihood)

        self.measurement_likelihood = measurement_likelihood    
        self.null_space_noise_covariance = null_space_noise_covariance
        self.denoiser_network = denoiser_network

    def sample_z_t_given_x_0(self, x_0, t, y_shape=None):
        if y_shape is None:
            y_shape = self.measurement_likelihood.mean_system_response.A(x_0).shape
        while len(t.shape) < len(x_0.shape):
            t = t.unsqueeze(-1)
        white_noise = torch.randn(y_shape, dtype=x_0.dtype, device=x_0.device)
        return self.sample_z_t_given_x_0_and_white_noise(x_0, t, white_noise)
    
    def sample_z_t_given_x_0_and_white_noise(self, x_0, t, white_noise):
        while len(t.shape) < len(x_0.shape):
            t = t.unsqueeze(-1)
        range_space_noise = self.measurement_likelihood.noise_covariance.sqrtSigma(white_noise)
        range_space_noise = self.measurement_likelihood.mean_system_response.pinvA(range_space_noise)
        range_space_noise = torch.sqrt(t)*range_space_noise
        null_space_noise = torch.randn(x_0.shape, dtype=x_0.dtype, device=x_0.device)
        null_space_noise = self.null_space_noise_covariance.sqrtSigma(null_space_noise)
        null_space_noise = torch.sqrt(t)*null_space_noise
        return x_0 + range_space_noise + null_space_noise

    def sample_z_t_plus_dt_given_z_t(self, z_t, dt, y_shape=None):
        if y_shape is None:
            y_shape = self.measurement_likelihood.mean_system_response.A(z_t).shape
        while len(t.shape) < len(z_t.shape):
            t = t.unsqueeze(-1)
        if type(dt) is torch.Tensor:
            dt = float(dt.detach().cpu().numpy())
        white_noise = torch.randn(y_shape, dtype=z_t.dtype, device=z_t.device)
        return self.sample_z_t_plus_dt_given_z_t_and_white_noise(z_t, dt, white_noise)
    
    def sample_z_t_plus_dt_given_z_t_and_white_noise(self, z_t, dt, white_noise):
        while len(t.shape) < len(z_t.shape):
            t = t.unsqueeze(-1)
        if type(dt) is torch.Tensor:
            dt = float(dt.detach().cpu().numpy())
        range_space_noise = self.measurement_likelihood.noise_covariance.sqrtSigma(white_noise)
        range_space_noise = self.measurement_likelihood.mean_system_response.pinvA(range_space_noise)
        range_space_noise = np.sqrt(dt)*range_space_noise
        null_space_noise = torch.randn(z_t.shape, dtype=z_t.dtype, device=z_t.device)
        null_space_noise = self.null_space_noise_covariance.sqrtSigma(null_space_noise)
        null_space_noise = np.sqrt(dt)*null_space_noise
        return z_t + range_space_noise + null_space_noise
    
    def sample_z_t_minus_dt_given_z_t(self, z_t, t, dt, y_shape=None):
        if y_shape is None:
            y_shape = self.measurement_likelihood.mean_system_response.A(z_t).shape
        while len(t.shape) < len(z_t.shape):
            t = t.unsqueeze(-1)
        if type(dt) is torch.Tensor:
            dt = float(dt.detach().cpu().numpy())
        white_noise = torch.randn(y_shape, dtype=z_t.dtype, device=z_t.device)
        return self.sample_z_t_minus_dt_given_z_t_and_white_noise(z_t, t, dt, white_noise)

    def sample_z_t_minus_dt_given_z_t_and_white_noise(self, z_t, t, dt, white_noise):
        while len(t.shape) < len(z_t.shape):
            t = t.unsqueeze(-1)
        if type(dt) is torch.Tensor:
            dt = float(dt.detach().cpu().numpy())
        mean_estimate = self.denoiser_network(z_t, t)
        drift = dt*(1/t)*(mean_estimate - z_t)
        range_space_noise = self.measurement_likelihood.noise_covariance.sqrtSigma(white_noise)
        range_space_noise = self.measurement_likelihood.mean_system_response.pinvA(range_space_noise)
        range_space_noise = np.sqrt(dt)*range_space_noise
        null_space_noise = torch.randn(z_t.shape, dtype=z_t.dtype, device=z_t.device)
        null_space_noise = self.null_space_noise_covariance.sqrtSigma(null_space_noise)
        null_space_noise = np.sqrt(dt)*null_space_noise
        return z_t + drift + range_space_noise + null_space_noise
    
    def train(self, dataset, batch_size, num_epochs, learning_rate, warmup_steps=0, batches_per_epoch=100, loss_fn=None, time_sampler=None):
            
            _y = self.measurement_likelihood.mean_system_response.A(dataset[0:1][0])
            y_shape = _y.shape

            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            optimizer = torch.optim.Adam(self.denoiser_network.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: min(1.0, (epoch+1)/(warmup_steps+1)))
            
            if time_sampler is None:
                def time_sampler():
                    return torch.rand((1,1), device=device)**4.0

            if loss_fn is None:
                def loss_fn(z_0, z_0_hat, t):
                    residual = z_0 - z_0_hat
                    residual = residual/torch.sqrt(t.unsqueeze(-1).unsqueeze(-1))
                    return torch.mean(residual**2)

            for epoch in range(num_epochs):
                for iBatch, [batch] in enumerate(data_loader):
                    if iBatch > batches_per_epoch:
                        break
                    optimizer.zero_grad()
                    x_0 = batch
                    t = torch.rand((x_0.shape[0],1), device=x_0.device)
                    z_t = self.sample_z_t_given_x_0(x_0, t, y_shape)
                    x_0_hat = self.denoiser_network(z_t, t)
                    loss = loss_fn(x_0, x_0_hat, t)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    print('Epoch: %d, Iteration: %d, Loss: %f' % (epoch, iBatch, loss.item()))


    def sample(self, y, time_steps=None, num_time_steps=None, return_all_timesteps=False, verbose=False):
        
        if time_steps is None:
            if num_time_steps is None:
                raise ValueError('Either timesteps or num_timesteps must be provided')
            time_steps = torch.linspace(1, 0, num_time_steps+1).to(y.device)

        assert type(time_steps) is torch.Tensor
        assert time_steps.shape[0] > 1, 'Time steps must have at least 2 elements'
        assert time_steps[0] == 1.0, 'Initial time must be 1.0'
        for i in range(1, time_steps.shape[0]):
            assert time_steps[i] < time_steps[i-1], 'Time steps must be decreasing'

        num_timesteps = time_steps.shape[0] - 1
        
        pinvA_y = self.measurement_likelihood.mean_system_response.pinvA(y)
        null_space_noise = self.null_space_noise_covariance.sqrtSigma(torch.randn(pinvA_y.shape, dtype=pinvA_y.dtype, device=pinvA_y.device))
        z_T = pinvA_y + null_space_noise
        
        z_t = z_T.clone()
        t = torch.ones((z_T.shape[0],1), dtype=z_T.dtype, device=z_T.device)
        if return_all_timesteps:
            z_t_all = z_t.clone().unsqueeze(0).repeat(num_timesteps+1,1,1,1,1)
            z_t_all[0] = z_t

        for i in range(num_timesteps):

            dt = time_steps[i] - time_steps[i+1]

            if verbose:
                print('Timestep: %d / %d, from time %f to time %f' % (i+1, num_timesteps, t[0,0].item(), time_steps[i+1].item()) )
            if i < num_timesteps-1:
                with torch.no_grad():
                    z_t = self.sample_z_t_minus_dt_given_z_t(z_t, t, dt)
            else:
                with torch.no_grad():
                    z_t = self.denoiser_network(z_t, t)
            t[:,0] = time_steps[i+1]
            if return_all_timesteps:
                z_t_all[i+1] = z_t
        
        if return_all_timesteps:
            return z_t_all
        else:
            return z_t
        
    def forward(self, y, num_timesteps=128, return_all_timesteps=False, verbose=False):
        return self.sample(y, num_timesteps, return_all_timesteps, verbose)
    




class NullspaceDiffusionModel(torch.nn.Module):
    def __init__(self,  
                measurement_likelihood,
                denoiser_network,
                null_space_noise_covariance=None,
                null_space_variance=None):

        super(NullspaceDiffusionModel, self).__init__()
        assert isinstance(measurement_likelihood, LinearSystemPlusGaussianNoise)

        if null_space_noise_covariance is None:
            if null_space_variance is None:
                raise ValueError('Either null_space_noise_covariance or null_space_variance must be provided')
            null_space_noise_covariance = NullspaceCovariance(null_space_variance, measurement_likelihood)

        self.measurement_likelihood = measurement_likelihood    
        self.null_space_noise_covariance = null_space_noise_covariance
        self.denoiser_network = denoiser_network

    def sample_z_t_given_x_0(self, x_0, t, y_shape=None):
        if y_shape is None:
            y_shape = self.measurement_likelihood.mean_system_response.A(x_0).shape
        while len(t.shape) < len(x_0.shape):
            t = t.unsqueeze(-1)
        white_noise = torch.randn(y_shape, dtype=x_0.dtype, device=x_0.device)
        return self.sample_z_t_given_x_0_and_white_noise(x_0, t, white_noise)
    
    def sample_z_t_given_x_0_and_white_noise(self, x_0, t, white_noise):
        while len(t.shape) < len(x_0.shape):
            t = t.unsqueeze(-1)
        range_space_noise = self.measurement_likelihood.noise_covariance.sqrtSigma(white_noise)
        range_space_noise = self.measurement_likelihood.mean_system_response.pinvA(range_space_noise)
        range_space_noise = range_space_noise
        null_space_noise = torch.randn(x_0.shape, dtype=x_0.dtype, device=x_0.device)
        null_space_noise = self.null_space_noise_covariance.sqrtSigma(null_space_noise)
        null_space_noise = torch.sqrt(t)*null_space_noise
        return x_0 + range_space_noise + null_space_noise
    
    def sample_z_t_plus_dt_given_z_t(self, z_t, dt, y_shape=None):
        if y_shape is None:
            y_shape = self.measurement_likelihood.mean_system_response.A(z_t).shape
        while len(t.shape) < len(z_t.shape):
            t = t.unsqueeze(-1)
        if type(dt) is torch.Tensor:
            dt = float(dt.detach().cpu().numpy())
        white_noise = torch.randn(y_shape, dtype=z_t.dtype, device=z_t.device)
        return self.sample_z_t_plus_dt_given_z_t_and_white_noise(z_t, dt, white_noise)
    
    def sample_z_t_plus_dt_given_z_t_and_white_noise(self, z_t, dt, white_noise):
        # measurement domain white noise is not really used
        # just a carry over from the pseudoinverse diffusion model
        while len(t.shape) < len(z_t.shape):
            t = t.unsqueeze(-1)
        if type(dt) is torch.Tensor:
            dt = float(dt.detach().cpu().numpy())
        null_space_noise = torch.randn(z_t.shape, dtype=z_t.dtype, device=z_t.device)
        null_space_noise = self.null_space_noise_covariance.sqrtSigma(null_space_noise)
        null_space_noise = np.sqrt(dt)*null_space_noise
        return z_t + null_space_noise
    
    def sample_z_t_minus_dt_given_z_t(self, z_t, t, dt, y_shape=None):
        if y_shape is None:
            y_shape = self.measurement_likelihood.mean_system_response.A(z_t).shape
        while len(t.shape) < len(z_t.shape):
            t = t.unsqueeze(-1)
        if type(dt) is torch.Tensor:
            dt = float(dt.detach().cpu().numpy())
        white_noise = torch.randn(y_shape, dtype=z_t.dtype, device=z_t.device)
        return self.sample_z_t_minus_dt_given_z_t_and_white_noise(z_t, t, dt, white_noise)
    
    def sample_z_t_minus_dt_given_z_t_and_white_noise(self, z_t, t, dt, white_noise):
        while len(t.shape) < len(z_t.shape):
            t = t.unsqueeze(-1)
        if type(dt) is torch.Tensor:
            dt = float(dt.detach().cpu().numpy())
        mean_estimate = self.denoiser_network(z_t, t)
        drift = dt*(1/t)*(mean_estimate - z_t)
        drift_range_space = self.measurement_likelihood.mean_system_response.pinvA(self.measurement_likelihood.mean_system_response.A(drift))
        drift_null_space = drift - drift_range_space
        null_space_noise = torch.randn(z_t.shape, dtype=z_t.dtype, device=z_t.device)
        null_space_noise = self.null_space_noise_covariance.sqrtSigma(null_space_noise)
        null_space_noise = np.sqrt(dt)*null_space_noise
        return z_t + drift_null_space + null_space_noise
    
    def train(self, dataset, batch_size, num_epochs, learning_rate, warmup_steps=0, batches_per_epoch=100, loss_fn=None):
        
        _y = self.measurement_likelihood.mean_system_response.A(dataset[0:1][0])
        y_shape = _y.shape

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.denoiser_network.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: min(1.0, (epoch+1)/(warmup_steps+1)))
        
        if loss_fn is None:
            def loss_fn(z_0, z_0_hat, t):
                residual = z_0 - z_0_hat
                residual = residual/torch.sqrt(t.unsqueeze(-1).unsqueeze(-1))
                return torch.mean(residual**2)

        for epoch in range(num_epochs):
            for iBatch, [batch] in enumerate(data_loader):
                if iBatch > batches_per_epoch:
                    break
                optimizer.zero_grad()
                x_0 = batch
                t = torch.rand((x_0.shape[0],1), device=x_0.device)
                z_t = self.sample_z_t_given_x_0(x_0, t, y_shape)
                x_0_hat = self.denoiser_network(z_t, t)
                loss = loss_fn(x_0, x_0_hat, t)
                loss.backward()
                optimizer.step()
                scheduler.step()
                print('Epoch: %d, Iteration: %d, Loss: %f' % (epoch, iBatch, loss.item()))

    def sample(self, y, time_steps=None, num_time_steps=None, return_all_timesteps=False, verbose=False):

        if time_steps is None:
            if num_time_steps is None:
                raise ValueError('Either timesteps or num_timesteps must be provided')
            time_steps = torch.linspace(1, 0, num_time_steps+1).to(y.device)

        assert type(time_steps) is torch.Tensor
        assert time_steps.shape[0] > 1, 'Time steps must have at least 2 elements'
        assert time_steps[0] == 1.0, 'Initial time must be 1.0'
        for i in range(1, time_steps.shape[0]):
            assert time_steps[i] < time_steps[i-1], 'Time steps must be decreasing'

        num_timesteps = time_steps.shape[0] - 1
        
        pinvA_y = self.measurement_likelihood.mean_system_response.pinvA(y)
        null_space_noise = self.null_space_noise_covariance.sqrtSigma(torch.randn(pinvA_y.shape, dtype=pinvA_y.dtype, device=pinvA_y.device))
        z_T = pinvA_y + null_space_noise
        
        z_t = z_T.clone()
        t = torch.ones((z_T.shape[0],1), dtype=z_T.dtype, device=z_T.device)
        if return_all_timesteps:
            z_t_all = z_t.clone().unsqueeze(0).repeat(num_timesteps+1,1,1,1,1)
            z_t_all[0] = z_t

        for i in range(num_timesteps):

            dt = time_steps[i] - time_steps[i+1]

            if verbose:
                print('Timestep: %d / %d, from time %f to time %f' % (i+1, num_timesteps, t[0,0].item(), time_steps[i+1].item()) )
            if i < num_timesteps-1:
                with torch.no_grad():
                    z_t = self.sample_z_t_minus_dt_given_z_t(z_t, t, dt)
            else:
                with torch.no_grad():
                    z_t = self.denoiser_network(z_t, t)
            t[:,0] = time_steps[i+1]
            if return_all_timesteps:
                z_t_all[i+1] = z_t
        
        if return_all_timesteps:
            return z_t_all
        else:
            return z_t