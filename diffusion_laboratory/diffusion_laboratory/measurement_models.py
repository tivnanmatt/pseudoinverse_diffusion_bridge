import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# this is one piece of a framework called pseudo inverse diffusion models
# the user is a domain expert in one area of medical imaging
# N is the multivariate normal distribution
# they should define the forward model using the formate: N(Ax, Sigma)
# where x is the ground truth image vector
# y is the noisy measurements vector
# A is the mean system response matrix
# Sigma is the measurement noise covariance matrix


# for pseudo inverse diffusion models we also need B = (A^T A)^-1 A^T
# so we need (A^T A)^-1 and A^T also 

# to sample from the noisy measurements we need to apply sqrt(Sigma) to a sample from N(0, I)
# to evaluate likelihoods, we need to apply inv(Sigma) to the residual


# lets start with additive gaussian white noise
# A is the identity matrix
# Sigma is a scalar times the identity matrix
# A^T is the identity matrix
# (A^T A)^-1 is the identity matrix
# B is the identity matrix

# its a torch.Module and the forward is just the sample method
# it both sample and log_prob methods should handle batched inputs and outputs

class LinearSystemPlusGaussianNoise(torch.nn.Module):
    def __init__(self, mean_system_response, noise_covariance):
        super(LinearSystemPlusGaussianNoise, self).__init__()
        self.mean_system_response = mean_system_response
        self.noise_covariance = noise_covariance
    def sample(self, x):
        Ax = self.mean_system_response.A(x)
        white_noise = torch.randn_like(Ax)
        correlated_noise = self.noise_covariance.sqrtSigma(white_noise)
        return Ax + correlated_noise
    def log_prob(self, x, y):
        Ax = self.mean_system_response.A(x)
        return -0.5 * torch.sum((y - Ax) * self.noise_covariance.invSigma(y - Ax), 
                                dim=[dim for dim in range(1, len(y.shape))])
    def forward(self, x):
        return self.sample(x)
    
class MeanSystemResponse(torch.nn.Module):
    def __init__(self):
        super(MeanSystemResponse, self).__init__()
    def A(self, x):
        raise NotImplementedError
    def AT(self, y):
        raise NotImplementedError
    def pinvATA(self, x):
        raise NotImplementedError
    def pinvA(self, y):
        return self.pinvATA(self.AT(y))
    def forward(self, x):
        return self.A(x)

class NoiseCovariance(torch.nn.Module):
    def __init__(self):
        super(NoiseCovariance, self).__init__()
    def sqrtSigma(self, y):
        raise NotImplementedError
    def invSigma(self, y):
        raise NotImplementedError
    def forward(self, y):
        return self.Sigma(y)
    
class IdentityMeanSystemResponse(MeanSystemResponse):
    def __init__(self):
        super(IdentityMeanSystemResponse, self).__init__()
    def A(self, x):
        return x
    def AT(self, y):
        return y
    def pinvATA(self, x):
        return x
# lets take a different approach and redefine it based on the save average pooling operators are used in CNNs
    
class DownSampleMeanSystemResponse(MeanSystemResponse):
    def __init__(self, kernel_size):
        super(DownSampleMeanSystemResponse, self).__init__()
        # only support kernel_size = 1, 2, 4, 8
        assert kernel_size in [1, 2, 4, 8]
        self.kernel_size = kernel_size
        
    def A(self, x):
        kernel_weights = torch.ones(1, 1, self.kernel_size, self.kernel_size, device=x.device, dtype=torch.float32)/(self.kernel_size**2)
        y_list = []
        for i in range(x.shape[1]):
            y_list.append(torch.nn.functional.conv2d(x[:, i:i+1], kernel_weights, bias=None, stride=self.kernel_size, padding=0, groups=1))
        y = torch.cat(y_list, dim=1)
        return y

    def AT(self, y):
        # the transpose of average pooling is just repeating the values
        kernel_weights = torch.ones(1, 1, self.kernel_size, self.kernel_size, device=y.device, dtype=torch.float32)/(self.kernel_size**2)
        z_list = []
        for i in range(y.shape[1]):
            z_list.append(torch.nn.functional.conv_transpose2d(y[:, i:i+1], kernel_weights, bias=None, stride=self.kernel_size, padding=0, groups=1))
        z = torch.cat(z_list, dim=1)
        return z
    
    def pinvATA(self,z):
        return self.kernel_size**2 * z
    

class RFFT2FilterMeanSystemResponse(MeanSystemResponse):
    def __init__(self, filter):
        super(RFFT2FilterMeanSystemResponse, self).__init__()
        self.filter = filter
        return
    
    def A(self, x):
        x_rfft2 = torch.fft.rfft2(x, norm='ortho')
        y_rfft2 = x_rfft2 * self.filter.to(x.device)
        y = torch.fft.irfft2(y_rfft2, norm='ortho')
        return y
    
    def AT(self, y):
        y_rfft2 = torch.fft.rfft2(y, norm='ortho')
        z_rfft2 = y_rfft2 * torch.conj(self.filter.to(y.device))
        z = torch.fft.irfft2(z_rfft2, norm='ortho')
        return z

    def pinvATA(self, z):
        z_rfft2 = torch.fft.rfft2(z, norm='ortho')
        x_rfft2 = z_rfft2 / (torch.abs(self.filter.to(z.device))**2.0)
        x = torch.fft.irfft2(x_rfft2, norm='ortho')
        return x


class GaussianFilterMeanSystemResponse(MeanSystemResponse):
    def __init__(self, filter_sigma):
        super(GaussianFilterMeanSystemResponse, self).__init__()
        self.filter_sigma = filter_sigma
        # N = 2*(int(filter_sigma*3)//2) + 1
        self.filter_rfft2 = self._create_filter([3,3])
        
    # def _ft(self,x):
    #     # first zero pad with x.shape in -2 and -1 dimensions
    #     # x = torch.nn.functional.pad(x, (x.shape[-1], x.shape[-1], x.shape[-2], x.shape[-2]))
    #     x_flipLR = torch.flip(x, [-1])
    #     x = torch.cat((x_flipLR, x, x_flipLR), dim=-1)
    #     x_flipUD = torch.flip(x, [-2])
    #     x = torch.cat((x_flipUD, x, x_flipUD), dim=-2)
    #     x_rfft2 = torch.fft.rfft2(x, norm='ortho')
    #     return x_rfft2
    
    # def _ift(self,x):
    #     x = torch.fft.irfft2(x, norm='ortho')
    #     return x[..., x.shape[-2]//3:2*x.shape[-2]//3, x.shape[-1]//3:2*x.shape[-1]//3]
    
    def _ft(self,x):
        x_rfft2 = torch.fft.rfft2(x, norm='ortho')
        return x_rfft2

    def _ift(self,x):
        x = torch.fft.irfft2(x, norm='ortho')
        return x

    def _create_filter(self, xShape):
        x = torch.arange(0, xShape[-2], dtype=torch.float32)
        y = torch.arange(0, xShape[-1], dtype=torch.float32)
        xGrid,yGrid = torch.meshgrid(x,y)
        cx = int(xShape[-2]/2)
        cy = int(xShape[-1]/2)
        r2 = (xGrid-cx)**2 + (yGrid-cy)**2
        h = torch.exp(-r2/(2*self.filter_sigma**2))
        h = h / h.sum()
        impulse = torch.zeros(xShape[-2], xShape[-1], dtype=torch.float32)
        impulse[cx,cy] = 1
        h_rfft2 = self._ft(h)
        impulse_rfft2 = self._ft(impulse)
        filter_rfft2 = h_rfft2 / impulse_rfft2
        # unsqueeze 0 until it has the same number of dimensions as xShape
        while filter_rfft2.dim() < len(xShape):
            filter_rfft2 = filter_rfft2.unsqueeze(0)
        null_space_idx = filter_rfft2.abs() < 0.1*filter_rfft2.abs().max()
        filter_rfft2[null_space_idx] = 0
        self.A_filter_rfft2 = filter_rfft2
        self.AT_filter_rfft2 = torch.conj(filter_rfft2)
        self.pinvATA_filter_rfft2 = 0*filter_rfft2
        self.pinvATA_filter_rfft2[~null_space_idx] = (1 / filter_rfft2[~null_space_idx].abs()**2).to(torch.complex64)
        return 
    
    def A(self, x):
        x_rfft2 = self._ft(x)
        if (x_rfft2.shape[-2:] != self.A_filter_rfft2.shape[-2:] or x_rfft2.shape[-4:-2] != self.A_filter_rfft2.shape[-4:-2]):
            self._create_filter(x.shape)
        y_rfft2 = x_rfft2 * self.A_filter_rfft2.to(x.device)
        y = self._ift(y_rfft2)
        return y
    
    def AT(self, y):
        y_rfft2 = self._ft(y)
        if (y_rfft2.shape[-2:] != self.A_filter_rfft2.shape[-2:] or y_rfft2.shape[-4:-2] != self.A_filter_rfft2.shape[-4:-2]):
            self.filter_rfft2 = self._create_filter(y.shape)
        z_rfft2 = y_rfft2 * self.AT_filter_rfft2.to(y.device)
        z = self._ift(z_rfft2)
        return z
    
    def pinvATA(self, z):
        z_rfft2 = self._ft(z)
        if (z_rfft2.shape[-2:] != self.A_filter_rfft2.shape[-2:] or z_rfft2.shape[-4:-2] != self.A_filter_rfft2.shape[-4:-2]):
            self.filter_rfft2 = self._create_filter(z.shape)
        x = z_rfft2 * self.pinvATA_filter_rfft2.to(z.device)
        x = self._ift(x)
        return x
    



    


class UndersampledMRIMeanSystemResponse(MeanSystemResponse):
    def __init__(self, image_size, mask_seed=42):
        super(UndersampledMRIMeanSystemResponse, self).__init__()
        self.image_size = image_size
        
        # Generate kx values for RFFT2, which outputs the non-negative frequency components
        # For a 128x128 image, the size in RFFT2 space along the width will be 65
        kx_length = image_size // 2 + 1  # General formula for RFFT output size
        kx = np.linspace(0, 0.5, kx_length)  # Linear space from 0 to 1 for kx
        kx = torch.tensor(kx, dtype=torch.float32)
        
        # Set a random seed for reproducibility
        torch.manual_seed(mask_seed)

        # Generate a random vector for comparison to determine active frequencies
        rand_vector = torch.rand(kx_length)
        
        # Identify active regions by comparing scaled kx^2 to the random vector
        active_regions = (2*kx)**0.5 < rand_vector
        
        # Create the mask for the RFFT2 space
        # This mask is 1D along kx, to be broadcasted across the ky dimension during application
        self.mask_rfft2 = active_regions.float().unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions for broadcasting
        
    def A(self, x):
        # because of the default behavior of torch.fft.rfft2, we need to permute the dimensions
        x = x.permute(0, 1, 3, 2)
        # Apply the mask in the frequency domain, and make sure its unitary
        x_rfft2 = torch.fft.rfft2(x, norm='ortho')
        y_rfft2 = x_rfft2 * self.mask_rfft2.to(x.device)
        y = torch.fft.irfft2(y_rfft2, norm='ortho')
        # permute the dimensions back
        y = y.permute(0, 1, 3, 2)
        return y
    
    def AT(self, y):
        return self.A(y)
    
    def pinvATA(self, z):
        return self.A(z)
    






class DenseSVDMeanSystemResponse(MeanSystemResponse):
    def __init__(self, U, S, V, xShape, yShape):
        super(DenseSVDMeanSystemResponse, self).__init__()
        # U, S, V are expected to be tensors here
        self.xShape = xShape  # Expected shape of input x
        self.yShape = yShape  # Expected shape of output y
        # set these as parameters of the model so when I run model.to(device) they are moved to the device
        self.U = torch.nn.Parameter(U)
        self.S = torch.nn.Parameter(S)
        self.V = torch.nn.Parameter(V)
        self.xShape = xShape
        self.yShape = yShape

        # but make sure they are not trainable
        self.U.requires_grad = False
        self.S.requires_grad = False
        self.V.requires_grad = False

    def A(self, x):
        # Flatten x and add a batch dimension if it's not there: [nBatch, nCol]
        x_flat = x.view(-1, np.prod(self.xShape), 1)
        # Apply SVD components for the operation A = U S V^T
        # Adjust U and V for batch operations, S needs to be handled for element-wise multiplication
        # _tmp = torch.bmm(self.V.t().unsqueeze(0).repeat(x_flat.shape[0], 1, 1), x_flat)
        _tmp = torch.matmul(self.V.t().unsqueeze(0), x_flat)
        _tmp = _tmp * self.S.unsqueeze(0).unsqueeze(-1)
        y_flat = torch.matmul(self.U.unsqueeze(0), _tmp)
        # Reshape back to original y shape with batch dimension
        return y_flat.view(-1, *self.yShape)

    def AT(self, y):
        # Flatten y and add a batch dimension if it's not there: [nBatch, nRow]
        y_flat = y.view(-1, np.prod(self.yShape))
        # Apply SVD components for the operation A^T = V S U^T
        # Adjust U and V for batch operations, S needs to be handled for element-wise multiplication
        # _tmp = torch.bmm(self.U.t().unsqueeze(0).repeat(y_flat.shape[0], 1, 1), y_flat.unsqueeze(-1))
        _tmp = torch.matmul(self.U.t().unsqueeze(0), y_flat.unsqueeze(-1))
        _tmp = _tmp * self.S.unsqueeze(0).unsqueeze(-1)
        # x_flat = torch.bmm(self.V.unsqueeze(0).repeat(y_flat.shape[0], 1, 1), _tmp)
        x_flat = torch.matmul(self.V.unsqueeze(0), _tmp)
        # Reshape back to original x shape with batch dimension
        return x_flat.view(-1, *self.xShape)
    
    def pinvATA(self, x):
        # Apply pseudo-inverse of ATA using SVD components, which is V (S^-2) V^T for diagonal S
        # Flatten x and add a batch dimension if it's not there: [nBatch, nCol]
        x_flat = x.view(-1, np.prod(self.xShape))
        # Apply SVD components for the operation A = U S V^T
        # Adjust U and V for batch operations, S needs to be handled for element-wise multiplication
        # _tmp = torch.bmm(self.V.t().unsqueeze(0).repeat(x_flat.shape[0], 1, 1), x_flat.unsqueeze(-1))
        _tmp = torch.matmul(self.V.t().unsqueeze(0), x_flat.unsqueeze(-1))
        scale = (1 / self.S**2)
        scale[scale > scale.min()*100] = 0  # Zero out singular values below a threshold
        _tmp = _tmp * scale.unsqueeze(0).unsqueeze(-1)
        # z_flat = torch.bmm(self.V.unsqueeze(0).repeat(x_flat.shape[0], 1, 1), _tmp)
        z_flat = torch.matmul(self.V.unsqueeze(0), _tmp)
        # Reshape back to original z shape with batch dimension
        return z_flat.view(-1, *self.xShape)

class ScalarNoiseCovariance(NoiseCovariance):
    def __init__(self, scalarVariance):
        super(ScalarNoiseCovariance, self).__init__()
        # handle if scalar is a float or int, make it a tensor with dtype=torch.float32
        scalarVariance = torch.tensor(scalarVariance, dtype=torch.float32)
        self.scalarVariance = scalarVariance
    def Sigma(self, x):
        return self.scalarVariance * x
    def sqrtSigma(self, x):
        return torch.sqrt(self.scalarVariance) * x
    def invSigma(self, x):
        return x / self.scalarVariance
 
class DiagonalNoiseCovariance(NoiseCovariance):
    def __init__(self, diagonalVariance):
        super(DiagonalNoiseCovariance, self).__init__()
        self.diagonalVariance = diagonalVariance
    def Sigma(self, x):
        return x * self.diagonalVariance
    def sqrtSigma(self, x):
        return x * torch.sqrt(self.diagonalVariance)
    def invSigma(self, x):
        return x / self.diagonalVariance
    def forward(self, x):
        return self.Sigma(x)
    

class StationaryNoiseCovariance(NoiseCovariance):
    def __init__(self, variance_rfft2, kernel=False):
        super(StationaryNoiseCovariance, self).__init__()
        if kernel:
            variance_rfft2 = torch.fft.rfft2(variance_rfft2, norm='ortho')
        self.variance_rfft2 = variance_rfft2
    def Sigma(self, x):
        x_rfft2 = torch.fft.rfft2(x, norm='ortho')
        y_rfft2= x_rfft2 * self.variance_rfft2
        return torch.fft.irfft2(y_rfft2, norm='ortho').to(x.device)
    def sqrtSigma(self, x):
        x_rfft2 = torch.fft.rfft2(x, norm='ortho')
        y_rfft2 = x_rfft2* torch.sqrt(self.variance_rfft2).to(x.device)
        return torch.fft.irfft2(y_rfft2, norm='ortho')
    def invSigma(self, x):
        x_rfft2 = torch.fft.rfft2(x, norm='ortho')
        y_rfft2= x_rfft2 / self.variance_rfft2.to(x.device)
        return torch.fft.irfft2(y_rfft2, norm='ortho')
    

class MeanResponseAppliedToWhiteNoiseCovariance(NoiseCovariance):
    def __init__(self, variance, mean_system_response):
        super(MeanResponseAppliedToWhiteNoiseCovariance, self).__init__()
        self.variance = torch.tensor(variance, dtype=torch.float32)
        self.mean_system_response = mean_system_response
    def Sigma(self, x):
        return self.variance*self.mean_system_response.A(self.mean_system_response.AT(x))
    def sqrtSigma(self, x):
        return torch.sqrt(self.variance)*self.mean_system_response.A(x)
    def invSigma(self, x):
        return (1/self.variance)*self.mean_system_response.pinvATA(x)
    

# Rician Noise
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2254141/
    


# okay now lets do the same with chest x-rays. it is going to be a normal model of Poisson likelihood with a constant mean
# A is a diagonal matrix, or elementwise multiplication with a vector
# A^T is the same as A
# (A^T A)^-1 is the inverse of the diagonal vector squared
# B is the inverse of the diagonal vector


if __name__ == "__main__":

    if False:
        from datasets import MICRONS_Dataset

        my_MICRONS_Dataset = MICRONS_Dataset('../20231104_microns/data/microns_samples', verbose=True)

        X = my_MICRONS_Dataset[0:200]

        likelihood = MultivariateGaussianMeasurementLikelihood(IdentityMeanSystemResponse(), ScalarNoiseCovariance(100))
        Y = likelihood.sample(X)

        Z = likelihood.pinvA(Y)

        # wide figure
        fig = plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        im1 = plt.imshow(X[0, 0, :, :].T, vmin=100, vmax=148, cmap='gray')
        plt.subplot(1, 3, 2)
        im2 = plt.imshow(Y[0, 0, :, :].T, cmap='gray')
        # plt.colorbar()
        plt.subplot(1, 3, 3)
        im3 = plt.imshow(Z[0, 0, :, :].T, vmin=100, vmax=148, cmap='gray')

        nFrames = 200

        def updatefig(i):
            print('Animating frame %d/%d' % (i, nFrames))
            im1.set_array(X[i, 0, :, :].T)
            im2.set_array(Y[i, 0, :, :].T)
            im3.set_array(Z[i, 0, :, :].T)
            plt.subplot(1, 3, 1)
            plt.title('X %d' % i)
            plt.subplot(1, 3, 2)
            plt.title('Y %d' % i)
            plt.subplot(1, 3, 3)
            plt.title('Z %d' % i)
            return im1, im2, im3
        
        ani = animation.FuncAnimation(fig, updatefig, frames=range(nFrames), blit=True)
        writer = animation.writers['ffmpeg'](fps=15)
        ani.save('MICRONS_AGWN.mp4', writer=writer, dpi=300)


        



        from datasets import CelebA_Dataset

        my_CelebA_Dataset = CelebA_Dataset('../../data/celeba_aligned/img_align_celeba/', verbose=True)

        X = my_CelebA_Dataset[0:200]

        likelihood = MultivariateGaussianMeasurementLikelihood(DownSampleMeanSystemResponse(4), ScalarNoiseCovariance(0.01))
        Y = likelihood.sample(X)

        Z = likelihood.pinvA(Y)

        # wide figure
        fig = plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        im1 = plt.imshow(X[0].permute(1,2,0))
        plt.subplot(1, 3, 2)
        im2 = plt.imshow(Y[0].permute(1,2,0))
        # plt.colorbar()
        plt.subplot(1, 3, 3)
        im3 = plt.imshow(Z[0].permute(1,2,0))

        nFrames = 200

        def updatefig(i):
            print('Animating frame %d/%d' % (i, nFrames))
            im1.set_array(X[i].permute(1,2,0))
            im2.set_array(Y[i].permute(1,2,0))
            im3.set_array(Z[i].permute(1,2,0))
            plt.subplot(1, 3, 1)
            plt.title('X %d' % i)
            plt.subplot(1, 3, 2)
            plt.title('Y %d' % i)
            plt.subplot(1, 3, 3)
            plt.title('Z %d' % i)
            return im1, im2, im3
        
        ani = animation.FuncAnimation(fig, updatefig, frames=range(nFrames), blit=True)
        writer = animation.writers['ffmpeg'](fps=15)
        ani.save('CelebA_Downsample8_AGWN.mp4', writer=writer, dpi=300)



        from datasets import SynthRad2023Task1MRISlice_Dataset

        my_SynthRad2023Task1MRISlice_Dataset = SynthRad2023Task1MRISlice_Dataset('../../data/SynthRad2023/Task1/brain/', verbose=True)

        nFrames = 1000


        X = my_SynthRad2023Task1MRISlice_Dataset[0:nFrames]

        mean_response = UndersampledMRIMeanSystemResponse(256)
        mask_rfft2 = mean_response.mask_rfft2.repeat(1,1,256,1)
        mask = torch.fft.irfft2(mask_rfft2, norm='ortho')
        mask = mask.permute(0, 1, 3, 2)
        noise_kernel = 1000*mask
        noise_covariance = StationaryNoiseCovariance(noise_kernel, kernel=True)
        likelihood = MultivariateGaussianMeasurementLikelihood(mean_response, noise_covariance)
        Y = likelihood.sample(X)

        Z = likelihood.pinvA(Y)

        fig = plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        im1 = plt.imshow(X[0, 0, :, :].numpy().T, cmap='gray', vmin=0, vmax=400)
        # plt.colorbar()
        plt.subplot(1, 3, 2)
        im2 = plt.imshow(torch.fft.fftshift(torch.log(torch.abs(torch.fft.fft2(Y[0, 0, :, :], norm='ortho'))), dim=(-1,-2)).numpy().T, cmap='gray',vmin=-2, vmax=8)
        # plt.colorbar()
        plt.subplot(1, 3, 3)
        im3 = plt.imshow(Z[0, 0, :, :].numpy().T, cmap='gray', vmin=0, vmax=400)
        # plt.colorbar()


        def updatefig(i):
            print('Animating frame %d/%d' % (i, nFrames))
            im1.set_array(X[i, 0, :, :].numpy().T)
            im2.set_array(torch.fft.fftshift(torch.log(torch.abs(torch.fft.fft2(Y[i, 0, :, :], norm='ortho'))), dim=(-1,-2)).numpy().T)
            im3.set_array(Z[i, 0, :, :].numpy().T)
            plt.subplot(1, 3, 1)
            plt.title('X %d' % i)
            plt.subplot(1, 3, 2)
            plt.title('Y %d' % i)
            plt.subplot(1, 3, 3)
            plt.title('Z %d' % i)
            return im1, im2, im3

        ani = animation.FuncAnimation(fig, updatefig, frames=range(nFrames), blit=True)
        writer = animation.writers['ffmpeg'](fps=15)
        ani.save('SynthRad2023Task1MRISlice_AGWN.mp4', writer=writer, dpi=300)


        from datasets import SynthRad2023Task1CTSlice_Dataset
        # Load the dataset
        my_SynthRad2023Task1CTSlice_Dataset = SynthRad2023Task1CTSlice_Dataset('../../data/SynthRad2023/Task1/brain/', verbose=True)
        nFrames = 200

        # Assuming X is loaded as per the instructions, with appropriate adjustments for CT
        X = my_SynthRad2023Task1CTSlice_Dataset[30:30+nFrames]

        # Load U, S, V from local .npy files for the DenseSVDMeanSystemResponse
        U = torch.tensor(np.load('U.npy'), dtype=torch.float32)
        S = torch.tensor(np.load('S.npy'), dtype=torch.float32)
        V = torch.tensor(np.load('V.npy'), dtype=torch.float32)

        # convert to tensor
        U = torch.tensor(U, dtype=torch.float32)
        S = torch.tensor(S, dtype=torch.float32)
        V = torch.tensor(V, dtype=torch.float32)

        # Replace mean_response and noise_covariance with corresponding CT versions
        # Define shapes for CT data
        xShape = [256, 256]  # Adjust according to your CT data
        yShape = [10, 375]  # Adjust based on the transformation
        mean_response = DenseSVDMeanSystemResponse(U, S, V, xShape, yShape)

        # Placeholder for the noise covariance for CT data, adjust as needed
        noise_covariance = ScalarNoiseCovariance(100000)

        # Define the likelihood as per your CT context
        likelihood = MultivariateGaussianMeasurementLikelihood(mean_response, noise_covariance)

        # Sample Y using the likelihood model
        Y = likelihood.sample(X)

        # Reconstruct Z using the pseudo-inverse of A
        # Z = likelihood.pinvA(Y)
        Z = likelihood.mean_system_response.pinvA(Y)

        # Prepare the animation
        fig = plt.figure(figsize=(15, 5))

        # Display the first frame to set up the plots
        im1 = plt.subplot(1, 3, 1).imshow(X[0, :, :].numpy().T, cmap='gray',vmin=-200, vmax=200)
        im2 = plt.subplot(1, 3, 2).imshow(Y[0, :, :].numpy(), cmap='gray',aspect='auto')
        im3 = plt.subplot(1, 3, 3).imshow(Z[0, :, :].numpy().T, cmap='gray',vmin=-200, vmax=200)

        def updatefig(i):
            print('Animating frame %d/%d' % (i, nFrames))
            im1.set_data(X[i, :, :].numpy().T)
            im2.set_data(Y[i, :, :].numpy())
            im3.set_data(Z[i, :, :].numpy().T)
            return im1, im2, im3

        ani = animation.FuncAnimation(fig, updatefig, frames=range(nFrames), blit=True)
        writer = animation.writers['ffmpeg'](fps=15)
        ani.save('SynthRad2023Task1CTSlice_AGWN.mp4', writer=writer, dpi=300)

    from datasets import CovidChestXRay_Dataset
    nFrames = 20

    my_CovidChestXRay_Dataset = CovidChestXRay_Dataset('../../data/chest_xray/train/COVID19/', verbose=True)
    X = my_CovidChestXRay_Dataset[0:nFrames]

    mean_system_response = GaussianFilterMeanSystemResponse(3)
    # noise_covariance = DiagonalNoiseCovariance(0.1*X)
    noise_covariance = MeanResponseAppliedToWhiteNoiseCovariance(0.1, mean_system_response)
    likelihood = MultivariateGaussianMeasurementLikelihood(mean_system_response, noise_covariance)
    Y = likelihood.sample(X)

    Z = likelihood.mean_system_response.pinvA(Y)

    # wide figure
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    im1 = plt.imshow(X[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
    plt.subplot(1, 3, 2)
    im2 = plt.imshow(Y[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
    plt.subplot(1, 3, 3)
    im3 = plt.imshow(Z[0, 0, :, :], cmap='gray', vmin=0, vmax=1)


    def updatefig(i):
        print('Animating frame %d/%d' % (i, nFrames))
        im1.set_array(X[i, 0, :, :])
        im2.set_array(Y[i, 0, :, :])
        im3.set_array(Z[i, 0, :, :])
        plt.subplot(1, 3, 1)
        plt.title('X %d' % i)
        plt.subplot(1, 3, 2)
        plt.title('Y %d' % i)
        plt.subplot(1, 3, 3)
        plt.title('Z %d' % i)
        return im1, im2, im3

    ani = animation.FuncAnimation(fig, updatefig, frames=range(nFrames), blit=True)
    writer = animation.writers['ffmpeg'](fps=15)
    ani.save('CovidChestXRay.mp4', writer=writer, dpi=300)
