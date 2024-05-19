import monai
import torch

class DenoiserNetwork(torch.nn.Module):
    def __init__(self):
        super(DenoiserNetwork, self).__init__()
        self.unet = monai.networks.nets.UNet(spatial_dims=2, 
                                 in_channels=32, 
                                 out_channels=1, 
                                 channels=[32,64,128,256], 
                                 strides=[2,2,2,2],
                                 kernel_size=3, 
                                 up_kernel_size=3, 
                                 num_res_units=3, 
                                 act='PRELU', 
                                 norm='INSTANCE', 
                                 dropout=0.0, 
                                 bias=True, 
                                 adn_ordering='NDA')
        
        self.time_encoder = monai.networks.nets.FullyConnectedNet(
                                                    in_channels=1, 
                                                    out_channels=31, 
                                                    hidden_channels=[256,256,256], 
                                                    dropout=None, 
                                                    act='PRELU', 
                                                    bias=True, 
                                                    adn_ordering='NDA')


    def forward(self, z_t, t):
        t_enc = self.time_encoder(t)
        t_enc = t_enc.unsqueeze(-1).unsqueeze(-1)
        t_enc = t_enc.repeat(1,1,z_t.shape[2],z_t.shape[3])
        z_t_and_t = torch.cat([z_t, t_enc], dim=1)
        return self.unet(z_t_and_t)



from diffusers import UNet2DModel


class convMLP(torch.nn.Module):
    def __init__(   self, 
                    input_size, 
                    hidden_size, 
                    output_size, 
                    num_hidden_layers=4):
        
        super().__init__()

        # mudule list
        self.conv_layers = torch.nn.ModuleList()

        current_size = input_size
        for iLayer in range(num_hidden_layers):
            self.conv_layers.append(torch.nn.Conv2d(current_size, hidden_size, kernel_size=1, stride=1, padding=0))
            current_size = hidden_size
        self.conv_layers.append(torch.nn.Conv2d(current_size, output_size, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        for iLayer in range(len(self.conv_layers)-1):
            x = torch.nn.functional.relu(self.conv_layers[iLayer](x))
        x = self.conv_layers[-1](x)
        return x

class DiffusersUnet(torch.nn.Module):

    def __init__(   self,  
                    input_channels=1,
                    time_encoder_hidden_size=256,
                    image_size=512,
                    unet_in_channels=64, 
                    unet_base_channels=64,
                    unet_out_channels=1,
                    ):

        super().__init__()

        # Create a UNet2DModel for noise prediction given x_t and t
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=unet_out_channels,
            layers_per_block=2,
            norm_num_groups=1,
            block_out_channels=(unet_base_channels, unet_base_channels, 2*unet_base_channels, 2*unet_base_channels, 4*unet_base_channels, 4*unet_base_channels),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

        self.time_encoder = monai.networks.nets.FullyConnectedNet(
                                                    in_channels=1, 
                                                    out_channels=unet_in_channels-input_channels, 
                                                    hidden_channels=[time_encoder_hidden_size,time_encoder_hidden_size,time_encoder_hidden_size],
                                                    dropout=None, 
                                                    act='PRELU', 
                                                    bias=True, 
                                                    adn_ordering='NDA')

    def forward(self, z_t, t):

        while len(t.shape) < len(z_t.shape):
            t = t.unsqueeze(-1)

        # apply the MLP to the status concatenated with the time
        t_enc = self.time_encoder(t)

        # now repeat it so its the same size as z_t
        t_enc = t_enc.unsqueeze(-1).unsqueeze(-1)
        t_enc = t_enc.repeat(1,1,z_t.shape[2],z_t.shape[3])

        # now concatenate it with x_t
        z_t_and_t = torch.cat([z_t, t_enc], dim=1)

        # now apply the unet
        residual = self.unet(z_t_and_t,t.squeeze())[0]

        z_0_hat = z_t + (torch.sqrt(t))*residual

        # trying out an idea from consistency models...
        # eps = 1e-3
        # weight = (t-eps)**2.0
        # weight[t<eps] = 0.0
        # z_0_hat = (1.0 - weight)*z_t + weight*(z_0_hat)

        return z_0_hat