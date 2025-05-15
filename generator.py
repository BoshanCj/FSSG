import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function
from torch.nn import init
from .involution import involution
from random import *

# 定义一个用于生成和存储随机排列的类
class BatchPermutation:
    def __init__(self):
        self.current_perm = None
        
    def get_permutation(self, batch_size, device):
        """获取当前批次的随机排列，如果不存在则创建新的"""
        if self.current_perm is None or self.current_perm.size(0) != batch_size:
            self.current_perm = torch.randperm(batch_size, device=device)
        return self.current_perm
    
    def reset(self):
        """重置当前排列，下次调用将生成新的排列"""
        self.current_perm = None

# 创建全局排列生成器实例
global_perm_generator = BatchPermutation()

class Spectral_Weight(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False, device=0):
        super(Spectral_Weight, self).__init__()
        self.f_inv_11 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, dilation, groups, bias)
        self.f_inv_12 = involution(in_channels, kernel_size, 1)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(device)
        
        # 添加拉普拉斯噪声的可学习参数
        self.noise_scale = nn.Parameter(torch.tensor(0.01), requires_grad=True).to(device)
        self.privacy_budget = nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)
        self.device = device

    def laplace_noise(self, shape, sensitivity=1.0):
        """生成拉普拉斯噪声用于差分隐私"""
        # 根据隐私预算和敏感度计算噪声比例
        scale = sensitivity / max(self.privacy_budget.abs(), 1e-10)
        # 生成拉普拉斯分布噪声
        noise = torch.distributions.Laplace(0, scale).sample(shape).to(self.device)
        # 应用可学习的噪声强度参数
        return noise * self.noise_scale

    def forward(self, X_h, idx_swap=None):
        # Apply involution and convolution
        X_h = self.f_inv_11(self.f_inv_12(X_h))
        
        # Extract dimensions
        N, C, H, W = X_h.size()
        
        # Process each channel as a 1D signal (more suitable for spectral information)
        # Reshape to [N, C, H*W] to process each spectral band as 1D signal
        X_h_flat = X_h.view(N, C, -1)
        
        # 1D FFT for spectral processing (along the spatial dimension)
        f = torch.fft.fft(X_h_flat, dim=2)
        abs_f = torch.abs(f)
        angle_f = torch.angle(f)
        
        # 添加拉普拉斯噪声到角度信息以保护隐私
        angle_noise = self.laplace_noise(angle_f.shape)
        angle_f_noisy = angle_f + angle_noise
        
        # Mix batch samples - 使用传入的idx_swap或获取全局的排列
        if idx_swap is None:
            idx_swap = global_perm_generator.get_permutation(N, X_h.device)
        
        # Mix magnitudes across batches while keeping spectral structure
        # 注意：只混合abs_f，保持angle_f不变
        abs_f_mixed = self.alpha * abs_f + (1 - self.alpha) * abs_f[idx_swap]
        angle_f_mixed = angle_f_noisy  # 使用添加了噪声的角度，不再混合
        
        # Reconstruct signal
        f_reconstructed = abs_f * torch.exp(1j * angle_f_mixed)
        X_h_processed = torch.abs(torch.fft.ifft(f_reconstructed, dim=2))
        
        # Reshape back to original dimensions
        return X_h_processed.view(N, C, H, W)

class Spatial_Weight(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False, device=0):
        super(Spatial_Weight, self).__init__()
        self.Conv_weight = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(device)
        self.device = device
        
    def forward(self, X_h, idx_swap=None):
        # Extract dimensions
        N, C, H, W = X_h.size()
        
        # 2D FFT for spatial image processing 
        f = torch.fft.fftn(X_h, dim= (-2,-1))
        
        # Get magnitude and phase
        abs_f = torch.abs(f)
        angle_f = torch.angle(f)
        
        # Mix batch samples - 使用传入的idx_swap或获取全局的排列
        if idx_swap is None:
            idx_swap = global_perm_generator.get_permutation(N, X_h.device)
        
        # Apply frequency domain mixing
        abs_f = self.alpha * abs_f + (1 - self.alpha) * abs_f[idx_swap]
        angle_f = self.alpha * angle_f + (1 - self.alpha) * angle_f[idx_swap]
        
        # Combine magnitude and phase
        f = abs_f * torch.exp(1j * angle_f)

        # Reconstruct signal
        X_h_processed = torch.abs(torch.fft.ifftn(f, dim=(-2,-1)))

        # Apply convolution
        X_h_processed = self.Conv_weight(X_h_processed)
        
        return X_h_processed

class NormalNorm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        with torch.no_grad():
            mu = weight.mean()
            std = weight.std()
        return (weight - mu) / std

    @staticmethod
    def apply(module, name):
        fn = NormalNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        module.register_buffer(name, weight)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn = self.compute_weight(module)
        setattr(module, self.name, weight_sn)


def spectral_norm(module, name='weight'):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        NormalNorm.apply(module, name)
    return module


def spectral_init(module, gain=1):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        init.xavier_uniform_(module.weight, gain)
    return spectral_norm(module)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class GNet1(nn.Module):
    def __init__(self, args):
        super(GNet1, self).__init__()
        ch = args.GIN_ch
        device = getattr(args, 'device', 0)
        
        self.Spectral_Weight_11 = Spectral_Weight(args.n_bands, ch, kernel_size=3, stride=1, padding=1, device=device)
        self.Spatial_Weight_11 = Spatial_Weight(args.n_bands, ch, kernel_size=3, stride=1, padding=1, device=device)
        self.Spectral_Weight_12 = Spectral_Weight(ch, ch, kernel_size=3, stride=1, padding=1, device=device)
        self.Spatial_Weight_12 = Spatial_Weight(ch, ch, kernel_size=3, stride=1, padding=1, device=device)
        self.Spectral_Weight_13 = Spectral_Weight(ch, ch, kernel_size=3, stride=1, padding=1, device=device)
        self.Spatial_Weight_13 = Spatial_Weight(ch, ch, kernel_size=3, stride=1, padding=1, device=device)
        self.generate1 = nn.Conv2d(ch, args.n_bands, 3, padding=1)
        self.activate1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(ch)
        self.Weight_Alpha1 = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
        self.__initialize_weights()

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1.0)

    def normalize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                weight = m.weight.data * 1.0
                m.weight.data = (m.weight.data - weight.mean()) / weight.std()

    def forward(self, x):
        # 为每个批次生成随机排列
        N = x.size(0)
        idx_swap = global_perm_generator.get_permutation(N, x.device)
        
        # First layer
        out1 = self.Spectral_Weight_11(x, idx_swap) + self.Spatial_Weight_11(x, idx_swap)
        out1 = self.activate1(self.bn1(out1))
        
        # 重新生成新的排列用于下一层
        global_perm_generator.reset()
        idx_swap = global_perm_generator.get_permutation(N, x.device)
        
        # Second layer
        out1 = self.Spectral_Weight_12(out1, idx_swap) + self.Spatial_Weight_12(out1, idx_swap)
        out1 = self.activate1(self.bn1(out1))
        
        # 重新生成新的排列用于下一层
        global_perm_generator.reset()
        idx_swap = global_perm_generator.get_permutation(N, x.device)
        
        # Third layer
        out1 = self.Spectral_Weight_13(out1, idx_swap) + self.Spatial_Weight_13(out1, idx_swap)
        out1 = self.activate1(self.bn1(out1))
        
        # Output layer
        out1 = self.generate1(out1)
        
        # Weighted skip connection
        weight_alpha1 = F.softmax(self.Weight_Alpha1, dim=0)
        out1 = weight_alpha1[0] * x + weight_alpha1[1] * out1
        
        return out1

class GNet2(nn.Module):
    def __init__(self, args):
        super(GNet2, self).__init__()
        ch = args.GIN_ch
        device = getattr(args, 'device', 0)
        
        self.Spectral_Weight_21 = Spectral_Weight(args.n_bands, ch, kernel_size=3, stride=1, padding=1, device=device)
        self.Spatial_Weight_21 = Spatial_Weight(args.n_bands, ch, kernel_size=3, stride=1, padding=1, device=device)
        self.Spectral_Weight_22 = Spectral_Weight(ch, ch, kernel_size=3, stride=1, padding=1, device=device)
        self.Spatial_Weight_22 = Spatial_Weight(ch, ch, kernel_size=3, stride=1, padding=1, device=device)
        self.Spectral_Weight_23 = Spectral_Weight(ch, ch, kernel_size=3, stride=1, padding=1, device=device)
        self.Spatial_Weight_23 = Spatial_Weight(ch, ch, kernel_size=3, stride=1, padding=1, device=device)
        self.generate2 = nn.Conv2d(ch, args.n_bands, 3, padding=1)
        self.activate2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(ch)
        self.Weight_Alpha2 = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
        self.__initialize_weights()
       
    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1.0)

    def normalize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                weight = m.weight.data * 1.0
                m.weight.data = (m.weight.data - weight.mean()) / weight.std()

    def forward(self, x):
        # 为每个批次生成随机排列
        N = x.size(0)
        idx_swap = global_perm_generator.get_permutation(N, x.device)
        
        # First layer
        out2 = self.Spectral_Weight_21(x, idx_swap) + self.Spatial_Weight_21(x, idx_swap)
        out2 = self.activate2(self.bn2(out2))
        
        # 重新生成新的排列用于下一层
        global_perm_generator.reset()
        idx_swap = global_perm_generator.get_permutation(N, x.device)
        
        # Second layer
        out2 = self.Spectral_Weight_22(out2, idx_swap) + self.Spatial_Weight_22(out2, idx_swap)
        out2 = self.activate2(self.bn2(out2))
        
        # 重新生成新的排列用于下一层
        global_perm_generator.reset()
        idx_swap = global_perm_generator.get_permutation(N, x.device)
        
        # Third layer
        out2 = self.Spectral_Weight_23(out2, idx_swap) + self.Spatial_Weight_23(out2, idx_swap)
        out2 = self.activate2(self.bn2(out2))
        
        # Output layer
        out2 = self.generate2(out2)
        
        # Weighted skip connection
        weight_alpha2 = F.softmax(self.Weight_Alpha2, dim=0)
        out2 = weight_alpha2[0] * x + weight_alpha2[1] * out2

        return out2

class SSDGnet(nn.Module):
    def __init__(self, args, device=0):
        super(SSDGnet, self).__init__()
        
        # Set device in args for child networks
        args.device = device
        
        self.Net1 = GNet1(args)
        self.Net2 = GNet2(args)
        
        self.b = nn.Parameter(torch.tensor(0.3), requires_grad=True).to(device)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(device)

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1.0)

    def normalize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                weight = m.weight.data * 1.0
                m.weight.data = (m.weight.data - weight.mean()) / weight.std()

    def forward(self, x):
        # 确保每次前向传播重置全局排列
        global_perm_generator.reset()
        
        # Get outputs from both networks
        out1 = self.Net1(x)
        
        # 确保每次前向传播重置全局排列
        global_perm_generator.reset()
        
        out2 = self.Net2(x)

        return out1, out2
