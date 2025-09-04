import torch
import torch.nn as nn


"""
input image size: (batch, channel, D, H, W)

"""


class XGradient(nn.Module):
    def __init__(self, in_channel):
        super(XGradient, self).__init__()
        # self.dx = [[[[[0, 0, 0],
        #              [-1 / 2, 0, 1 / 2],
        #              [0, 0, 0]]]*3]] * in_channel    # 2-order central difference
        self.dx = [[[[[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]],

                     [[0, 0, 0],
                      [-1 / 2, 0, 1 / 2],
                      [0, 0, 0]],

                     [[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]]
                     ]]] * in_channel  # 2-order central difference
        self.conv_dx = nn.Conv3d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, padding_mode='replicate',
                                 groups=in_channel, bias=False)
        self.conv_dx.weight = nn.Parameter(torch.FloatTensor(self.dx), requires_grad=False)

    def forward(self, x):
        # Conv2D input: [batch, C, D, H, W]
        dx_value = self.conv_dx(x)
        return dx_value


class YGradient(nn.Module):
    def __init__(self, in_channel):
        super(YGradient, self).__init__()

        self.dy = [[[[[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]],

                     [[0, -1 / 2, 0],
                      [0, 0, 0],
                      [0, 1 / 2, 0]],

                     [[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]]
                     ]]] * in_channel  # 2-order central difference
        self.conv_dy = nn.Conv3d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, padding_mode='replicate',
                                 groups=in_channel, bias=False)
        self.conv_dy.weight = nn.Parameter(torch.FloatTensor(self.dy), requires_grad=False)

    def forward(self, x):
        # Conv2D input: [batch, C, D, H, W]
        dy_value = self.conv_dy(x)
        return dy_value


class ZGradient(nn.Module):
    def __init__(self, in_channel):
        super(ZGradient, self).__init__()

        self.dz = [[[[[-1 / 2, -1 / 2, -1 / 2],
                      [-1 / 2, -1 / 2, -1 / 2],
                      [-1 / 2, -1 / 2, -1 / 2]],

                     [[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]],

                     [[1 / 2, 1 / 2, 1 / 2],
                      [1 / 2, 1 / 2, 1 / 2],
                      [1 / 2, 1 / 2, 1 / 2]]
                     ]]] * in_channel  # 2-order central difference
        self.conv_dz = nn.Conv3d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, padding_mode='replicate',
                                 groups=in_channel, bias=False)
        self.conv_dz.weight = nn.Parameter(torch.FloatTensor(self.dz), requires_grad=False)

    def forward(self, x):
        # Conv2D input: [batch, C, D, H, W]
        dz_value = self.conv_dz(x)
        return dz_value


class TimeDerivative(nn.Module):
    def __init__(self, variable_type, device):
        super(TimeDerivative, self).__init__()

        # input tensor; [b, t, d, h, w] --> [b, t, d, h, w]
        # calculate the differentiations along the t axis
        self.variable_type = variable_type
        self.device = device

        self.layer_0_p = torch.full(size=(50, 50), fill_value=0.1596775866105e8).unsqueeze(0)
        self.layer_1_p = torch.full(size=(50, 50), fill_value=0.1577430881238e8).unsqueeze(0)
        self.layer_2_p = torch.full(size=(50, 50), fill_value=0.1558079129567e8).unsqueeze(0)
        self.layer_3_p = torch.full(size=(50, 50), fill_value=0.1538722846012e8).unsqueeze(0)
        self.layer_4_p = torch.full(size=(50, 50), fill_value=0.1519363185176e8).unsqueeze(0)
        self.layer_5_p = torch.full(size=(50, 50), fill_value=0.1500000000000e8).unsqueeze(0)
        self.min_pressure, self.max_pressure = 14724027.0, 19856224.0
        self.init_p = torch.cat((self.layer_0_p, self.layer_1_p, self.layer_2_p, self.layer_3_p, self.layer_4_p, self.layer_5_p), dim=0)
        self.init_p = (self.init_p - self.min_pressure) / (self.max_pressure - self.min_pressure)  # (6, 50, 50)
        self.init_p = self.init_p.unsqueeze(0)  # (1, 6, 50, 50)

    def forward(self, x, init_state=None):
        b, t, d, h, w = x.shape
        if self.variable_type == 'saturation':
            init_state = torch.zeros((b, 1, d, h, w)).to(self.device)
        if self.variable_type == 'pressure':
            init_state = self.init_p.expand(b, 1, d, h, w).to(self.device)
        out = torch.diff(x, n=1, dim=1, prepend=init_state)
        return out


if __name__ == "__main__":
    my_device = 'cuda'

    x1 = torch.arange(48.).reshape((1, 1, 3, 4, 4)).to(my_device)  # 3D geo information
    print(f'x1 has shape: {x1.shape}, x1 = \n{x1}')
    mm_mm = XGradient(1).to(my_device)
    y1 = mm_mm(x1)
    print(f'y1 has shape: {y1.shape}, y1 = \n{y1}')

    m_t = TimeDerivative('pressure')
    print(torch.max(m_t.init_p))
