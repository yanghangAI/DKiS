import torch
from torch import nn


def round_diff(x):
    sign = torch.ones_like(x)
    sign[torch.floor(x) % 2 == 0] = -1
    y = sign * torch.cos(x * torch.pi) / 2
    out = torch.round(x) + y - y.detach()
    return out

def attack(img, method):

    if method[:8] == 'gaussian':
        level = int(method[8:])
        img = img + level * torch.randn(img.shape).to(img.device) / 255.
        img = img.clip(0, 1)


    elif method == 'round':
        img = img * 255
        img = round_diff(img)
        img = img / 255


    elif method == 'none':
        pass

    else:
        print('no attack is taken')

    return img

def gauss_noise(shape):
    noise = torch.randn(shape).cuda()
    return noise

def mse_loss(a, b):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(a, b)
    return loss

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def computePSNR(origin,pred):
    mse = torch.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * torch.log10(1.0/mse).item()

def ssim(x, y, c1=1e-6, c2=1e-6):
    # [bs, c, h, w]
    bs = x.shape[0]
    x = x.reshape([bs, -1])
    y = y.reshape([bs, -1])
    Ex = torch.mean(x, dim=-1) # [bs, ]
    Ey = torch.mean(y, dim=-1)
    x = torch.cat([x, y], dim=0)
    cov = torch.cov(x) # [2*bs, 2*bs]
    index = [torch.arange(bs), torch.arange(bs, 2*bs)]
    cxy = cov[index]
    cov = torch.diag(cov)
    cxx = cov[:bs]
    cyy = cov[bs:]
    s1 = (2 * Ex * Ey + c1) / ((Ex ** 2) + (Ey ** 2) + c1)
    s2 = (2 * cxy + c2) / (cxx + cyy + c2)
    s = s1 * s2
    return torch.mean(s).item()

def norm(x, a, b):
    x = x - a
    x = x / b
    return x

def inorm(x, a, b):
    x = x * b
    x = x + a
    return x

def ndwt_init(x, a, b):
    x = norm(x, a=a, b=b)
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def niwt_init(x, a, b):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    h = inorm(h, a=a, b=b)
    return h

class NDWT(nn.Module):
    def __init__(self, a, b):
        super(NDWT, self).__init__()
        self.requires_grad = True
        self.a = a
        self.b = b

    def forward(self, x):
        return ndwt_init(x, self.a, self.b)
class NIWT(nn.Module):
    def __init__(self, a, b):
        super(NIWT, self).__init__()
        self.a = a
        self.b = b
        self.requires_grad = True

    def forward(self, x):
        return niwt_init(x, self.a, self.b)