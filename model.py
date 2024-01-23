from torch import nn
import torch.nn.init as init
import torch
import config as c

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

# Dense connection
class ResidualDenseBlock_out(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # initialization
        initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5

class DKiS_block(nn.Module):
    def __init__(self, subnet_constructor=ResidualDenseBlock_out, clamp=c.clamp, in_1=3, in_2=3, alpha=1.0):
        super().__init__()

        self.split_len1 = in_1
        self.split_len2 = in_2

        self.clamp = clamp
        self.f = subnet_constructor(self.split_len1, self.split_len2)
        self.g = subnet_constructor(self.split_len1, self.split_len2)
        self.h = subnet_constructor(self.split_len2, self.split_len1)
        self.alpha = alpha

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def shuffle(self, x, k, rev=False):
        # [bs, c, w, h]
        bs, c, w, h = x.shape
        if rev:
            k = k.argsort()
        x = x.reshape([bs, c, 4, w // 4, h])
        x = torch.permute(x, [0, 1, 2, 4, 3])
        x = torch.reshape(x, [bs, c, 16, h // 4, w // 4])
        x = x[:, :, k]
        x = x.reshape([bs, c, 4, h, w // 4])
        x = torch.permute(x, [0, 1, 2, 4, 3])
        x = x.reshape([bs, c, w, h])
        return x

    def forward(self, x, k, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        k1 = k[0]
        k2 = k[1]
        if not rev:

            y1 = x1 + self.alpha * self.f(x2)
            x2 = x2 * k1
            x2 = self.shuffle(x2, k2, rev)
            y2 = self.e(self.g(y1)) * (x2) + self.h(y1)

        else:

            y2 = (x2 - self.h(x1)) / self.e(self.g(x1))
            y2 = self.shuffle(y2, k2, rev)
            y2 = y2 / k1
            y1 = x1 - self.alpha * self.f(y2)

        return torch.cat((y1, y2), 1)

class DKiS(nn.Module):

    def __init__(self, alpha_list, in_1=12, in_2=12):
        super().__init__()

        self.inv1 =  DKiS_block(in_1=in_1, in_2=in_2, alpha=alpha_list[0])
        self.inv2 =  DKiS_block(in_1=in_1, in_2=in_2, alpha=alpha_list[1])
        self.inv3 =  DKiS_block(in_1=in_1, in_2=in_2, alpha=alpha_list[2])
        self.inv4 =  DKiS_block(in_1=in_1, in_2=in_2, alpha=alpha_list[3])
        self.inv5 =  DKiS_block(in_1=in_1, in_2=in_2, alpha=alpha_list[4])
        self.inv6 =  DKiS_block(in_1=in_1, in_2=in_2, alpha=alpha_list[5])
        self.inv7 =  DKiS_block(in_1=in_1, in_2=in_2, alpha=alpha_list[6])
        self.inv8 =  DKiS_block(in_1=in_1, in_2=in_2, alpha=alpha_list[7])
        self.inv9 =  DKiS_block(in_1=in_1, in_2=in_2, alpha=alpha_list[8])
        self.inv10 = DKiS_block(in_1=in_1, in_2=in_2, alpha=alpha_list[9])
        self.inv11 = DKiS_block(in_1=in_1, in_2=in_2, alpha=alpha_list[10])
        self.inv12 = DKiS_block(in_1=in_1, in_2=in_2, alpha=alpha_list[11])
        self.inv13 = DKiS_block(in_1=in_1, in_2=in_2, alpha=alpha_list[12])
        self.inv14 = DKiS_block(in_1=in_1, in_2=in_2, alpha=alpha_list[13])
        self.inv15 = DKiS_block(in_1=in_1, in_2=in_2, alpha=alpha_list[14])
        self.inv16 = DKiS_block(in_1=in_1, in_2=in_2, alpha=alpha_list[15])

    def forward(self, x, k, rev=False):

        if not rev:

            out = self.inv1(x, k[0])
            out = self.inv2(out, k[1])
            out = self.inv3(out, k[2])
            out = self.inv4(out, k[3])
            out = self.inv5(out, k[4])
            out = self.inv6(out, k[5])
            out = self.inv7(out, k[6])
            out = self.inv8(out, k[7])

            out = self.inv9(out, k[8])
            out = self.inv10(out, k[9])
            out = self.inv11(out, k[10])
            out = self.inv12(out, k[11])
            out = self.inv13(out, k[12])
            out = self.inv14(out, k[13])
            out = self.inv15(out, k[14])
            out = self.inv16(out, k[15])

        else:
            out = self.inv16(x, k[15], rev=True)
            out = self.inv15(out, k[14], rev=True)
            out = self.inv14(out, k[13], rev=True)
            out = self.inv13(out, k[12], rev=True)
            out = self.inv12(out, k[11], rev=True)
            out = self.inv11(out, k[10], rev=True)
            out = self.inv10(out, k[9], rev=True)
            out = self.inv9(out, k[8],rev=True)

            out = self.inv8(out, k[7], rev=True)
            out = self.inv7(out, k[6], rev=True)
            out = self.inv6(out, k[5], rev=True)
            out = self.inv5(out, k[4], rev=True)
            out = self.inv4(out, k[3], rev=True)
            out = self.inv3(out, k[2], rev=True)
            out = self.inv2(out, k[1], rev=True)
            out = self.inv1(out, k[0], rev=True)

        return out



def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad and 'alpha' not in key:
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5':
                param.data.fill_(0.)

