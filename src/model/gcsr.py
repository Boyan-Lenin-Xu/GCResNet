from model import common
import torch
import torch.nn as nn

def make_model(args, parent=False):
    return GCSR(args)

class GCSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(GCSR, self).__init__()

        n_resblocks = args.n_resblocks // 2
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        n_graph_features = 18
        n_ResGCN = 5
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body_pre = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body_pre.append(conv(n_feats, n_feats, kernel_size))

        #self.downsample_graph = nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, stride=2, padding=1)

        _adj = common.get_Adj(adj_file='./Adj_matrix/fullSR.mat')
        self.A = nn.Parameter(torch.from_numpy(_adj).float())

        GCN_body = [
            common.ResGCN(
                n_graph_features, _adj
            ) for _ in range(n_ResGCN)
        ]
        self.graoh_convhead = common.GraphConvolution(1, n_graph_features)
        self.graph_convtail = common.GraphConvolution(n_graph_features, 1)

        #self.upsample_graph = nn.ConvTranspose2d(n_feats, n_feats, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)


        m_body_after = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body_after.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]
        self.relu = nn.LeakyReLU(0.2)

        self.head = nn.Sequential(*m_head)
        self.m_body_pre = nn.Sequential(*m_body_pre)
        self.GCN_body = nn.Sequential(*GCN_body)
        self.m_body_after = nn.Sequential(*m_body_after)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.m_body_pre(x)
        yin = res #self.downsample_graph(res)

        yin = yin.permute(0,2,3,1)
        yin = yin.unsqueeze(4)
        #print(yin.size())

        adj = common.gen_adj(self.A).detach()
        #print(adj.size())
        res_g = self.graoh_convhead(yin, adj) #make(yin, adj) as dict to let it available in nn.Sequential
        res_g = self.GCN_body(res_g)
        res_g = self.graph_convtail(res_g, adj)
        res_g = self.relu(res_g)

        res_g = res_g.squeeze(4)
        yout = res_g.permute(0,3,1,2)
        res_up3 = yout + res #self.upsample_graph(yout)

        res = self.m_body_after(res_up3)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

