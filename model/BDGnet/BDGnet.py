import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class BoundaryModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        # self.conv2_1 = nn.Conv2d(
        #     dim, dim, (1, 21), padding=(0, 10), groups=dim)
        # self.conv2_2 = nn.Conv2d(
        #     dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)
        self.norm=nn.BatchNorm2d(dim)
        self.act=nn.GELU()

    def forward(self, x):
        shortcut = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        # attn_2 = self.conv2_1(attn)
        # attn_2 = self.conv2_2(attn_2)
        attn = attn * (attn_0 + attn_1) #+ attn_2

        attn = self.conv3(attn)

        return self.act(self.norm(attn + shortcut))

class Boundary_Branch(nn.Module):
    def __init__(self, dim,block_num):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i_block in range(block_num):
            block=BoundaryModule(dim)
            self.blocks.append(block)
        # self.boundarymodule1=BoundaryModule(dim)
        # self.boundarymodule2=BoundaryModule(dim)
        self.conv=nn.Conv2d(dim,2,1)

    def forward(self, x):
        shortcut=x.clone()
        for block in self.blocks:
            x=block(x)
        # x=self.boundarymodule1(x)
        # x=self.boundarymodule2(x)
        boundary=self.conv(x)

        return boundary,x+shortcut

class Detail_Local_Module(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1=nn.Conv2d(dim,dim,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(dim,dim,kernel_size=3,padding=1)
        self.norm1=nn.BatchNorm2d(dim)
        self.norm2=nn.BatchNorm2d(dim)
        self.act1=nn.ReLU()
        self.act2=nn.ReLU()

    def forward(self,x):
        shortcut=x.clone()
        x=self.act1(self.norm1(self.conv1(x)))
        x=self.norm2(self.conv2(x))
        return self.act2(x+shortcut)

class Detail_Local_Branch(nn.Module):
    def __init__(self, dim,block_num):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i_block in range(block_num):
            block=Detail_Local_Module(dim)
            self.blocks.append(block)
        # self.detail_module1=Detail_Local_Module(dim)
        # self.detail_module2=Detail_Local_Module(dim)
        self.conv=nn.Conv2d(dim,6,1)
    def forward(self,x):
        shortcut=x.clone()
        for block in self.blocks:
            x=block(x)
        # x=self.detail_module2(self.detail_module1(x))
        detail_map=self.conv(x)



        return detail_map,x+shortcut



class Global_attention_Module(nn.Module):
    def __init__(self, dim,L=256, eps=1e-6, kernel_function=nn.ReLU()):
        super().__init__()
        self.pos_emb=nn.Parameter(torch.randn(1, L, dim))
        self.linear_Q = nn.Linear(dim, dim)
        self.linear_K = nn.Linear(dim, dim)
        self.linear_V = nn.Linear(dim, dim)

        self.linear_Q1 = nn.Linear(dim, dim)
        self.linear_K1 = nn.Linear(dim, dim)

        self.eps = eps
        self.kernel_fun = kernel_function
        self.gamma = nn.Parameter(torch.zeros(1))
        self.norm=nn.BatchNorm2d(dim)
        self.act=nn.ReLU()


    def forward(self, x):
        B,C,H,W=x.shape

        x=x.permute(0,2,3,1).reshape(B,-1,C).contiguous()
        x=x+self.pos_emb
        Q = self.linear_Q(x)  # blc
        K = self.linear_K(x)  # blc
        V = self.linear_V(x)  # blc
        Q1 = self.linear_Q1(x)
        K1 = self.linear_K1(x)

        Q = self.kernel_fun(Q)
        K = self.kernel_fun(K)
        K = K.transpose(-2, -1)  # bcl
        KV = torch.einsum("bml, blc->bmc", K, V)  # bcc

        Z = 1 / (torch.einsum("blc,bc->bl", Q, K.sum(dim=-1) + self.eps))  # bl

        result = torch.einsum("blc,bcc,bl->blc", Q, KV, Z)  # blc

        mid_result = (Q1.transpose(-1, -2) @ K1).softmax(dim=-1)
        result = result @ mid_result

        x = x + self.gamma * result
        x=x.permute(0,2,1).reshape(B,C,H,W).contiguous()
        x=self.act(self.norm(x))

        return x

class Global_Branch(nn.Module):
    def __init__(self, dim, block_num=2):
        super().__init__()
        self.blocks=nn.ModuleList()
        for i_block in range(block_num):
            block=Global_attention_Module(dim)
            self.blocks.append(block)
        # self.global_module1=Global_attention_Module(dim)
        # self.global_module2=Global_attention_Module(dim)
        self.conv=nn.Conv2d(dim,6,1)
    def forward(self,x):
        shortcut=x.clone()
        for block in self.blocks:
            x=block(x)
        # x=self.global_module2(self.global_module1(x))
        global_map=self.conv(x)
        return global_map,x+shortcut


#还没试过这个
# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
#         self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
#         self.norm1=nn.BatchNorm2d(out_channels)
#         self.norm2=nn.BatchNorm2d(out_channels)
#         self.conv1_1=nn.Conv2d(in_channels,out_channels,kernel_size=1)
#         self.act1=nn.ReLU()
#         self.act2=nn.ReLU()
#
#     def forward(self,x):
#         shortcut=x.clone()
#         x=self.act1(self.norm1(self.conv1(x)))
#         x=self.norm2(self.conv2(x))
#         shortcut=self.conv1_1(shortcut)
#         return self.act2(x+shortcut)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True,upsample_ratio=2):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=upsample_ratio, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# class Spatial_Channel_Attn(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0 = nn.Conv2d(dim, dim, 3, padding=1)
#         self.avg_pool=nn.AdaptiveAvgPool2d((1,1))
#         self.conv1=nn.Conv2d(dim,1,1)
#         self.conv2=nn.Conv2d(dim,dim,1)
#         self.act=nn.Sigmoid()
#         self.norm=nn.BatchNorm2d(dim)
#         self.relu=nn.ReLU()
#
#
#     def forward(self, x):
#         shortcut=x.clone()
#         x=self.conv0(x)
#         x1=self.avg_pool(x)
#         x2=self.conv1(x)
#         x=x1*x2*x
#         x=self.act(x)
#         x=self.relu(self.norm(x+shortcut))
#         return x
#
# class Fuse(nn.Module):
#     def __init__(self, dim):
#         super(Fuse, self).__init__()
#         self.attn_G=Spatial_Channel_Attn(dim)
#         self.attn_D = Spatial_Channel_Attn(dim)
#         self.attn_B = Spatial_Channel_Attn(dim)
#
#
#
#
#     def forward(self, G, D, B):
#         G=self.attn_G(G)
#         D = self.attn_D(D)
#         B = self.attn_B(B)
#         x=torch.cat((G,D,B),dim=1)
#
#         return x

class segmenthead(nn.Module):
    def __init__(self,in_channels,mid_channels,out_channels):
        super(segmenthead,self).__init__()
        self.bn1=nn.BatchNorm2d(in_channels)
        self.conv1=nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        return out

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class BDGnet(nn.Module):
    def __init__(self, n_classes, n_channels=3,dim=64, bilinear=True,istrain=True,block_num=[2,2,2]):
        super(BDGnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.istrain=istrain

        self.inc = DoubleConv(n_channels, dim)
        self.down1 = Down(dim, dim*2)
        self.down2 = Down(dim*2, dim*4)
        self.down3 = Down(dim*4, dim*8)
        #factor = 2 if bilinear else 1
        self.down4 = Down(dim*8, dim*8)
        self.boundary_branch=Boundary_Branch(dim,block_num[0])
        self.detail_local_branch=Detail_Local_Branch(4*dim,block_num[1])
        self.up_detail_branch=nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.detail_adj_channel=nn.Conv2d(4*dim, dim, kernel_size=1)
        self.global_branch=Global_Branch(8*dim,block_num[2])
        self.up_global_branch = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.global_adj_channel = nn.Conv2d(8 * dim, dim, kernel_size=1)
        #self.fuse=Fuse(dim=dim)
        #self.outc = OutConv(3*dim, n_classes)
        self.head_seg=segmenthead(3*dim,dim,n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        ifo_boundary, x_boundary=self.boundary_branch(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        ifo_detail, x_detail = self.detail_local_branch(x3)
        x_detail=self.detail_adj_channel(self.up_detail_branch(x_detail))
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        ifo_global, x_global=self.global_branch(x5)
        x_global=self.global_adj_channel(self.up_global_branch(x_global))
        #print(x_boundary.shape,x_detail.shape)
        x=torch.cat((x_boundary,x_detail),dim=1)
        x=torch.cat((x,x_global),dim=1)

        # x=self.fuse(x_global,x_detail,x_boundary)

        #可视化分支
        # ifo_detail = nn.functional.upsample(ifo_detail, (256, 256))
        # ifo_detail = torch.sum(ifo_detail, dim=1)
        # ifo_detail=torch.sigmoid(ifo_detail)
        # print(ifo_detail.shape)
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(3.36, 3.36))
        # # plt.axis('off')  # 去坐标轴
        # # plt.xticks([])  # 去刻度
        # # plt.figure(dpi=1000)
        # plt.imshow(ifo_detail[0].detach().cpu().numpy())
        # # plt.imsave('pic2.png',ifo_boundary[0][0].detach().cpu().numpy())
        # plt.show()

        # logits = self.outc(x)
        logits=self.head_seg(x)
        if self.istrain:
            return {'out':logits}#,ifo_boundary,ifo_detail,ifo_global
        else:
            return {'out':logits}