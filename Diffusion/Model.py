
   
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class Swish(nn.Module):
    """
    定义swish激活函数，可参考https://blog.csdn.net/bblingbbling/article/details/107105648
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    定义``时间嵌入``模块
    """
    def __init__(self, T, d_model, dim):
        """
        初始的time-embedding是由一系列不同频率的正弦、余弦函数采样值表示，
        即：[[sin(w_0*x), cos(w_0*x)],
            [sin(w_1*x), cos(w_1*x)],
             ...,
            [sin(w_T)*x, cos(w_T*x)]], 维度为 T * d_model
        在本实例中，频率范围是[0:T], x在1e-4~1范围，共d_model // 2个离散点；将sin, cos并在一起组成d_model个离散点
        Args:
            T: int, 总迭代步数，本实例中T=1000
            d_model: 输入维度(通道数/初始embedding长度)
            dim: 输出维度(通道数)
        """
        assert d_model % 2 == 0
        super().__init__()
        # 前两行计算x向量，共64个点
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        # T个时间位置组成频率部分
        pos = torch.arange(T).float()
        # 两两相乘构成T*(d_model//2)的矩阵，并assert形状
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        # 计算不同频率sin, cos值，判断形状，并reshape到T*d_model
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        # MLP层，通过初始编码计算提取特征后的embedding
        # 包含两个线性层，第一个用swish激活函数，第二个不使用激活函数
        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    """
    通过stride=2的卷积层进行降采样
    """
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    """
    通过conv+最近邻插值进行上采样
    """
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    """
    自注意力模块，其中线性层均用kernel为1的卷积层表示
    """
    def __init__(self, in_ch):
        # ``self.proj_q``, ``self.proj_k``, ``self.proj_v``分别用于学习query, key, value
        # ``self.proj``作为自注意力后的线性投射层
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        # 输入经过组归一化以及全连接层后分别得到query, key, value
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        # 用矩阵乘法计算query与key的相似性权重w
        # 其中的``torch.bmm``的效果是第1维不动，第2，3维的矩阵做矩阵乘法，
        # 如a.shape=(_n, _h, _m), b.shape=(_n, _m, _w) --> torch.bmm(a, b).shape=(_n, _h, _w)
        # 矩阵运算后得到的权重要除以根号C, 归一化(相当于去除通道数对权重w绝对值的影响)
        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        # 再用刚得到的权重w对value进行注意力加权，操作也是一次矩阵乘法运算
        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)

        # 最后经过线性投射层输出，返回值加上输入x构成跳跃连接(残差连接)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    """
    残差网络模块
    """
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        """
        Args:
            in_ch: int, 输入通道数
            out_ch: int, 输出通道数
            tdim: int, time-embedding的长度/维数
            dropout: float, dropout的比例
            attn: bool, 是否使用自注意力模块
        """
        super().__init__()
        # 模块1: gn -> swish -> conv
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        # time_embedding 映射层: swish -> fc
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        # 模块2: gn -> swish -> dropout -> conv
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        # 如果输入输出通道数不一样，则添加一个过渡层``shortcut``, 卷积核为1, 否则什么也不做
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        # 如果需要加attention, 则添加一个``AttnBlock``, 否则什么也不做
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)                           # 输入特征经过模块1编码
        h += self.temb_proj(temb)[:, :, None, None]  # 将time-embedding加入到网络
        h = self.block2(h)                           # 将混合后的特征输入到模块2进一步编码

        h = h + self.shortcut(x)                     # 残差连接
        h = self.attn(h)                             # 经过自注意力模块(如果attn=True的话)
        return h


class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        """

        Args:
            T: int, 总迭代步数，本实例中T=1000
            ch: int, UNet第一层卷积的通道数，每下采样一次在这基础上翻倍, 本实例中ch=128
            ch_mult: list, UNet每次下采样通道数翻倍的乘数，本实例中ch_mult=[1,2,3,4]
            attn: list, 表示在第几次降采样中使用attention
            num_res_blocks: int, 降采样或者上采样中每一层次的残差模块数目
            dropout: float, dropout比率
        """
        super().__init__()
        # assert确保需要加attention的位置小于总降采样次数
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        # 将time-embedding从长度为ch初始化编码到tdim = ch * 4
        tdim = ch * 4
        # 实例化初始的time-embedding层
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        # 实例化头部卷积层
        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)

        # 实例化U-Net的编码器部分，即降采样部分，每一层次由``num_res_blocks``个残差块组成
        # 其中chs用于记录降采样过程中的各阶段通道数，now_ch表示当前阶段的通道数
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):  # i表示列表ch_mult的索引, mult表示ch_mult[i]
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        # 实例化U-Net编码器和解码器的过渡层，由两个残差块组成
        # 这里我不明白为什么第一个残差块加attention, 第二个不加……问就是``工程科学``
        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        # 实例化U-Net的解码器部分, 与编码器几乎对称
        # 唯一不同的是，每一层次的残差块比编码器多一个，
        # 原因是第一个残差块要用来融合当前特征图与跳转连接过来的特征图，第二、三个才是和编码器对称用来抽特征
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        # 尾部模块: gn -> swish -> conv, 目的是回到原图通道数
        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        # 注意这里只初始化头部和尾部模块，因为其他模块在实例化的时候已经初始化过了
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
    print(y.shape)

