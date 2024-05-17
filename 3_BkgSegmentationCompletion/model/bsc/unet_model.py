""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from einops import rearrange, repeat
class ResBasicBlock(nn.Module):
	def __init__(self, channel_num):
		super(ResBasicBlock, self).__init__()
		
		#TODO: 3x3 convolution -> relu
		#the input and output channel number is channel_num
		self.conv_block1 = nn.Sequential(
			nn.Conv2d(channel_num, channel_num, 3, padding=1),
			nn.BatchNorm2d(channel_num),
			nn.ReLU(),
		) 
		self.conv_block2 = nn.Sequential(
			nn.Conv2d(channel_num, channel_num, 3, padding=1),
			nn.BatchNorm2d(channel_num),
		)
		self.relu = nn.ReLU()
	
	def forward(self, x):
		
		#TODO: forward
		residual = x
		x = self.conv_block1(x)
		x = self.conv_block2(x)
		x = x + residual
		out = self.relu(x)
		return out
class UNet(nn.Module):
    def __init__(self, args, label_nc=200, bilinear=False):
        super(UNet, self).__init__()
        self.in_channels = args['dim']
        self.out_channels = args['dim']
        self.label_nc = label_nc + 2
        self.seg2embed = nn.Embedding(self.label_nc , args['dim'])
        self.bilinear = bilinear

        self.inc = DoubleConv(self.in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        # self.resblock = ResBasicBlock(1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, self.out_channels)

        self.final_bias = nn.Parameter(torch.zeros(args['initial_seg_size'], args['initial_seg_size'], self.label_nc))

    def forward(self, x):
        
        x = self.seg2embed(x)
        x = rearrange(x, "B H W E-> B E H W")
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # x6 = self.resblock(x5) ## bottleneck
        # x = self.up1(x6,x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        
        x = rearrange(x, "B E H W -> B H W E")
        logits = torch.matmul(x, self.seg2embed.weight.T) + self.final_bias

        return logits