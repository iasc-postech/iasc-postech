import math
import torch
import torch.nn as nn
from einops import rearrange, repeat


def weights_init(m):
    classname = m.__class__.__name__
    if "Linear" in classname or "Embedding" == classname:
        print(f"Initializing Module {classname}.")
        nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)
    # elif "Parameter" in classname:
    #     return nn.init.trunc_normal_(m, 0.0, 0.02)

class Attention(nn.Module):
    """
    Simple Self-Attention algorithm. Potential for optimization using a non-quadratic attention mechanism in complexity.
    -> Linformer, Reformer etc.
    """
    def __init__(self, dim=768, heads=8):
        super(Attention, self).__init__()
        d = dim // heads
        self.q, self.k, self.v = nn.Linear(dim, d), nn.Linear(dim, d), nn.Linear(dim, d)
        self.norm = d ** 0.5
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        qk = torch.softmax(q @ torch.transpose(k, 1, 2) / self.norm, dim=1)
        qk = self.dropout(qk)
        attn = torch.matmul(qk, v)
        return attn


class MultiHeadAttention(nn.Module):
    """
    Implementation of MultiHeadAttention, splitting it up to multiple Self-Attention layers and concatenating
    the results and subsequently running it through one linear layer of same dimension.
    """
    def __init__(self, dim=768, heads=8):
        super(MultiHeadAttention, self).__init__()
        self.self_attention_heads = nn.ModuleList([Attention(dim, heads) for _ in range(heads)])
        self.projector = nn.Linear(dim, dim)

    def forward(self, x):
        for i, sa_head in enumerate(self.self_attention_heads):
            if i == 0:
                out = sa_head(x)
            else:
                out = torch.cat((out, sa_head(x)), axis=-1)
        out = self.projector(out)
        return out


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class Encoder(nn.Module):
    """
    Transformer encoder using MultiHeadAttention and MLP along with skip connections and LayerNorm
    """
    def __init__(self, dim=768, hidden_dim=3072):
        super(Encoder, self).__init__()
        # self.MultiHeadAttention = MultiHeadAttention(dim)
        self.MultiHeadAttention = nn.MultiheadAttention(dim, num_heads=8, batch_first=True, dropout=0.1)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = nn.Sequential(*[
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        ])
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # attn = self.MultiHeadAttention(x)
        attn, _ = self.MultiHeadAttention(x, x, x)#, need_weights=False)
        attn = self.dropout(attn)
        x = x.add(attn)
        x = self.LayerNorm1(x)
        mlp = self.MLP(x)
        x = x.add(mlp)
        x = self.LayerNorm2(x)
        return x

class BidirectionalTransformer_avgpool(nn.Module):
    def __init__(self, args, label_nc=200):
        super(BidirectionalTransformer_avgpool, self).__init__()
        self.args = args
        self.label_nc = label_nc + 2
        self.tok_emb = nn.Embedding(self.label_nc , args['dim'])
        self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(args['inter_seg_size']**2, args['dim'])), 0., 0.02)

        self.ln = nn.LayerNorm(args['dim'], eps=1e-12)
        self.drop = nn.Dropout(p=0.1)       
        self.blocks = nn.Sequential(*[Encoder(args['dim'], args['hidden_dim']) for _ in range(args['n_layers'])])

        self.final_token_prediction = nn.Sequential(*[
            nn.Linear(in_features=args['dim'], out_features=args['dim']),
            nn.GELU(),
            nn.LayerNorm(args['dim'], eps=1e-12)
        ])
        self.final_bias = nn.Parameter(torch.zeros(args['initial_seg_size'], args['initial_seg_size'], self.label_nc))
        
        if (args['initial_seg_size'] != args['inter_seg_size']):
            self.neural_renderer = NeuralRenderer(args)
        else:
            self.neural_renderer = None
        
        self.avgpool = nn.AvgPool2d(kernel_size=4)

        self.apply(weights_init)

    def forward(self, x):

        token_embeddings = self.tok_emb(x)
        token_embeddings = rearrange(token_embeddings, "B H W E-> B E H W")
        token_embeddings = self.avgpool(token_embeddings)
        token_embeddings = rearrange(token_embeddings, "B E H W-> B (H W) E")

        position_embeddings = self.pos_emb
        embed = self.drop(self.ln(token_embeddings + position_embeddings))
        embed = self.blocks(embed)

        if self.neural_renderer is not None:
            embed = rearrange(embed, "B (H W) E -> B E H W", H = self.args['inter_seg_size'])
            embed = self.neural_renderer(embed)
            embed = rearrange(embed, "B E H W -> B H W E")
        else:
            embed = rearrange(embed, "B (H W) E -> B H W E", H = self.args['inter_seg_size'])

        final_embed = self.final_token_prediction(embed)
        final_logits = torch.matmul(final_embed, self.tok_emb.weight.T) + self.final_bias

        return final_logits

class BidirectionalTransformer_sconv(nn.Module):
    def __init__(self, args, label_nc=200):
        super(BidirectionalTransformer_sconv, self).__init__()
        self.args = args
        self.label_nc = label_nc + 2
        self.seg2embed = nn.Embedding(self.label_nc , args['dim'])
        self.token_dim = 8*args['dim']

        self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(args['inter_seg_size']**2, self.token_dim)), 0., 0.02)

        self.ln = nn.LayerNorm(self.token_dim , eps=1e-12)

        self.drop = nn.Dropout(p=0.1)       
        self.blocks = nn.Sequential(*[Encoder(self.token_dim , args['hidden_dim']) for _ in range(args['n_layers'])])

        self.final_token_prediction = nn.Sequential(*[
            nn.Linear(in_features=args['dim'], out_features=args['dim']),
            nn.GELU(),
            nn.LayerNorm(args['dim'], eps=1e-12)
        ])
        self.final_bias = nn.Parameter(torch.zeros(args['initial_seg_size'], args['initial_seg_size'], self.label_nc))
        
        if (args['initial_seg_size'] != args['inter_seg_size']):
            self.neural_renderer = NeuralRenderer(args)
        else:
            self.neural_renderer = None
        
        # self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.sconv1 = nn.Conv2d(in_channels=args['dim'], out_channels=2*args['dim'], kernel_size=4, stride=2, padding=1)
        self.sconv2 = nn.Conv2d(in_channels=2*args['dim'], out_channels=4*args['dim'], kernel_size=4, stride=2, padding=1)
        self.sconv3 = nn.Conv2d(in_channels=4*args['dim'], out_channels=8*args['dim'], kernel_size=4, stride=2, padding=1)

        self.apply(weights_init)

    def forward(self, x):

        seg_embeddings = self.seg2embed(x)
        seg_embeddings = rearrange(seg_embeddings, "B H W E-> B E H W")
        seg_embeddings = self.sconv3(self.sconv2(self.sconv1(seg_embeddings)))
        token_embeddings = rearrange(seg_embeddings, "B E H W-> B (H W) E")
        position_embeddings = self.pos_emb
        embed = self.drop(self.ln(token_embeddings + position_embeddings))

        embed = self.blocks(embed)

        if self.neural_renderer is not None:
            embed = rearrange(embed, "B (H W) E -> B E H W", H = self.args['inter_seg_size'])
            embed = self.neural_renderer(embed)
            embed = rearrange(embed, "B E H W -> B H W E")
        else:
            embed = rearrange(embed, "B (H W) E -> B H W E", H = self.args['inter_seg_size'])

        final_embed = self.final_token_prediction(embed)
        final_logits = torch.matmul(final_embed, self.seg2embed.weight.T) + self.final_bias

        return final_logits

class BidirectionalTransformer(nn.Module):
    def __init__(self, args):
        super(BidirectionalTransformer, self).__init__()
        self.args = args
        self.num_image_tokens = args['inter_seg_size']**2
        self.tok_emb = nn.Embedding(args['label_nc'] + 2, args['dim'])
        self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(self.num_image_tokens, args['dim'])), 0., 0.02)
        self.ln = nn.LayerNorm(args['dim'], eps=1e-12)
        self.drop = nn.Dropout(p=0.1)       
        self.blocks = nn.Sequential(*[Encoder(args['dim'], args['hidden_dim']) for _ in range(args['n_layers'])])

        self.final_token_prediction = nn.Sequential(*[
            nn.Linear(in_features=args['dim'], out_features=args['dim']),
            nn.GELU(),
            nn.LayerNorm(args['dim'], eps=1e-12)
        ])

        self.final_bias = nn.Parameter(torch.zeros(self.args['final_seg_size'], self.args['final_seg_size'], args['label_nc'] + 2))
        if self.args['inter_ce_loss']:
            self.inter_token_prediction = nn.Sequential(*[
                nn.Linear(in_features=args['dim'], out_features=args['dim']),
                nn.GELU(),
                nn.LayerNorm(args['dim'], eps=1e-12)
                ])
            self.inter_bias = nn.Parameter(torch.zeros(self.args['inter_seg_size']*self.args['inter_seg_size'], args['label_nc'] + 2))

        self.apply(weights_init)

        self.no_neural_renderer = args['no_neural_renderer']
        
        if not self.no_neural_renderer and (args['final_seg_size'] != args['inter_seg_size']):
            self.neural_renderer = NeuralRenderer(args)

    def forward(self, x):

        x = rearrange(x, "B H W -> B (H W)")
        token_embeddings = self.tok_emb(x)
        position_embeddings = self.pos_emb
        embed = self.drop(self.ln(token_embeddings + position_embeddings))
        embed = self.blocks(embed)

        inter_logits = None
        if self.args['inter_ce_loss']:
            inter_embed = self.inter_token_prediction(embed)
            inter_logits = torch.matmul(inter_embed, self.tok_emb.weight.T) + self.inter_bias

        if not self.no_neural_renderer:
            embed = rearrange(embed, "B (H W) E -> B E H W", H = self.args['inter_seg_size'])
            embed = self.neural_renderer(embed)
            embed = rearrange(embed, "B E H W -> B H W E")
        else:
            embed = rearrange(embed, "B (H W) E -> B H W E", H = self.args['inter_seg_size'])

        final_embed = self.final_token_prediction(embed)
        final_logits = torch.matmul(final_embed, self.tok_emb.weight.T) + self.final_bias

        return inter_logits, final_logits


from model.bsc.swin_transformer import BasicLayer, PatchEmbed, PatchMerging, PatchUpsample

class BidirectionalSwinTransformer_sconv(nn.Module):
    def __init__(self, args, label_nc=200):
        super(BidirectionalSwinTransformer_sconv, self).__init__()
        self.args = args
        self.label_nc = label_nc + 2
        self.seg2embed = nn.Embedding(self.label_nc , args['dim'])
        self.token_dim = 8*args['dim']
        self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(args['inter_seg_size']**2, self.token_dim)), 0., 0.02)
        self.ln = nn.LayerNorm(self.token_dim , eps=1e-12)
        self.drop = nn.Dropout(p=0.1)

        # build layers
        self.swin_layers = nn.ModuleList()

        # from 32 -> 8 -> 32
        res = args['inter_seg_size']
        depths = [2, 3, 4, 3, 2]
        # merge = ['None', 'None', 'None', 'None', 'None']
        num_heads = 8
        window_sizes = [4, 8, 8, 8, 4]
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        for i_layer, depth in enumerate(depths):
        
            # if merge[i_layer] == 'down':
            #     merge_layer = PatchMerging
            # elif merge[i_layer] == 'up':
            #     merge_layer = PatchUpsample
            # else:
            #     merge_layer = None

            layer = BasicLayer(dim=self.token_dim,
                               input_resolution=(res,res),
                               depth=depth,
                               num_heads=num_heads,
                               window_size=window_sizes[i_layer],
                               mlp_ratio=2.,
                               qkv_bias=True, qk_scale=None,
                               drop=0, attn_drop=0,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=nn.LayerNorm,
                               downsample=None,
                               use_checkpoint=False)

            self.swin_layers.append(layer)


        self.final_token_prediction = nn.Sequential(*[
            nn.Linear(in_features=args['dim'], out_features=args['dim']),
            nn.GELU(),
            nn.LayerNorm(args['dim'], eps=1e-12)
        ])
        self.final_bias = nn.Parameter(torch.zeros(args['initial_seg_size'], args['initial_seg_size'], self.label_nc))
        
        if (args['initial_seg_size'] != args['inter_seg_size']):
            self.neural_renderer = NeuralRenderer(args)
        else:
            self.neural_renderer = None
        
        # self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.sconv1 = nn.Conv2d(in_channels=args['dim'], out_channels=2*args['dim'], kernel_size=4, stride=2, padding=1)
        self.sconv2 = nn.Conv2d(in_channels=2*args['dim'], out_channels=4*args['dim'], kernel_size=4, stride=2, padding=1)
        self.sconv3 = nn.Conv2d(in_channels=4*args['dim'], out_channels=8*args['dim'], kernel_size=4, stride=2, padding=1)

        self.apply(weights_init)


    def forward(self, x):

        seg_embeddings = self.seg2embed(x)
        seg_embeddings = rearrange(seg_embeddings, "B H W E-> B E H W")
        seg_embeddings = self.sconv3(self.sconv2(self.sconv1(seg_embeddings)))
        token_embeddings = rearrange(seg_embeddings, "B E H W-> B (H W) E")
        position_embeddings = self.pos_emb
        embed = self.drop(self.ln(token_embeddings + position_embeddings))

        for layer in self.swin_layers:
            embed = layer(embed)

        seg_embeddings = self.seg2embed(x)
        seg_embeddings = rearrange(seg_embeddings, "B H W E-> B E H W")
        seg_embeddings = self.sconv3(self.sconv2(self.sconv1(seg_embeddings)))
        token_embeddings = rearrange(seg_embeddings, "B E H W-> B (H W) E")
        position_embeddings = self.pos_emb
        embed = self.drop(self.ln(token_embeddings + position_embeddings))

        for layer in self.swin_layers:
            embed = layer(embed)

        if self.neural_renderer is not None:
            embed = rearrange(embed, "B (H W) E -> B E H W", H = self.args['inter_seg_size'])
            embed = self.neural_renderer(embed)
            embed = rearrange(embed, "B E H W -> B H W E")
        else:
            embed = rearrange(embed, "B (H W) E -> B H W E", H = self.args['inter_seg_size'])

        final_embed = self.final_token_prediction(embed)
        final_logits = torch.matmul(final_embed, self.seg2embed.weight.T) + self.final_bias

        return final_logits

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.surplus_conv = conv3x3(in_channels, out_channels, 1)
        else:
            self.surplus_conv = None

    def forward(self, x):
        residual = x
        
        if self.surplus_conv is not None:
            residual = self.surplus_conv(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class nr_block(nn.Module):
    def __init__(self, in_channels=512, out_channels=512):
        super(nr_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=2)
        self.residual_block = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        
        x = self.upsample(x)
        x = self.residual_block(x)
        return x

class NeuralRenderer(nn.Module):
    def __init__(self, args):
        super(NeuralRenderer, self).__init__()
        import math
        self.num_blocks = int(math.log2((args['initial_seg_size']//args['inter_seg_size'])))
        self.blocks = nn.Sequential(*[nr_block((2**(3-i))*args['dim'], (2**(2-i))*args['dim']) for i in range(self.num_blocks)])
    
    def forward(self, x):
        x = self.blocks(x)
        return x
