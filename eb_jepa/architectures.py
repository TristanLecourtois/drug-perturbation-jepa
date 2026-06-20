from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from eb_jepa.nn_utils import TemporalBatchMixin, init_module_weights


class conv3d2(nn.Sequential):
    """Simple 3D convnet with 2 layers."""

    def __init__(self, in_d, h_d, out_d, tk, ts, sk, ss, pad):
        super(conv3d2, self).__init__(
            nn.Conv3d(
                in_d, h_d, kernel_size=(tk, sk, sk), stride=(1, 1, 1), padding=pad
            ),
            nn.ReLU(),
            nn.Conv3d(
                h_d, out_d, kernel_size=(tk, sk, sk), stride=(ts, ss, ss), padding=pad
            ),
        )
        self.apply(init_module_weights)
        self.input_dim = in_d
        self.hidden_dim = h_d
        self.output_dim = out_d
        # t_shift is the index (in the time dimension) of the first output
        # cannot see its coresponding input
        if pad == "valid":
            self.t_shift = 2 * tk - 1
        elif pad == "same":
            self.t_shift = 2 * (tk - 1)
        else:
            raise NameError("invalid padding for con3d2. Must be 'valid' or 'same'")


class ResidualBlock(nn.Module):
    """Standard residual block with skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet5(TemporalBatchMixin, nn.Module):
    """
    A lightweight ResNet with 5 layers (2 blocks).
    Supports both 4D [B, C, H, W] and 5D [B, C, T, H, W] inputs via TemporalBatchMixin.
    """

    def __init__(self, in_d, h_d, out_d, s1=1, s2=1, s3=1, avg_pool=False):
        super().__init__()
        self.avg_pool = avg_pool
        self.conv1 = nn.Conv2d(
            in_d, h_d, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(h_d)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = ResidualBlock(h_d, h_d, stride=s1)
        self.layer2 = ResidualBlock(h_d, h_d * 2, stride=s2)
        self.layer3 = ResidualBlock(h_d * 2, out_d, stride=s3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) if avg_pool else torch.nn.Identity()

    def _forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        if self.avg_pool:
            out = out.flatten(1)
        return out


class SimplePredictor(nn.Module):
    """Wrapper that concatenates states and actions channel-wise before prediction."""

    def __init__(self, predictor, context_length):
        super().__init__()
        self.predictor = predictor
        self.is_rnn = predictor.is_rnn
        self.context_length = context_length

    def forward(self, x, a):
        return self.predictor(torch.cat([x, a], dim=1))


class StateOnlyPredictor(SimplePredictor):
    """Wrapper for a simple predictor which concatenates states and actions channel wise."""

    def forward(self, x, a):
        # action not used on purpose
        prev_state = x[:, :, :-1]  # [B, C, T-1, H, W]
        next_state = x[:, :, 1:]  # [B, C, T-1, H, W]
        combined_xa = torch.cat((prev_state, next_state), dim=1)
        return self.predictor(combined_xa)


class ResUNet(TemporalBatchMixin, nn.Module):
    """
    A small UNet with residual encoder blocks and transposed-conv upsampling.
    Channels scale like h, 2h, 4h, 8h. Output keeps the input HxW.
    Supports both 4D [B, C, H, W] and 5D [B, C, T, H, W] inputs via TemporalBatchMixin.
    """

    def __init__(self, in_d, h_d, out_d, is_rnn=False):
        super().__init__()
        self.is_rnn = is_rnn
        # Stem
        self.conv1 = nn.Conv2d(
            in_d, h_d, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(h_d)
        self.relu = nn.ReLU(inplace=True)

        # Encoder
        self.enc1 = ResidualBlock(h_d, h_d, stride=1)  # H, W
        self.enc2 = ResidualBlock(h_d, 2 * h_d, stride=2)  # H/2, W/2
        self.enc3 = ResidualBlock(2 * h_d, 4 * h_d, stride=2)  # H/4, W/4
        self.bott = ResidualBlock(4 * h_d, 8 * h_d, stride=2)  # H/8, W/8

        # Decoder upsamples, then fuses skip with a residual block that reduces channels
        self.up3 = nn.ConvTranspose2d(8 * h_d, 4 * h_d, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(8 * h_d, 4 * h_d, stride=1)

        self.up2 = nn.ConvTranspose2d(4 * h_d, 2 * h_d, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(4 * h_d, 2 * h_d, stride=1)

        self.up1 = nn.ConvTranspose2d(2 * h_d, 1 * h_d, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(2 * h_d, 1 * h_d, stride=1)

        # Head
        self.head = nn.Conv2d(h_d, out_d, kernel_size=1)

    @staticmethod
    def _match_size(x, ref):
        # Guards against odd input sizes by resizing the upsample to the skip spatial dims
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(
                x, size=ref.shape[-2:], mode="bilinear", align_corners=False
            )
        return x

    def _forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))

        # Encoder with skips
        s1 = self.enc1(x0)  # h
        s2 = self.enc2(s1)  # 2h
        s3 = self.enc3(s2)  # 4h
        b = self.bott(s3)  # 8h

        # Decoder stage 3
        d3 = self.up3(b)
        d3 = self._match_size(d3, s3)
        d3 = torch.cat([d3, s3], dim=1)  # 4h + 4h = 8h
        d3 = self.dec3(d3)  # → 4h

        # Decoder stage 2
        d2 = self.up2(d3)
        d2 = self._match_size(d2, s2)
        d2 = torch.cat([d2, s2], dim=1)  # 2h + 2h = 4h
        d2 = self.dec2(d2)  # → 2h

        # Decoder stage 1
        d1 = self.up1(d2)
        d1 = self._match_size(d1, s1)
        d1 = torch.cat([d1, s1], dim=1)  # h + h = 2h
        d1 = self.dec1(d1)  # → h

        out = self.head(d1)  # → out_d channels
        return out


class Projector(nn.Module):
    """MLP projector built from a spec string like '256-512-128'."""

    def __init__(self, mlp_spec):
        super().__init__()
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        self.net = nn.Sequential(*layers)
        self.out_dim = f[-1]  # Store output dimension as attribute

    def forward(self, x):
        return self.net(x)


class DetHead(nn.Module):
    """Detection head that pools features and predicts binary maps."""

    def __init__(self, in_d, h_d, out_d):
        super().__init__()
        self.head = nn.Sequential(conv3d2(in_d, h_d, out_d, 1, 1, 3, 1, "same"))
        self.apply(init_module_weights)

    def forward(self, x):
        """Forward pass on predictor output of shape (B, C, T, H, W)."""
        # (Batch, Feature, Time, Height, Width)
        # [8, 8, T, 8, 8]
        x = [F.adaptive_avg_pool2d(x[:, :, t], (8, 8)) for t in range(x.shape[2])]
        x = torch.stack(x, 2)
        # [8, T, 8, 8]
        x = self.head(x).squeeze(1)

        return torch.sigmoid(x)

    @torch.no_grad()
    def score(self, preds, targets):

        scores = []
        for T in range(len(preds) - 1):
            x = preds[T]
            x = [F.adaptive_avg_pool2d(x[:, :, t], (8, 8)) for t in range(x.shape[2])]
            x = torch.stack(x, 2)
            x = self.head(x).squeeze(1)

            y = targets[:, T:]
            x = x[:, T:]

            ap = average_precision_score(
                y.flatten().detach().long().cpu().numpy(),
                x.flatten().detach().cpu().numpy(),
                average="weighted",
            )
            scores.append(ap)

        return scores


class ResnetBlock(nn.Module):
    """ResNet Block."""

    def __init__(self, num_features):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + identity)


class ResnetStack(nn.Module):
    """ResNet stack module."""

    def __init__(self, input_channels, num_features, num_blocks, max_pooling=True):
        super(ResnetStack, self).__init__()
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.max_pooling = max_pooling
        self.initial_conv = nn.Conv2d(
            input_channels, num_features, kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList(
            [ResnetBlock(num_features) for _ in range(num_blocks)]
        )
        if max_pooling:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.max_pool = nn.Identity()

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.max_pool(x)
        for block in self.blocks:
            x = block(x)
        return x


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    def __init__(
        self,
        width=1,
        stack_sizes=(16, 32, 32),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=2,
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(2, 65, 65),
    ):
        super(ImpalaEncoder, self).__init__()
        self.width = width
        self.stack_sizes = stack_sizes
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        self.input_shape = input_shape
        self.mlp_output_dim = mlp_output_dim

        input_channels = [input_channels] + list(stack_sizes)

        self.stack_blocks = nn.ModuleList(
            [
                ResnetStack(
                    input_channels=input_channels[i],
                    num_features=stack_size * width,
                    num_blocks=num_blocks,
                )
                for i, stack_size in enumerate(stack_sizes)
            ]
        )

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity()

        # Compute MLP input dimension dynamically
        with torch.no_grad():
            # Create a dummy input (assuming typical input size for this encoder)
            dummy_input = torch.zeros(1, *self.input_shape)  # (1, C, H, W)
            conv_out = dummy_input
            for stack_block in self.stack_blocks:
                conv_out = stack_block(conv_out)  # b c w h
            flattened_dim = conv_out.view(conv_out.size(0), -1).shape[1]  # c * w * h

        self.mlp = nn.Linear(flattened_dim, self.mlp_output_dim)

        if final_ln:
            self.final_ln = nn.LayerNorm(self.mlp_output_dim)
        else:
            self.final_ln = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            out: [B, D, T, 1, 1]
        """

        # [B, C, T, H, W] --> [T, B, C, H, W]
        (
            _,
            _,
            t,
            _,
            _,
        ) = x.shape
        x = x.permute(2, 0, 1, 3, 4)

        features = []

        for i in range(t):

            conv_out = x[i]

            for i, stack_block in enumerate(self.stack_blocks):
                conv_out = stack_block(conv_out)
                if self.dropout_rate is not None:
                    conv_out = self.dropout(conv_out)

            conv_out = F.relu(conv_out)
            if self.layer_norm:
                conv_out = nn.LayerNorm(conv_out.size()[1:])(conv_out)  # b c w h
            # flatten
            out = conv_out.view(conv_out.size(0), -1)
            out = self.mlp(out)
            out = self.final_ln(out)

            features.append(out)

        features = torch.stack(features, dim=1)

        features = features.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)

        return features


class MultiSourceFusion(nn.Module):
    """Fuse several embedding sources into one model latent, with a learned
    FALLBACK for missing sources (so any subset can be present per example).

    Mirrors multi-source foundation encoders: each source has its own linear
    projection to a common width; rows where a source is absent are replaced by
    a learned per-source fallback vector; the projected sources are concatenated
    and fused by an MLP. Sources can be cell- or gene-level embeddings
    (e.g. MosaicFM, expression-PCA, pathway activity, scGPT, ESM2, KGE).

    forward(sources): dict name -> (x[B, d_name], mask[B] bool). A name absent
    from the dict, or rows with mask=False, use that source's fallback.
    Returns [B, h_model].
    """

    def __init__(self, source_dims: dict, h_proj: int = 256, h_model: int = 256):
        super().__init__()
        self.names = list(source_dims)
        self.proj = nn.ModuleDict({n: nn.Linear(d, h_proj) for n, d in source_dims.items()})
        self.fallback = nn.ParameterDict(
            {n: nn.Parameter(torch.zeros(h_proj)) for n in self.names}
        )
        self.mlp = nn.Sequential(
            nn.Linear(h_proj * len(self.names), h_model), nn.GELU(),
            nn.Linear(h_model, h_model),
        )
        self.apply(init_module_weights)

    def forward(self, sources: dict):
        B = next(v[0].shape[0] for v in sources.values())
        parts = []
        for n in self.names:
            if n in sources:
                x, m = sources[n]
                p = self.proj[n](x)
                p = torch.where(m[:, None].bool(), p, self.fallback[n][None, :])
            else:
                p = self.fallback[n][None, :].expand(B, -1)
            parts.append(p)
        return self.mlp(torch.cat(parts, dim=-1))


class SetTransformer(nn.Module):
    """Perceiver-style set-transformer cell encoder over a gene panel.

    Every gene of the top-K panel is a TOKEN. A gene's token vector is the FUSION
    of one or more per-gene source embeddings, additively combined:

        token[b, g] = sum_s  W_s @ source_s[g]          # gene-init (multi-source)
                    + id_emb(g)                          # always-on learned source
                    + value_proj( expression[b, g] )     # the cell's own signal

    The "sources" are frozen per-gene tables aligned to the gene panel, e.g.
    scGPT gene embeddings, a biomedical-KG (KGE) gene vector, ESM2 of the gene's
    protein. None are required: with no source registered the encoder still runs
    on the learned gene-id embedding alone (so it trains today; real sources plug
    in later via `register_gene_source`).

    A small set of learned latents cross-attends to the K gene tokens (Perceiver),
    then self-attends, and is mean-pooled to the cell latent z. This is O(K * M)
    rather than O(K^2), so it scales to large panels — the GeneJEPA recipe.

    forward(x): x is [B, K] expression over the panel -> returns [B, out_d].
    """

    def __init__(self, n_genes, out_d=128, d_model=192, n_latents=32,
                 depth=2, heads=4, source_dims=None, ffn_mult=2):
        super().__init__()
        self.n_genes = n_genes
        self.d_model = d_model
        self.id_emb = nn.Embedding(n_genes, d_model)         # always-on learned source
        self.value_proj = nn.Linear(1, d_model)              # per-gene expression -> token
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)  # JEPA re-masking embedding
        self.src_proj = nn.ModuleDict()                      # filled by register_gene_source
        if source_dims:
            for name, d in source_dims.items():
                self.src_proj[name] = nn.Linear(d, d_model)
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.cross = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.cross_ln_q = nn.LayerNorm(d_model)
        self.cross_ln_kv = nn.LayerNorm(d_model)
        self.self_blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, heads, batch_first=True),
                "ln2": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(nn.Linear(d_model, d_model * ffn_mult), nn.GELU(),
                                     nn.Linear(d_model * ffn_mult, d_model)),
            }) for _ in range(depth)
        ])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_d))
        self.out_dim = out_d
        self.apply(init_module_weights)

    def register_gene_source(self, name, table):
        """Attach a frozen per-gene embedding table [n_genes, d] (scGPT/KGE/ESM2).
        Adds a trainable projection but keeps the table itself frozen."""
        table = torch.as_tensor(table, dtype=torch.float32)
        assert table.shape[0] == self.n_genes, f"{name}: {table.shape[0]} != {self.n_genes} genes"
        self.register_buffer(f"src_{name}", table, persistent=True)
        self.src_proj[name] = nn.Linear(table.shape[1], self.d_model)
        init_module_weights(self.src_proj[name])

    def _gene_base(self, device):
        """Per-gene token base = learned id + sum of frozen-source projections. [K, d]"""
        idx = torch.arange(self.n_genes, device=device)
        base = self.id_emb(idx)
        for name, proj in self.src_proj.items():
            table = getattr(self, f"src_{name}").to(device)
            base = base + proj(table)
        return base

    def forward(self, x, gene_mask=None):
        """x: [B, K] expression. gene_mask: optional [B, K] bool, True = gene is
        MASKED. Masked genes' tokens are replaced by the learned `mask_token`
        (JEPA re-masking: the encoder never sees the masked gene's value), so a
        context (masked) and a target (full) view share every other token."""
        B = x.shape[0]
        base = self._gene_base(x.device)                          # [K, d]
        tok = base[None] + self.value_proj(x[..., None])          # [B, K, d]
        if gene_mask is not None:
            mt = self.mask_token.to(tok.dtype).view(1, 1, -1)     # [1, 1, d]
            tok = torch.where(gene_mask.unsqueeze(-1), mt, tok)   # masked -> mask_token
        tok = self.cross_ln_kv(tok)
        q = self.cross_ln_q(self.latents)[None].expand(B, -1, -1)  # [B, M, d]
        z, _ = self.cross(q, tok, tok)                            # [B, M, d]
        for blk in self.self_blocks:
            h = blk["ln1"](z)
            a, _ = blk["attn"](h, h, h)
            z = z + a
            z = z + blk["ffn"](blk["ln2"](z))
        return self.head(z.mean(dim=1))                           # [B, out_d]


class LatentPredictor(nn.Module):
    """JEPA predictor head g_φ mapping a context (masked-view) global latent to
    the target (full-view) global latent. For mean-pooled encoders JEPA-DNA uses
    a small MLP predictor (the attention-predictor variant is for token-aggregated
    backbones); here the SetTransformer already pools to a single vector, so an
    MLP over the pooled latent is the matching design. Depth is configurable
    because the predictor's depth is the single biggest knob in their ablation.
    """

    def __init__(self, dim, hidden=None, depth=3):
        super().__init__()
        hidden = hidden or dim * 2
        layers, d = [], dim
        for _ in range(max(1, depth - 1)):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, dim)]
        self.net = nn.Sequential(*layers)
        self.apply(init_module_weights)

    def forward(self, z):
        return self.net(z)


class SetEncoder(TemporalBatchMixin, nn.Module):
    """Permutation-invariant, abundance-weighted DeepSets encoder for microbiome
    communities (or any set of token embeddings with per-token weights).

    A "community" at one timestep is a SET of OTUs. Each OTU is a token carrying
    a fixed sequence embedding (e.g. ProkBERT) plus its (log) relative abundance.
    The set is laid out as a [B, C, N, 1] image so it slots into the library's
    5D [B, C, T, N, 1] convention via TemporalBatchMixin (T folded into batch).

    Channel layout of the input:
      - channels [0 : emb_dim]   -> per-OTU sequence embedding
      - channel  [emb_dim]       -> log1p(relative abundance); 0 for padded slots

    Permutation invariance comes from a per-token 1x1-conv MLP followed by an
    abundance-weighted sum-pool over the N (token) axis. Padded OTUs carry
    abundance 0, so they get zero pooling weight and are ignored for free.

    Output: [B, out_d, 1, 1] per timestep -> [B, out_d, T, 1, 1] for a sequence,
    matching what the (RNN) predictor and the VC/SquareLoss expect.
    """

    def __init__(self, emb_dim=384, h_d=256, out_d=128, abundance_weighted=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.abundance_weighted = abundance_weighted
        in_d = emb_dim + 1  # embedding + log-abundance feature
        self.token_mlp = nn.Sequential(
            nn.Conv2d(in_d, h_d, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(h_d, h_d, kernel_size=1),
            nn.GELU(),
        )
        self.post = nn.Sequential(
            nn.Conv2d(h_d, out_d, kernel_size=1),
        )
        self.out_dim = out_d
        self.apply(init_module_weights)

    def _forward(self, x):
        # x: [B, emb_dim + 1, N, 1]
        logab = x[:, self.emb_dim : self.emb_dim + 1]  # [B, 1, N, 1]
        h = self.token_mlp(x)  # [B, h_d, N, 1]
        if self.abundance_weighted:
            w = F.relu(logab)  # padded slots (ab=0) -> weight 0
        else:
            w = (logab > 0).float()  # presence mask (unweighted mean over present)
        w = w / (w.sum(dim=2, keepdim=True) + 1e-6)  # normalize over the token axis
        z = (h * w).sum(dim=2, keepdim=True)  # [B, h_d, 1, 1] weighted pool
        z = self.post(z)  # [B, out_d, 1, 1]
        return z


class RNNPredictor(nn.Module):
    """GRU-based predictor for single-step state propagation."""

    def __init__(
        self,
        hidden_size: int = 512,
        action_dim: Optional[int] = 2,
        num_layers: int = 1,
        final_ln: Optional[torch.nn.Module] = None,
    ):
        super(RNNPredictor, self).__init__()

        self.num_layers = num_layers

        self.rnn = torch.nn.GRU(
            input_size=action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.final_ln = final_ln
        self.is_rnn = True
        self.context_length = 0

    def forward(self, state, action):
        """
        Propagate one step forward.

        Args:
            state: [B, D, 1, 1, 1]
            action: [B, A, 1]
        Returns:
            next_state: [B, D, 1, 1, 1]
        """
        # This only does one step
        rnn_state = state.flatten(1, 4).unsqueeze(0).contiguous()  # [1, B, D]
        rnn_input = action.squeeze(-1).unsqueeze(0).contiguous()  # [1, B, A]

        next_state, _ = self.rnn(rnn_input, rnn_state)

        next_state = self.final_ln(next_state)

        return next_state[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


class InverseDynamicsModel(nn.Module):
    """
    Predicts the action that caused a transition from state_t to state_t_plus_1.
    Used as auxiliary task for representation learning.
    """

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(init_module_weights)

    def forward(self, state_t, state_t_plus_1):
        """
        Args:
            state_t: State at time t, shape [B, D]
            state_t_plus_1: State at time t+1, shape [B, D]
        Returns:
            predicted_action: Action predicted to transform state_t to state_t_plus_1, shape [B, A]
        """
        combined_states = torch.cat([state_t, state_t_plus_1], dim=1)
        return self.model(combined_states)
