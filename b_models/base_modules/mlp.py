import math
import torch

# import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """Gaussian encoder, optionally conditioned on the diffusion timestep.

    `args.time_embedding_method_rec` selects the conditioning, mirroring the
    denoiser's `time_embedding_method`:
      - "none":      no time input at all (default; `forward(x)` as before).
      - "as_input":  project the embedding to input_dim and concatenate with x.
      - "per_layer": add a per-layer projection to every hidden pre-activation.
      - "film":      per-layer scale/shift, h <- (1 + scale) * h + shift, with
                     zero-initialised heads so it starts as identity conditioning.

    Time conditioning matters here because the encoder is applied to *noisy* inputs
    to form q(z | x_t). The correct posterior width varies by orders of magnitude
    across noise levels, so a t-independent encoder cannot represent that family and
    collapses to the prior. When `time_conditioned` is True, callers must pass `t`.
    """

    def __init__(self, args):
        super().__init__()

        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim_rec
        self.latent_dim = args.latent_dim
        self.bias = args.bias_rec
        self.num_layers = args.num_layers_rec

        self.time_method = getattr(args, "time_embedding_method_rec", "none")
        if self.time_method not in ("none", "as_input", "per_layer", "film"):
            raise ValueError(
                f"Unknown time_embedding_method_rec: {self.time_method}, "
                'expected "none", "as_input", "per_layer" or "film"'
            )
        self.time_conditioned = self.time_method != "none"

        if args.activation_rec == "relu":
            self.nonlin = nn.ReLU()
        elif args.activation_rec == "gelu":
            self.nonlin = nn.GELU()
        elif args.activation_rec == "silu":
            self.nonlin = nn.SiLU()
        else:
            raise ValueError('nonlin must be specified, e.g. "relu" or "gelu"')

        if self.time_conditioned:
            self.time_channels = args.time_channels
            self.time_emb = TimeEmbedding(args)
            if self.time_method == "as_input":
                self.time_projection = LinearTimeProjection(args)
            else:
                self.time_mlp = nn.Sequential(
                    nn.Linear(self.time_channels, self.time_channels),
                    nn.SiLU(),
                )
                width = self.hidden_dim if self.time_method == "per_layer" else 2 * self.hidden_dim
                self.time_heads = nn.ModuleList(
                    [nn.Linear(self.time_channels, width) for _ in range(self.num_layers)]
                )
                if self.time_method == "film":
                    # start at scale=0, shift=0 so conditioning begins as the identity
                    for head in self.time_heads:
                        nn.init.zeros_(head.weight)
                        nn.init.zeros_(head.bias)

        # layout kept identical to the original (Flatten, then Linear/act pairs) so
        # existing checkpoints still load under the default "none"
        in_dim = self.input_dim * 2 if self.time_method == "as_input" else self.input_dim
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Flatten())
        dims = [in_dim] + [self.hidden_dim] * self.num_layers
        for i in range(self.num_layers):
            self.encoder.append(nn.Linear(dims[i], dims[i + 1], bias=self.bias))
            self.encoder.append(self.nonlin)

        self.linear_mu = nn.Linear(dims[-1], self.latent_dim, bias=self.bias)
        self.linear_logvar = nn.Linear(dims[-1], self.latent_dim, bias=self.bias)

    def forward(self, x, t=None):
        if not self.time_conditioned:
            for layer in self.encoder:
                x = layer(x)
            return self.linear_mu(x), self.linear_logvar(x)

        if t is None:
            raise ValueError(
                "MLPEncoder was built with time_embedding_method_rec="
                f'"{self.time_method}" and requires a timestep t'
            )
        emb = self.time_emb(t.float())

        if self.time_method == "as_input":
            x = self.encoder[0](x)  # flatten
            x = torch.cat([x, self.time_projection(emb)], dim=1)
            for layer in self.encoder[1:]:
                x = layer(x)
            return self.linear_mu(x), self.linear_logvar(x)

        emb = self.time_mlp(emb)
        x = self.encoder[0](x)  # flatten
        i = 0
        for layer in self.encoder[1:]:
            if isinstance(layer, nn.Linear):
                x = layer(x)
                if self.time_method == "per_layer":
                    x = x + self.time_heads[i](emb)
                else:  # film
                    scale, shift = self.time_heads[i](emb).chunk(2, dim=1)
                    x = (1 + scale) * x + shift
                i += 1
            else:
                x = layer(x)
        return self.linear_mu(x), self.linear_logvar(x)

    def sample(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std



# ------------------------ conditional diffusion model ----------------------- #
class TimeEmbedding(nn.Module):
    """sinusoidal position embedding"""

    def __init__(self, args):
        super().__init__()
        self.dim = args.time_channels

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LinearTimeProjection(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.time_channels = args.time_channels
        self.input_dim = args.input_dim
        self.proj = nn.Sequential(
            nn.Linear(self.time_channels, self.input_dim),  # (B, 40) => (B, 40**2)
            # nn.ReLU(),
        )

    def forward(self, t):
        return self.proj(t)




class MLPDenoiser(nn.Module):
    """MLP denoiser with a selectable time-conditioning method.

    `args.time_embedding_method` picks how the sinusoidal time embedding reaches
    the network:
      - "as_input":  project the embedding down to input_dim and concatenate it
                     with x at the input layer (only entry point for time).
      - "per_layer": add a per-layer projection of the embedding to every hidden
                     pre-activation (DDPM ResBlock style).
      - "film":      per-layer scale/shift of every hidden pre-activation,
                     h <- (1 + scale) * h + shift. The heads are zero-initialised,
                     so the network starts as the identity conditioning
                     (scale=0, shift=0) and learns the modulation.
    """

    def __init__(self, args):
        super().__init__()

        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.bias = args.bias
        self.num_layers = args.num_layers
        self.time_channels = args.time_channels
        self.time_embedding_method = getattr(args, "time_embedding_method", "as_input")
        if self.time_embedding_method not in ("as_input", "per_layer", "film"):
            raise ValueError(
                f"Unknown time_embedding_method: {self.time_embedding_method}, "
                'expected "as_input", "per_layer" or "film"'
            )

        self.time_emb = TimeEmbedding(args)
        self.nonlin = nn.ReLU()
        self.flatten = nn.Flatten()

        if self.time_embedding_method == "as_input":
            # time enters only at the input, squashed to input_dim and concatenated
            self.time_projection = LinearTimeProjection(args)
            in_dim = self.input_dim * 2
        else:
            # shared trunk over the sinusoidal embedding, then one head per layer
            self.time_mlp = nn.Sequential(
                nn.Linear(self.time_channels, self.time_channels),
                nn.SiLU(),
            )
            out_per_layer = self.hidden_dim if self.time_embedding_method == "per_layer" else 2 * self.hidden_dim
            self.time_heads = nn.ModuleList(
                [nn.Linear(self.time_channels, out_per_layer) for _ in range(self.num_layers)]
            )
            if self.time_embedding_method == "film":
                # start at scale=0, shift=0 so conditioning begins as the identity
                for head in self.time_heads:
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)
            in_dim = self.input_dim

        dims = [in_dim] + [self.hidden_dim] * self.num_layers
        self.denoiser = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1], bias=self.bias) for i in range(self.num_layers)]
        )
        self.linear_out = nn.Linear(dims[-1], self.input_dim, bias=self.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = self.time_emb(t)
        x = self.flatten(x)

        if self.time_embedding_method == "as_input":
            x = torch.cat([x, self.time_projection(t)], dim=1)
            for layer in self.denoiser:
                x = self.nonlin(layer(x))
            return self.linear_out(x)

        t = self.time_mlp(t)
        for layer, head in zip(self.denoiser, self.time_heads):
            x = layer(x)
            if self.time_embedding_method == "per_layer":
                x = x + head(t)
            else:  # film
                scale, shift = head(t).chunk(2, dim=1)
                x = (1 + scale) * x + shift
            x = self.nonlin(x)
        return self.linear_out(x)