import abc

import torch

from .ae import PatchAutoEncoder


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self._codebook_bits = codebook_bits
        self.embedding_dim = embedding_dim

        self.down_proj = torch.nn.Linear(embedding_dim, codebook_bits)
        self.up_proj = torch.nn.Linear(codebook_bits, embedding_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down_proj(x)
        h = torch.nn.functional.normalize(h, p=2, dim=-1) # p=2, no eps
        codes = diff_sign(h)
        return codes

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * 2 ** torch.arange(x.size(-1)).to(x.device)).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self._codebook_bits).to(x.device))) > 0).float() - 1


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    def __init__(self, patch_size=5, latent_dim=128, codebook_bits=10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim)
        self._codebook_bits = codebook_bits
        self.bsq = BSQ(codebook_bits, latent_dim)

    def encode(self, x):
        z = PatchAutoEncoder.encode(self, x)  # latent embeddings (not quantized)
        return self.bsq.encode(z)             # apply BSQ quantization

    def decode(self, x):
        z = self.bsq.decode(x)              # decode BSQ quantized embeddings
        return PatchAutoEncoder.decode(self, z)  # decode the latent embeddings into an image

    def encode_index(self, x):
        z = PatchAutoEncoder.encode(self, x)  # latent embeddings (not quantized)
        codes = self.bsq.encode(z)     # apply BSQ quantization
        return self.bsq._code_to_index(codes)

    def decode_index(self, x):
        codes = self.bsq._index_to_code(x)
        z = self.bsq.decode(codes)
        return PatchAutoEncoder.decode(self, z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)

        cnt = torch.bincount(self.encode_index(x).flatten(), minlength=2**self._codebook_bits)
        codebook_usage = {
            "cb0": (cnt == 0).float().mean().detach(),
            "cb2": (cnt <= 2).float().mean().detach(),
        }

        return x_recon, codebook_usage
