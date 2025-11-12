import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens
        self.token_embedding = torch.nn.Embedding(n_tokens, d_latent)
        
        ### The code below this line was written using Copilot (GPT-4.1)
        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model=d_latent, nhead=8)
        
        self.output_layer = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ### This code was written using Copilot (GPT-4.1)
        B, h, w = x.shape
        seq_len = h * w
        x = x.view(B, seq_len)
        x_embedded = self.token_embedding(x)
        x_embedded = torch.nn.functional.pad(x_embedded, (0, 0, 1, 0))[:, :-1]
        x_embedded = x_embedded.transpose(0, 1)
        
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)  # (h*w, h*w)
        
        transformer_output = self.transformer_layer(x_embedded, src_mask=causal_mask)
        transformer_output = transformer_output.transpose(0, 1)
        output_logits = self.output_layer(transformer_output)
        output_logits = output_logits.view(B, h, w, self.n_tokens)
        return output_logits, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        ### This code was written using Copilot (Claude Sonnet 4)
        self.eval()
        seq_len = h * w
        generated = torch.zeros(B, seq_len, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for i in range(seq_len):
                output_logits, _ = self.forward(generated.view(B, h, w))
                next_token_logits = output_logits[:, i // w, i % w, :]
                
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                generated[:, i] = next_token
        
        return generated.view(B, h, w)
