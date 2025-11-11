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

class AutoregressiveModel(torch.nn.Module):
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
        self.norm_layer = torch.nn.LayerNorm(d_latent)

        self.start_token = torch.nn.Parameter(torch.zeros(d_latent), requires_grad=True)

        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_latent,
                nhead=4,
                dim_feedforward=4 * d_latent,
                activation="gelu",
            ),
            num_layers=4,
        )
        self.output_projection1 = torch.nn.Linear(d_latent, d_latent)
        self.output_projection2 = torch.nn.Linear(d_latent, n_tokens)
        self.output_skip = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.
        """
        # if x.dim() == 4:
        B, h, w, c = x.shape
        seq_len = h * w * c
        x = x.reshape(B, seq_len)
        # elif x.dim() == 3:
        #     B, h, w = x.shape
        #     seq_len = h * w
        #     x = x.reshape(B, seq_len)
        # else:
        #     raise ValueError(f"Unexpected input shape: {x.shape}")

        x = x.long()  # Ensure indices are long type for embedding
        embedding = self.token_embedding(x)  # (B, h*w, d_latent)
        embedding = self.norm_layer(embedding)

        # start_tokens = start_token.expand
        start_tokens = self.start_token.unsqueeze(0).unsqueeze(0).expand(B, 1, self.d_latent)  # (B, 1, d_latent)
        shifted_input = torch.cat([start_tokens, embedding[:, :-1, :]], dim=1)  # (B, h*w, d_latent)

        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        output = self.transformer(shifted_input, mask=causal_mask)  # (B, h*w, d_latent)

        # Two layer projection with skip connection and GELU
        output_proj1 = self.output_projection1(output)  # (B, h*w, d_latent)
        output_proj2 = self.output_projection2(output_proj1)  # (B, h*w, n_tokens)
        output_skip = self.output_skip(output)  # (B, h*w, n_tokens)

        # Reshape logits from two layer projection
        logits = output_proj2 + output_skip  # (B, h*w, n_tokens)
        logits = logits.reshape(B, h, w, self.n_tokens)  # (B, h, w, n_tokens)

        return logits, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        # x = torch.zeros((B, h * w), dtype=torch.long, device=device)
        x = torch.zeros((B, h * w), dtype=torch.long, device=device)

        # curr_input = self.start_token.expand(B, 1, self.d_latent)
        input = self.start_token.unsqueeze(0).unsqueeze(0).expand(B, 1, self.d_latent)  # (B, 1, d_latent)

        for i in range(h * w):
            causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(i + 1).to(device)
            output = self.transformer(input, mask=causal_mask)  # (B, i+1, d_latent)
            output_proj1 = self.output_projection1(output)  # (B, i+1, d_latent)
            output_proj2 = self.output_projection2(output_proj1)  # (B, i+1, n_tokens)
            output_skip = self.output_skip(output)  # (B, i+1, n_tokens)

            logits = output_proj2[:, -1, :] + output_skip[:, -1, :]  # (B, n_tokens)
            next_token = torch.multinomial(logits.softmax(dim=-1), num_samples=1).squeeze(-1)  # (B,)
            x[:, i] = next_token

            next_embedded = self.token_embedding(next_token).unsqueeze(1)  # (B, 1, d_latent)
            input = torch.cat([input, next_embedded], dim=1)  # (B, i+2, d_latent)

        # for i in range(h * w):
        #     causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(i + 1).to(device)
        #    output = self.transformer(curr_input, mask=causal_mask, is_causal=True)  # (B, i+1, d_latent)
        #     logits = self.output_projection(output[:, -1, :])  # (B, n_tokens)
    
        # logits = functional.gelu
        # logits = output_proj2
        # logits += output_skip

        # functional.softmax(logits, dim=-1)  # (B, n_tokens
        # torch.multinomial(logits.softmax(dim=-1), num_samples=1).squeeze(-1)  # (B,)
        # generated[:, i] = next_token

        # next_embedded = self.token_embedding(next_token).unsqueeze(1)  # (B, 1, d_latent)
        # next_embedded = self.embed_norm(next_embedded)
        # torch.cat(current_input, next_embedded, dim=1)  # (B, i+2, d_latent
        
        return x.reshape(B, h, w)

        #     next_token = torch.multinomial(logits.softmax(dim=-1), num_samples=1).squeeze(-1)  # (B,)
        #     x[:, i] = next_token
        #     next_embedded = self.token_embedding(next_token).unsqueeze(1)  # (B, 1, d_latent)
        #     curr_input = torch.cat([curr_input, next_embedded], dim=1)  # (B, i+2, d_latent)  
        # return x.reshape(B, h, w)
