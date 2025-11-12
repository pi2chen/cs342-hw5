from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image

from .autoregressive import Autoregressive
from .bsq import Tokenizer


class Compressor:
    def __init__(self, tokenizer: Tokenizer, autoregressive: Autoregressive):
        super().__init__()
        self.tokenizer = tokenizer
        self.autoregressive = autoregressive

    def compress(self, x: torch.Tensor) -> bytes:
        """
        Compress the image into a torch.uint8 bytes stream (1D tensor).

        Use arithmetic coding.
        """
        import struct
        
        # Input x has shape (H, W, C), we need to add batch dimension
        # Tokenize the image (H, W, C) -> (1, h, w) -> (h, w)
        with torch.no_grad():
            # Add batch dimension and tokenize
            x_batch = x.unsqueeze(0)  # (1, H, W, C)
            tokens = self.tokenizer.encode_index(x_batch)  # (1, h, w)
            
            # Remove batch dimension since we're processing a single image
            tokens = tokens.squeeze(0)  # (h, w)
            h, w = tokens.shape
            
            # Flatten tokens for sequential processing
            tokens_flat = tokens.view(-1)  # (h*w,)
            seq_len = h * w
            
            # Simple arithmetic coding implementation
            # Initialize range [0, 1) scaled to integer arithmetic
            precision = 32  # Use 32-bit precision
            range_size = 2**precision
            low = 0
            high = range_size - 1
            
            # Store compressed bits
            compressed_bits = []
            
            # For proper arithmetic coding, we need probabilities for each position
            current_tokens = torch.zeros_like(tokens)  # (h, w)
            
            for pos in range(seq_len):
                # Reshape current tokens to (1, h, w) for autoregressive model
                current_batch = current_tokens.unsqueeze(0)  # (1, h, w)
                
                # Get probabilities from autoregressive model
                with torch.no_grad():
                    logits, _ = self.autoregressive.forward(current_batch)
                    # logits shape: (1, h, w, n_tokens)
                    pos_h, pos_w = pos // w, pos % w
                    token_logits = logits[0, pos_h, pos_w, :]  # (n_tokens,)
                    probs = torch.softmax(token_logits, dim=0)
                
                # Get actual token at this position
                actual_token = tokens_flat[pos].item()
                
                # Convert probabilities to cumulative distribution
                probs_np = probs.cpu().numpy()
                cumulative_probs = np.cumsum(probs_np)
                cumulative_probs = np.concatenate([[0], cumulative_probs])
                
                # Scale to integer range
                cumulative_scaled = (cumulative_probs * (high - low + 1)).astype(np.int64)
                
                # Update range based on symbol probability
                symbol_low = low + cumulative_scaled[actual_token]
                symbol_high = low + cumulative_scaled[actual_token + 1] - 1
                
                # Emit bits when possible
                while True:
                    if symbol_high < range_size // 2:
                        # Both in lower half
                        compressed_bits.append(0)
                        symbol_low *= 2
                        symbol_high = symbol_high * 2 + 1
                    elif symbol_low >= range_size // 2:
                        # Both in upper half
                        compressed_bits.append(1)
                        symbol_low = (symbol_low - range_size // 2) * 2
                        symbol_high = (symbol_high - range_size // 2) * 2 + 1
                    else:
                        break
                
                low, high = symbol_low, symbol_high
                
                # Update current tokens for next iteration
                current_tokens[pos_h, pos_w] = actual_token
            
            # Final bits to distinguish the range
            if low < range_size // 4:
                compressed_bits.append(0)
            else:
                compressed_bits.append(1)
            
            # Convert bits to bytes
            # Pad to byte boundary
            while len(compressed_bits) % 8 != 0:
                compressed_bits.append(0)
            
            # Convert to bytes
            byte_array = []
            for i in range(0, len(compressed_bits), 8):
                byte_val = 0
                for j in range(8):
                    if i + j < len(compressed_bits):
                        byte_val |= (compressed_bits[i + j] << (7 - j))
                byte_array.append(byte_val)
            
            compressed_bytes = bytes(byte_array)
            
            # Create header with dimensions and bit length
            header = struct.pack('III', h, w, len(compressed_bits))
            result = header + compressed_bytes
            
            return result

    def decompress(self, x: bytes) -> torch.Tensor:
        """
        Decompress a tensor into a PIL image.
        You may assume the output image is 150 x 100 pixels.
        """
        import struct
        
        # Parse header to get dimensions and bit length
        header_size = 3 * 4  # 3 uint32 values for h, w, num_bits
        h, w, num_bits = struct.unpack('III', x[:header_size])
        data_bytes = x[header_size:]
        
        # Convert bytes back to bits
        compressed_bits = []
        for byte_val in data_bytes:
            for j in range(8):
                compressed_bits.append((byte_val >> (7 - j)) & 1)
        
        # Take only the actual compressed bits (remove padding)
        compressed_bits = compressed_bits[:num_bits]
        
        seq_len = h * w
        
        # Initialize arithmetic decoder
        precision = 32
        range_size = 2**precision
        
        # Read initial value from bit stream
        value = 0
        bit_index = 0
        for i in range(min(precision, len(compressed_bits))):
            if bit_index < len(compressed_bits):
                value = (value << 1) | compressed_bits[bit_index]
                bit_index += 1
            else:
                value = value << 1
        
        # Decode tokens
        decoded_tokens = torch.zeros(h, w, dtype=torch.long)
        device = next(self.tokenizer.parameters()).device
        decoded_tokens = decoded_tokens.to(device)
        
        low = 0
        high = range_size - 1
        
        for pos in range(seq_len):
            pos_h, pos_w = pos // w, pos % w
            
            # Get probabilities from autoregressive model
            current_batch = decoded_tokens.unsqueeze(0)  # (1, h, w)
            
            with torch.no_grad():
                logits, _ = self.autoregressive.forward(current_batch)
                # logits shape: (1, h, w, n_tokens)
                token_logits = logits[0, pos_h, pos_w, :]  # (n_tokens,)
                probs = torch.softmax(token_logits, dim=0)
            
            # Convert probabilities to cumulative distribution
            probs_np = probs.cpu().numpy()
            cumulative_probs = np.cumsum(probs_np)
            cumulative_probs = np.concatenate([[0], cumulative_probs])
            
            # Scale to current range
            range_width = high - low + 1
            cumulative_scaled = (cumulative_probs * range_width).astype(np.int64)
            
            # Find which symbol the current value corresponds to
            target = value - low
            symbol = 0
            for i in range(len(cumulative_scaled) - 1):
                if cumulative_scaled[i] <= target < cumulative_scaled[i + 1]:
                    symbol = i
                    break
            
            # Update range for this symbol
            symbol_low = low + cumulative_scaled[symbol]
            symbol_high = low + cumulative_scaled[symbol + 1] - 1
            
            # Remove bits from value as we narrow the range
            while True:
                if symbol_high < range_size // 2:
                    # Both in lower half
                    symbol_low *= 2
                    symbol_high = symbol_high * 2 + 1
                    value = value * 2
                    if bit_index < len(compressed_bits):
                        value |= compressed_bits[bit_index]
                        bit_index += 1
                elif symbol_low >= range_size // 2:
                    # Both in upper half  
                    symbol_low = (symbol_low - range_size // 2) * 2
                    symbol_high = (symbol_high - range_size // 2) * 2 + 1
                    value = (value - range_size // 2) * 2
                    if bit_index < len(compressed_bits):
                        value |= compressed_bits[bit_index]
                        bit_index += 1
                else:
                    break
                
                # Keep value in bounds
                value = value & (range_size - 1)
            
            low, high = symbol_low, symbol_high
            
            # Store decoded symbol
            decoded_tokens[pos_h, pos_w] = symbol
        
        # Decode tokens back to image
        with torch.no_grad():
            # Add batch dimension for tokenizer
            tokens_batch = decoded_tokens.unsqueeze(0)  # (1, h, w)
            
            # Decode tokens to image
            decoded_image = self.tokenizer.decode_index(tokens_batch)  # (1, H, W, C)
            
            # Remove batch dimension
            decoded_image = decoded_image.squeeze(0)  # (H, W, C)
            
            return decoded_image


def compress(tokenizer: Path, autoregressive: Path, image: Path, compressed_image: Path):
    """
    Compress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    images: Path to the image to compress.
    compressed_image: Path to save the compressed image tensor.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    x = torch.tensor(np.array(Image.open(image)), dtype=torch.uint8, device=device)
    cmp_img = cmp.compress(x.float() / 255.0 - 0.5)
    with open(compressed_image, "wb") as f:
        f.write(cmp_img)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    """
    Decompress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    compressed_image: Path to the compressed image tensor.
    images: Path to save the image to compress.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    with open(compressed_image, "rb") as f:
        cmp_img = f.read()

    x = cmp.decompress(cmp_img)
    img = Image.fromarray(((x + 0.5) * 255.0).clamp(min=0, max=255).byte().cpu().numpy())
    img.save(image)


if __name__ == "__main__":
    from fire import Fire

    Fire({"compress": compress, "decompress": decompress})
