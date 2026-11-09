"""The line recognizer: CNN + BiLSTM + CTC head (a "CRNN").

Input:  a batch of grayscale line images, uint8 or float in [0, 255], shape
        (B, 1, 64, W). Images are given the normal way round; the model mirrors them
        itself so it reads right to left, like a person reading Arabic.
Output: log-probabilities, shape (T, B, num_classes), with T = W // 4. One row of
        scores per 4-pixel-wide slice of the image, in right-to-left reading order.

The three parts:
1. **CNN** (convolutional layers): finds visual patterns (strokes, dots, letter
   shapes). It shrinks the height from 64 to 1 and the width by 4, turning the image
   into a sequence of feature vectors, one per slice.
2. **BiLSTM** (bidirectional LSTM): reads that sequence in both directions so each
   slice "knows" its neighbours. Context matters in Arabic: a letter's shape depends
   on the letters around it, and a diacritic belongs to the letter under it.
3. **Linear head**: turns each slice's features into a score for every charset
   symbol plus the CTC blank.

Size: about 4.5 million weights, ~18 MB as float32 (PRD target: <= 20 MB).
"""

from __future__ import annotations

import itertools

import torch
from torch import nn

from arabic_ocr import charset

#: The model shrinks width by this factor: one output step per 4 input pixels.
WIDTH_STRIDE = 4
INPUT_HEIGHT = 64


def conv_block(c_in: int, c_out: int, kernel=(3, 3), padding=(1, 1)) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel, padding=padding, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class Recognizer(nn.Module):
    def __init__(
        self,
        num_classes: int = charset.NUM_CLASSES,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            conv_block(1, 32),
            nn.MaxPool2d(2, 2),  # 64 x W   -> 32 x W/2
            conv_block(32, 64),
            nn.MaxPool2d(2, 2),  # 32 x W/2 -> 16 x W/4
            conv_block(64, 128),
            conv_block(128, 128),
            nn.MaxPool2d((2, 1)),  # height 16 -> 8 (width kept: we need the steps)
            conv_block(128, 256),
            conv_block(256, 256),
            nn.MaxPool2d((2, 1)),  # 8 -> 4
            conv_block(256, 256),
            nn.MaxPool2d((2, 1)),  # 4 -> 2
            conv_block(256, 256, kernel=(2, 1), padding=(0, 0)),  # 2 -> 1
        )
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            256,
            rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.head = nn.Linear(2 * rnn_hidden, num_classes)

    def forward(self, images: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """`lengths` (optional): valid output steps per line, to ignore batch padding."""
        x = images.float() / 255.0
        x = (x - 0.5) / 0.5
        x = torch.flip(x, dims=[3])  # mirror: column 0 is now the right edge of the line
        f = self.cnn(x)  # (B, 256, 1, T)
        f = f.squeeze(2).permute(2, 0, 1)  # (T, B, 256)
        f = self.dropout(f)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(f, lengths.cpu(), enforce_sorted=False)
            out, _ = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, total_length=f.shape[0])
        else:
            out, _ = self.rnn(f)
        return self.head(self.dropout(out)).log_softmax(dim=2)


def output_length(width: int) -> int:
    """Number of output steps the model produces for an image of this width."""
    return width // WIDTH_STRIDE


def min_steps_for(target: list[int]) -> int:
    """Fewest output steps CTC needs for this target: one per symbol, plus a blank
    between each pair of equal neighbours (otherwise they would merge)."""
    repeats = sum(1 for a, b in itertools.pairwise(target) if a == b)
    return len(target) + repeats


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
