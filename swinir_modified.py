import torch
import torch.nn as nn

class SwinIR(nn.Module):
    def __init__(self, num_stages=5, num_blocks_per_stage=6, window_size=8, embedding_dim=96):
        super(SwinIR, self).__init__()

        self.stages = nn.ModuleList()
        for i in range(num_stages):
            stage = SwinTransformer(num_blocks=num_blocks_per_stage, window_size=window_size, embedding_dim=embedding_dim)
            self.stages.append(stage)

        self.reconstruction_block = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(embedding_dim, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)

        x = self.reconstruction_block(x)

        return x

class SwinTransformer(nn.Module):
    def __init__(self, num_blocks, window_size=8, embedding_dim=96):
        super(SwinTransformer, self).__init__()

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = SwinTransformerBlock(window_size=window_size, embedding_dim=embedding_dim)
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, window_size=8, embedding_dim=96):
        super(SwinTransformerBlock, self).__init__()

        self.self_attention = nn.MultiheadAttention(embedding_dim, 8, batch_first=True)
        self.window_shift_attention = nn.WindowShiftAttention(window_size=window_size, embedding_dim=embedding_dim)
        self.residual_connection = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.self_attention(x, x, x)
        x = x + x
        x = self.window_shift_attention(x, x, x)
        x = x + x
        x = self.residual_connection(x)
        x = x + x

        return x

