"""使用预训练参数的 NumPy 图像编码器。
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """沿特征轴拆分注意力头，保持 token 顺序。"""
        batch_size, token_count, _ = x.shape
        return x.reshape(batch_size, token_count, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        # 线性层权重按 [输出维度, 输入维度] 存储，计算时转置。
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # 转置键矩阵，让每个查询与所有键计算点积。
        # 注意力分数的形状为 [B,H,N,N]。
        scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(np.float32(self.head_dim))

        # 图像编码默认关闭因果掩码。
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # 沿键所在的轴计算 softmax

        # 在各头内对值向量加权求和，再拼回特征轴。
        # 拼接前为 [B,H,N,64]，拼接后为 [B,N,D]。
        context = coefficients @ v
        batch_size, token_count, _ = x.shape
        merged = context.transpose(0, 2, 1, 3).reshape(batch_size, token_count, self.width)

        # 对拼接后的特征做输出投影。
        # 输出形状为 [B,N,D]。
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # 逐 token 做层归一化，再计算注意力更新。
        update = self.attention(self.norm1(x), causal=causal)

        # 将注意力更新加回块的输入。
        residual = x + update

        # 对第一次残差相加的结果做第二次层归一化。
        u = self.norm2(residual)

        # 计算 MLP 更新量，再加回残差。
        # 图像模型的 MLP 先扩展到 3072 维，再投影回 768 维。
        # 两次残差相加均保持 [B,N,D]。
        hidden = self.gelu(u @ self.fc1_weight.T + self.fc1_bias)
        mlp_update = hidden @ self.fc2_weight.T + self.fc2_bias
        return residual + mlp_update


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512]，按 [输入维度, 输出维度] 存储

    def patchify(self, images):
        """按从左到右、从上到下的顺序展开 32×32 的 RGB 图像块。"""
        if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
            raise ValueError(f"Expected image shape (B, 3, 224, 224), got {images.shape}")

        # 把网格轴移到前面，使每块按通道、高度、宽度展开。
        batch_size = images.shape[0]
        grid = images.reshape(batch_size, 3, 7, 32, 7, 32)
        patches = grid.transpose(0, 2, 4, 1, 3, 5)
        return patches.reshape(batch_size, 49, 3072)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # 对图像块进行无偏置的线性投影。
        patch_tokens = patches @ self.patch_weight.T

        # 为每张图复制一个 CLS token。
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # 将 CLS 拼到序列开头，再把位置编码广播到整个批次。
        # 输出形状为 [B,50,768]。
        return np.concatenate((cls, patch_tokens), axis=1) + self.position

    def encode_tokens(self, images):
        """构造输入序列并依次通过 12 个 Transformer 块。"""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """为每张图返回一个单位长度的 512 维特征向量。"""
        tokens = self.encode_tokens(images)

        # 取出 CLS token，做层归一化后投影。
        # 形状依次为 [B,50,768]、[B,768] 和 [B,512]。
        # 最后将投影结果归一化为单位向量。
        cls_token = tokens[:, 0, :]
        output = self.post_norm(cls_token) @ self.projection

        return unit_vector(output)
