import sys

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline, set_seed

from transformers.models.gpt2.modeling_gpt2 import eager_attention_forward
import torch.nn.functional as F
import torch

import math

def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        ))

def mlp(x, block):
    fc_w = block.mlp.c_fc.weight  # [C, 4C]
    fc_b = block.mlp.c_fc.bias  # [4C]
    x = x @ fc_w + fc_b  # [B, T, 4C]

    # 2. 激活
    x = gelu_new(x)

    # 3. 第二层 c_proj
    proj_w = block.mlp.c_proj.weight  # [4C, C]
    proj_b = block.mlp.c_proj.bias  # [C]
    x = x @ proj_w + proj_b  # [B, T, C]

    # 4. dropout（推理模式下无作用）
    x = block.mlp.dropout(x)

    return x

def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    norm_x = (x - mean) / torch.sqrt(var + eps)
    print("norm shape:", norm_x.shape)
    print("weight shape:", weight.shape)
    print("mul shape:", (norm_x * weight).shape)
    return norm_x * weight + bias

def attention(x, block):
    B, T, C = x.shape
    n_head = block.attn.num_heads # 12
    head_dim = C // n_head # 64

    # 1. QKV 投影（直接用 c_attn 保证一致）
    # qkv = block.attn.c_attn(x)   # [B, T, 3C]
    qkv = x @ block.attn.c_attn.weight
    print("x max min:", x.max(), x.min())
    print("w max min:", block.attn.c_attn.weight.max(), block.attn.c_attn.weight.min())
    print("qkv max min:", qkv.max(), qkv.min())
    qkv = qkv + block.attn.c_attn.bias
    q, k, v = qkv.split(C, dim=-1) # [B, T, C]


    # 2. 拆分 heads
    def split_heads(t):
        return t.view(B, T, n_head, head_dim).transpose(1, 2)  # [B, nh, T, hd]

    q = split_heads(q)  # [B, n_head, T, head_dim]
    k = split_heads(k)
    v = split_heads(v)

    # 3. 注意力打分
    att = torch.matmul(q, k.transpose(-1, -2))

    att = att / math.sqrt(head_dim)

    # 4. 因果 Mask（用加法，而不是 masked_fill）
    causal_mask = block.attn.bias[:, :, :T, :T].to(torch.bool)
    mask_value = torch.full([], -256.0, dtype=att.dtype, device=att.device)
    att = torch.where(causal_mask, att, mask_value)

    # 5. softmax + dropout
    att = torch.softmax(att, dim=-1)
    att = block.attn.attn_dropout(att)

    # 6. 加权求和
    y = torch.matmul(att, v)

    # 7. 合并 heads
    y = y.transpose(1, 2).contiguous().view(B, T, C)

    # 8. 输出投影 + resid dropout
    y = block.attn.c_proj(y)
    y = block.attn.resid_dropout(y)
    return y

def block_manual(hidden, block, ln1_weight, ln1_bias, ln2_weight, ln2_bias):
    # 1. LayerNorm 1
    h_ln1 = layer_norm(hidden,
                       ln1_weight,
                       ln1_bias,
                       eps=block.ln_1.eps)
    print("input shape:", hidden.shape)
    print("weight shape:", ln1_weight.shape)
    print("bias shape:", ln1_bias.shape)
    print("h_ln1 shape:", h_ln1.shape)
    # 2. Attention
    attn_out = attention(h_ln1, block)

    # 3. 残差连接
    hidden_resid1 = hidden + attn_out

    # 4. LayerNorm 2
    h_ln2 = layer_norm(hidden_resid1,
                       ln2_weight,
                       ln2_bias,
                       eps=block.ln_2.eps)

    # 5. MLP
    mlp_out = mlp(h_ln2, block)

    # 6. 残差连接
    hidden_out = hidden_resid1 + mlp_out

    return hidden_out

model_path = "/data/dj/gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()
model.config._attn_implementation = "eager"

# 2. 准备输入
prompt = "Hello, I'm a language model,"
inputs = tokenizer(prompt, return_tensors="pt")  # 转成张量 (batch_size=1)

input_ids = inputs["input_ids"] # (1, 8)
pos_ids = torch.arange(0, input_ids.size(1)).unsqueeze(0) # (1, 8)
official_wte_out = model.transformer.wte(input_ids)
official_wpe_out = model.transformer.wpe(pos_ids)

wte_weight = model.state_dict()["transformer.wte.weight"]
wpe_weight = model.state_dict()["transformer.wpe.weight"]
manual_wte_out = wte_weight[input_ids] # 查表, (8, 786)
manual_wpe_out = wpe_weight[pos_ids]   # 查表, (8, 786)

hidden = manual_wte_out + manual_wpe_out

ln1_weight = model.state_dict()["transformer.h.0.ln_1.weight"]
ln2_weight = model.state_dict()["transformer.h.0.ln_2.weight"]
ln1_bias   = model.state_dict()["transformer.h.0.ln_1.bias"]
ln2_bias   = model.state_dict()["transformer.h.0.ln_2.bias"]

torch.set_printoptions(precision=10, sci_mode=False)  # 精度10位，不用科学计数法

for i, block in enumerate(model.transformer.h):
    # print(f"Layer {i}:", i)
    ln1_weight = block.ln_1.weight
    ln1_bias   = block.ln_1.bias
    ln2_weight = block.ln_2.weight
    ln2_bias   = block.ln_2.bias
    print("shape:", ln1_weight.shape, ln1_bias.shape, ln2_weight.shape, ln2_bias.shape)
    hidden = block_manual(hidden, block,
                                 ln1_weight, ln1_bias,
                                 ln2_weight, ln2_bias)
    print("hidden:", hidden[0][0][0])
    break

    # sys.exit(-1)
    # hidden_output = block(hidden)[0]


    # print("output1:", hidden_output)
    # print("output2:", hidden_manual)
    # print("Max diff:", ((hidden_manual - hidden_output).abs().max()).item())
    # print("allclose output layer ", i, ":", torch.allclose(hidden_output, hidden_manual, atol=1e-5, rtol=1e-5))

    # if i == 11:
    #     print("output1:", hidden_output)
    #     print("output2:", hidden_manual)

# print("input_ids:", input_ids)
# print("pos_ids:", pos_ids)
#
# print("hidden shape:", hidden.shape)
# print("hidden:", hidden)
#
# hidden = model.transformer.ln_f(hidden)
# print("Final hidden:", hidden.shape)
#
# # 4. logits
# logits = model.lm_head(hidden)
# print("Logits shape:", logits.shape)
# print("Logits:", logits[0][0][0])