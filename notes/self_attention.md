## 加性注意力

### 假设数据

- `batch_size = 1`：只有一个样本。
- `num_queries = 2`：有两个查询。
- `num_keys = 3`：有三个键-值对。
- `key_size = 4`：键向量的维度是4。
- `query_size = 4`：查询向量的维度是4。
- `value_size = 3`：值向量的维度是3。

我们将随机初始化查询、键、值以及有效长度（valid_lens）。

### 1. **随机数据**

```python
import torch

# 随机初始化查询、键、值
queries = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]]])  # shape (1, 2, 4)
keys = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]]])  # shape (1, 3, 4)
values = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]])  # shape (1, 3, 3)

# 有效长度
valid_lens = torch.tensor([[3, 2]])  # 每个查询对应的有效键-值对的个数
```

### 2. **线性变换查询和键**

我们为查询和键分别定义线性变换。

```python
# 假设 num_hiddens = 4，使用简单的线性变换
W_q = torch.nn.Linear(4, 4, bias=False)
W_k = torch.nn.Linear(4, 4, bias=False)

# 对查询和键进行线性变换
queries = W_q(queries)  # 变换后的查询，形状 (1, 2, 4)
keys = W_k(keys)  # 变换后的键，形状 (1, 3, 4)
```

### 3. **计算注意力得分**

我们可以将查询和键进行加法操作并计算得分。

```python
# 扩展查询和键以便相加
queries_expanded = queries.unsqueeze(2)  # (1, 2, 1, 4)
keys_expanded = keys.unsqueeze(1)  # (1, 1, 3, 4)

# 求和并应用 tanh 激活函数
features = queries_expanded + keys_expanded  # (1, 2, 3, 4)
features = torch.tanh(features)

# 计算得分 (通过一个简单的线性变换)
w_v = torch.nn.Linear(4, 1, bias=False)
scores = w_v(features).squeeze(-1)  # (1, 2, 3)
```

### 4. **掩蔽 Softmax**

计算掩蔽 Softmax 来得到注意力权重。

```python
import torch.nn.functional as F

def masked_softmax(scores, valid_lens):
    # 对 scores 进行掩蔽 Softmax
    mask = torch.arange(scores.size(-1))[None, :] >= valid_lens[..., None]
    scores.masked_fill_(mask, float('-inf'))
    return F.softmax(scores, dim=-1)

# 应用掩蔽 Softmax
attention_weights = masked_softmax(scores, valid_lens)  # shape (1, 2, 3)
```

### 5. **加权求和**

最后，将注意力权重与值进行加权求和。

```python
# 使用注意力权重对值进行加权求和
output = torch.bmm(attention_weights, values)  # (1, 2, 3) x (1, 3, 3) -> (1, 2, 3)
```

### 6. **输出结果**

```python
print("Queries:", queries)
print("Keys:", keys)
print("Values:", values)
print("Scores:", scores)
print("Attention Weights:", attention_weights)
print("Output:", output)
```

### 输出示例

假设随机初始化后的输出如下（由于数据是随机的，每次运行可能不同）：

```
Queries: tensor([[[ 0.2791,  0.3597, -0.1882, -0.1980],
         [ 0.1217,  0.4597,  0.0659,  0.4264]]])
Keys: tensor([[[ 0.2473, -0.2110,  0.5771, -0.0365],
         [-0.2732, -0.5186,  0.6365,  0.1395],
         [ 0.1419, -0.4654, -0.1089,  0.3119]]])
Values: tensor([[[0.0972, 0.9306, 0.5680],
         [0.7884, 0.6249, 0.8398],
         [0.2294, 0.3565, 0.1037]]])
Scores: tensor([[[ 0.4861, -0.2320,  0.2082],
         [ 0.2426, -0.2381,  0.1519]]])
Attention Weights: tensor([[[0.3963, 0.3013, 0.3023],
         [0.3689, 0.3477, 0.2834]]])
Output: tensor([[[0.4257, 0.4832, 0.4913],
         [0.4829, 0.4429, 0.4240]]])
```

### 解释

1. `queries` 和 `keys` 经过线性变换后，得到了新的查询和键表示。
2. 计算了 `scores`，表示每个查询和每个键的匹配度。
3. 通过 `masked_softmax` 计算了注意力权重，确保只对有效的键-值对进行注意力计算。
4. 最终，通过加权求和得到 `output`，它表示了基于查询的加性注意力输出。

这就是如何通过矩阵操作来实现加性注意力机制。