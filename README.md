# AutoCPD: 自动变点检测深度学习框架

AutoCPD 是一个基于深度学习的自动变点检测框架，支持多种神经网络架构和高维时间序列数据处理。本项目提供了灵活的训练配置和批量实验功能，适用于学习和研究变点检测算法。

## 📋 目录

- [项目特性](#项目特性)
- [安装说明](#安装说明)
- [快速开始](#快速开始)
- [详细使用说明](#详细使用说明)
- [参数配置](#参数配置)
- [批量实验](#批量实验)
- [项目结构](#项目结构)
- [实验示例](#实验示例)

## 🚀 项目特性

### 支持的模型类型
- **Simple NN**: 简单的多层感知机网络
- **Deep NN**: 深度卷积神经网络，支持残差块
- **Transformer**: 基于注意力机制的Transformer网络

### 数据处理能力
- 支持1维到高维时间序列数据
- 多种数据变换方式：flatten、channel、PCA、transpose
- 多种噪声类型：高斯、AR(1)、柯西、随机AR
- 灵活的变点配置预设

### 实验功能
- 单个实验训练和评估
- 批量实验比较
- 自动超参数调整
- 结果可视化和报告生成

## 💾 安装说明

### 1. 克隆项目
```bash
git clone https://github.com/DS-Zeng/Biostat-PJ-AutoCPD
cd AutoCPD
```

### 2. 安装依赖
使用开发模式安装：
```bash
pip install -e .
```

### 3. 验证安装
```bash
python -c "import autocpd; print('AutoCPD安装成功！')"
```

## 🏃 快速开始

### 基础实验示例

```bash
# 简单神经网络，3分类，1维数据
python train_configurable.py --model simple --classes 3 --dim 1

# 深度神经网络，5分类，5维数据
python train_configurable.py --model deep --classes 5 --dim 5

# Transformer网络，3分类，5维数据
python train_configurable.py --model transformer --classes 3 --dim 5
```

### 批量实验
```bash
# 运行系统比较实验
python batch_experiment.py
```

## 📖 详细使用说明

### train_configurable.py - 单个实验训练

这是主要的训练脚本，支持灵活的参数配置。

#### 基本用法
```bash
python train_configurable.py [选项]
```

#### 常用实验配置

**1. 简单神经网络实验**
```bash
# 基础配置
python train_configurable.py \
    --model simple \
    --classes 3 \
    --dim 1 \
    --samples 800 \
    --length 400

# 使用原版数据生成器（仅1维）
python train_configurable.py \
    --model simple \
    --classes 3 \
    --dim 1 \
    --use_original
```

**2. 深度神经网络实验**
```bash
# 使用多维6通道格式
python train_configurable.py \
    --model deep \
    --classes 3 \
    --dim 5 \
    --deep_input_format multidim_6channel

# 使用传统20x20格式
python train_configurable.py \
    --model deep \
    --classes 3 \
    --dim 8 \
    --deep_input_format reshape_20x20
```

**3. Transformer网络实验**
```bash
# 基础Transformer
python train_configurable.py \
    --model transformer \
    --classes 3 \
    --dim 5 \
    --samples 800

# 自定义Transformer参数
python train_configurable.py \
    --model transformer \
    --classes 5 \
    --dim 10 \
    --d_model 128 \
    --num_heads 8 \
    --ff_dim 256 \
    --num_layers 2
```

**4. 高级配置示例**
```bash
# 完整参数配置
python train_configurable.py \
    --model transformer \
    --classes 4 \
    --dim 3 \
    --length 200 \
    --samples 500 \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --dropout 0.2 \
    --noise_type ar1 \
    --noise_level 1.5 \
    --preset full \
    --validation_split 0.3 \
    --exp_name my_experiment \
    --save_model \
    --plot
```

## ⚙️ 参数配置

### 模型参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `deep` | 模型类型: simple/deep/transformer |
| `--classes` | int | 自动 | 分类数量 (1-5) |
| `--dim` | int | `1` | 数据维度 |
| `--length` | int | `400` | 时间序列长度 |
| `--samples` | int | `800` | 每类样本数量 |

### 训练参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--epochs` | int | 自动 | 训练轮数 |
| `--batch_size` | int | `64` | 批次大小 |
| `--learning_rate` | float | 自动 | 学习率 |
| `--dropout` | float | 自动 | Dropout率 |
| `--validation_split` | float | `0.2` | 验证集比例 |

### Transformer特定参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--d_model` | int | 自动 | 模型维度 |
| `--num_heads` | int | 自动 | 注意力头数 |
| `--ff_dim` | int | 自动 | 前馈网络维度 |
| `--num_layers` | int | 自动 | Transformer层数 |

### 数据处理参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--transform` | str | `auto` | 数据变换: auto/flatten/channel/pca/transpose |
| `--deep_input_format` | str | `reshape_20x20` | 深度网络格式: reshape_20x20/multidim_6channel |
| `--preset` | str | `basic` | 预设配置: basic/full/mean_var_only/correlation_focus |
| `--use_original` | bool | False | 使用原版数据生成器（仅1维） |

### 噪声参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--noise_type` | str | `gaussian` | 噪声类型: gaussian/ar1/cauchy/ar_random |
| `--noise_level` | float | `1.0` | 噪声强度 |
| `--ar_coef` | float | `0.7` | AR(1)系数 |
| `--cauchy_scale` | float | `0.3` | 柯西噪声尺度 |

## 🔬 批量实验

### batch_experiment.py - 系统比较实验

批量实验脚本用于系统比较不同模型在各种配置下的表现。

#### 运行批量实验
```bash
python batch_experiment.py
```

#### 实验矩阵
默认实验配置：
- **模型**: Simple NN, Deep NN, Transformer
- **分类数**: 3, 5
- **数据维度**: 1, 5, 8
- **总实验数**: 3 × 2 × 3 = 18个实验

#### 自定义批量实验
可以修改 `batch_experiment.py` 中的配置：

```python
# 修改实验参数
base_args = {
    'samples': 1000,      # 样本数量
    'length': 300,        # 序列长度
    'batch_size': 32,     # 批次大小
    'epochs': 50,         # 训练轮数
    'seed': 2024          # 随机种子
}

# 修改实验矩阵
models = ['simple', 'deep', 'transformer']
class_numbers = [2, 3, 4, 5]
dimensions = [1, 3, 5, 8, 10]
```

#### 实验结果
批量实验将生成：
- `experiment_results.csv`: 详细实验结果
- `experiment_summary.txt`: 文字摘要报告
- `experiment_overview.png`: 总览图表
- `accuracy_heatmap.png`: 准确率热力图
- `accuracy_ranking.png`: 准确率排行榜

## 📁 项目结构

```
AutoCPD/
├── src/autocpd/              # 核心代码包
│   ├── neuralnetwork.py      # 神经网络模型
│   ├── utils.py              # 基础工具函数
│   └── high_dim_utils.py     # 高维数据工具
├── train_configurable.py    # 单个实验训练脚本
├── batch_experiment.py      # 批量实验脚本
├── transformer_hyperparameter_search.py  # 超参数搜索
├── data/                    # 数据目录
├── results/                 # 单个实验结果
├── batch_experiment_results/ # 批量实验结果
├── docs/                    # 文档
├── test/                    # 测试文件
├── pyproject.toml           # 项目配置
└── README.md               # 本文档
```

## 🧪 实验示例

### 示例1: 比较不同模型在3分类任务上的表现

```bash
# Simple NN
python train_configurable.py --model simple --classes 3 --dim 5 --exp_name "simple_3class_5d"

# Deep NN  
python train_configurable.py --model deep --classes 3 --dim 5 --deep_input_format multidim_6channel --exp_name "deep_3class_5d"

# Transformer
python train_configurable.py --model transformer --classes 3 --dim 5 --exp_name "transformer_3class_5d"
```

### 示例2: 噪声鲁棒性测试

```bash
# 测试不同噪声类型
python train_configurable.py --model transformer --classes 3 --dim 5 --noise_type gaussian --noise_level 1.0
python train_configurable.py --model transformer --classes 3 --dim 5 --noise_type ar1 --noise_level 1.5
python train_configurable.py --model transformer --classes 3 --dim 5 --noise_type cauchy --cauchy_scale 0.5
```

### 示例3: 维度扩展性测试

```bash
# 测试不同维度
for dim in 1 3 5 8 10; do
    python train_configurable.py --model transformer --classes 3 --dim $dim --exp_name "transformer_dim${dim}"
done
```

### 示例4: 超参数搜索

```bash
# Transformer超参数搜索
python transformer_hyperparameter_search.py --classes 3 --dim 5 --samples 800
```

## 📊 结果解读

### 输出文件说明
- `config.json`: 实验配置参数
- `results.npz`: 训练结果和最佳准确率
- TensorBoard日志: `tensorboard_logs/` 目录

### 性能指标
- **最佳准确率**: 训练过程中验证集的最高准确率
- **训练时长**: 完整训练所需时间
- **各类别准确率**: 每个变点类型的分类准确率

### 模型选择建议
- **简单任务**: Simple NN (快速、稳定)
- **复杂模式**: Deep NN (特征提取能力强)  
- **长序列**: Transformer (注意力机制)
- **高维数据**: Deep NN with multidim_6channel

## 🔧 故障排除

### 常见问题

**1. GPU内存不足**
```bash
# 减小批次大小
python train_configurable.py --model transformer --batch_size 16

# 使用CPU训练
python train_configurable.py --gpu cpu
```

**2. 序列过长导致内存溢出**
```bash
# 减少序列长度
python train_configurable.py --model transformer --length 200

# 使用PCA降维
python train_configurable.py --model transformer --transform pca
```

**3. 收敛困难**
```bash
# 调整学习率
python train_configurable.py --learning_rate 0.0001

# 增加训练轮数
python train_configurable.py --epochs 200
```

## 📄 许可证

本项目遵循MIT许可证 - 详见 [LICENSE.txt](LICENSE.txt) 文件。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

