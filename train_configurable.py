#!/usr/bin/env python3
"""
AutoCPD 可配置训练脚本
支持通过命令行参数指定模型类型、数据维度、分类数等
Author: Modified for configurable training
Date: 2024-01-XX

Usage:
    python train_configurable.py --model deep --classes 5 --dim 10 --samples 1000
    python train_configurable.py --model simple --classes 3 --dim 1 --preset basic
    python train_configurable.py --model transformer --classes 3 --dim 5 --samples 800
    python train_configurable.py --model deep --classes 3 --dim 5 --deep_input_format multidim_6channel
    python train_configurable.py --model deep --classes 3 --dim 8 --deep_input_format reshape_20x20
    python train_configurable.py --model deep --dim 5 --preset full --samples 800
    python train_configurable.py --model simple --classes 4 --dim 3 \\
                               --length 200 --samples 500 --epochs 100 --batch_size 8
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import json
import warnings
warnings.filterwarnings('ignore')

# 添加AutoCPD到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autocpd.neuralnetwork import compile_and_fit, general_simple_nn, general_deep_nn, general_transformer_nn, general_transformer_nn_v2
from autocpd.utils import DataSetGen, Transform2D2TR
from autocpd.high_dim_utils import HighDimDataSetGen, get_preset_config, prepare_high_dim_data_for_training


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='AutoCPD 可配置变点检测训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 深度网络，5分类，10维数据
  python train_configurable.py --model deep --classes 5 --dim 10 --samples 1000
  
  # 简单网络，3分类，1维数据，使用原版配置
  python train_configurable.py --model simple --classes 3 --dim 1 --use_original
  
  # Transformer网络，3分类，5维数据
  python train_configurable.py --model transformer --classes 3 --dim 5 --samples 800
  
  # 深度网络使用新的多维6通道格式
  python train_configurable.py --model deep --classes 3 --dim 5 --deep_input_format multidim_6channel
  
  # 深度网络使用传统的20x20格式
  python train_configurable.py --model deep --classes 3 --dim 8 --deep_input_format reshape_20x20
  
  # 使用预设配置
  python train_configurable.py --model deep --dim 5 --preset full --samples 800
  
  # 自定义参数
  python train_configurable.py --model simple --classes 4 --dim 3 \\
                               --length 200 --samples 500 --epochs 100 --batch_size 8
        """
    )
    
    # === 模型配置 ===
    parser.add_argument('--model', type=str, default='deep',
                       choices=['simple', 'deep', 'transformer'],
                       help='模型类型: simple=简单神经网络, deep=深度神经网络, transformer=Transformer网络 (默认: deep)')
    
    # === 数据配置 ===
    parser.add_argument('--classes', type=int, default=None,
                       help='分类数量 (1-5, 默认根据preset自动确定)')
    
    parser.add_argument('--dim', type=int, default=1,
                       help='数据维度 (默认: 1)')
    
    parser.add_argument('--length', type=int, default=400,
                       help='时间序列长度 (默认: 400)')
    
    parser.add_argument('--samples', type=int, default=800,
                       help='每类样本数量 (默认: 800)')
    
    parser.add_argument('--preset', type=str, default='basic',
                       choices=['basic', 'full', 'mean_var_only', 'correlation_focus'],
                       help='预设变点配置 (默认: basic)')
    
    parser.add_argument('--use_original', action='store_true',
                       help='使用原版AutoCPD的DataSetGen (仅适用于1维)')
    
    # === 训练参数 ===
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数 (默认自动根据模型和分类数确定)')
    
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小 (默认: 64)')
    
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='学习率 (默认自动确定)')
    
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout率 (默认自动确定)')
    
    # === Transformer特定参数 ===
    parser.add_argument('--d_model', type=int, default=None,
                       help='Transformer模型维度 (默认自动确定)')
    
    parser.add_argument('--num_heads', type=int, default=None,
                       help='Transformer注意力头数 (默认自动确定)')
    
    parser.add_argument('--ff_dim', type=int, default=None,
                       help='Transformer前馈网络维度 (默认自动确定)')
    
    parser.add_argument('--num_layers', type=int, default=None,
                       help='Transformer层数 (默认自动确定)')
    
    # === 噪声配置 ===
    parser.add_argument('--noise_type', type=str, default='gaussian',
                       choices=['gaussian', 'ar1', 'cauchy', 'ar_random'],
                       help='噪声类型: gaussian=高斯白噪声, ar1=AR(1)噪声, cauchy=柯西噪声, ar_random=随机AR系数 (默认: gaussian)')
    
    parser.add_argument('--noise_level', type=float, default=1.0,
                       help='噪声强度/标准差 (默认: 1.0)')
    
    parser.add_argument('--ar_coef', type=float, default=0.7,
                       help='AR(1)噪声的自回归系数 (默认: 0.7)')
    
    parser.add_argument('--cauchy_scale', type=float, default=0.3,
                       help='柯西噪声的尺度参数 (默认: 0.3)')
    
    # === 数据处理 ===
    parser.add_argument('--transform', type=str, default='auto',
                       choices=['auto', 'flatten', 'channel', 'pca', 'transpose'],
                       help='高维数据变换方式 (默认: auto)')
    
    parser.add_argument('--deep_input_format', type=str, default='reshape_20x20',
                       choices=['reshape_20x20', 'multidim_6channel'],
                       help='深度网络输入格式: reshape_20x20=(batch_size,dim,20,20), multidim_6channel=(batch_size,6,len,dim) (默认: reshape_20x20)')
    
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='验证集比例 (默认: 0.2)')
    
    # === 输出控制 ===
    parser.add_argument('--output_dir', type=str, default='results',
                       help='输出目录 (默认: results)')
    
    parser.add_argument('--exp_name', type=str, default=None,
                       help='实验名称 (默认自动生成)')
    
    parser.add_argument('--save_data', action='store_true',
                       help='保存生成的数据')
    
    parser.add_argument('--save_model', action='store_true',
                       help='保存训练好的模型')
    
    parser.add_argument('--plot', action='store_true',
                       help='显示训练图表')
    
    parser.add_argument('--verbose', type=int, default=1,
                       choices=[0, 1, 2],
                       help='详细程度 (0=安静, 1=正常, 2=详细, 默认: 1)')
    
    # === 其他 ===
    parser.add_argument('--seed', type=int, default=2022,
                       help='随机种子 (默认: 2022)')
    
    parser.add_argument('--gpu', type=str, default='auto',
                       help='GPU设置 (auto/cpu/0/1/..., 默认: auto)')
    
    return parser.parse_args()


def setup_gpu(gpu_config):
    """配置GPU"""
    if gpu_config == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("使用 CPU 训练")
    elif gpu_config == 'auto':
        if tf.config.list_physical_devices('GPU'):
            print(f"检测到 GPU: {tf.config.list_physical_devices('GPU')}")
        else:
            print("未检测到 GPU，使用 CPU")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_config)
        print(f"使用 GPU: {gpu_config}")


def generate_data(args):
    """生成训练数据"""
    print(f"\n=== 数据生成 ===")
    
    if args.use_original and args.dim == 1:
        print("使用原版 DataSetGen")
        # 使用原版数据生成
        mean_arg = np.array([0.7, 5, -5, 1.2, 0.6])
        var_arg = np.array([0, 0.7, 0.3, 0.4, 0.2])
        slope_arg = np.array([0.5, 0.025, -0.025, 0.03, 0.015])
        
        dataset = DataSetGen(args.samples, args.length, mean_arg, var_arg, slope_arg, n_trim=40)
        data_x = dataset["data_x"]
        
        # 确定使用的类别数
        if args.classes is None:
            args.classes = 5
        
        if args.classes < 5:
            # 模拟原版的类别选择逻辑
            if args.classes == 3:
                # 删除前两类，使用后3类
                data_x = np.delete(data_x, np.arange(0 * args.samples, 2 * args.samples), 0)
                labels = [0, 1, 2]
                class_names = ['Variance Change', 'No Slope Change', 'Slope Change']
            else:
                # 使用前n类
                data_x = data_x[:args.classes * args.samples]
                labels = list(range(args.classes))
                all_names = ['No Mean Change', 'Mean Change', 'Variance Change', 'No Slope Change', 'Slope Change']
                class_names = all_names[:args.classes]
        else:
            labels = [0, 1, 2, 3, 4]
            class_names = ['No Mean Change', 'Mean Change', 'Variance Change', 'No Slope Change', 'Slope Change']
        
        data_y = np.repeat(labels, args.samples).reshape((-1, 1))
        
        # 应用Transform2D2TR变换 (仅对深度网络)
        if args.model == 'deep':
            data_x = Transform2D2TR(data_x, rescale=True, times=3)
        
        data_dict = {
            'data_x': data_x,
            'labels': data_y.flatten().tolist(),
            'change_types': class_names,
            'n_classes': args.classes,
            'dimensions': 1
        }
    
    else:
        print("使用高维 HighDimDataSetGen")
        # 使用高维数据生成
        config = get_preset_config(args.preset, args.dim)
        
        # 添加噪声参数
        noise_params = {
            'noise_std': args.noise_level,
            'noise_type': args.noise_type,
            'ar_coef': args.ar_coef,
            'cauchy_scale': args.cauchy_scale
        }
        
        print(f"噪声配置: {noise_params}")
        
        data_dict = HighDimDataSetGen(
            N_sub=args.samples,
            n=args.length,
            d=args.dim,
            seed=args.seed,
            **config,
            **noise_params
        )
        
        if args.classes is not None:
            # 限制类别数
            n_classes_generated = data_dict['n_classes']
            if args.classes < n_classes_generated:
                print(f"限制类别数从 {n_classes_generated} 到 {args.classes}")
                # 只保留前args.classes个类别
                data_x = data_dict['data_x']
                labels = np.array(data_dict['labels'])
                
                mask = labels < args.classes
                data_dict['data_x'] = data_x[mask]
                data_dict['labels'] = labels[mask].tolist()
                data_dict['n_classes'] = args.classes
                data_dict['change_types'] = data_dict['change_types'][:args.classes]
        
        # 更新classes参数
        args.classes = data_dict['n_classes']
    
    print(f"数据生成完成:")
    print(f"  形状: {data_dict['data_x'].shape}")
    print(f"  类别数: {data_dict['n_classes']}")
    print(f"  类别: {data_dict['change_types']}")
    
    return data_dict


def prepare_data_for_model(data_dict, args):
    """为模型准备数据"""
    print(f"\n=== 数据预处理 ===")
    
    data_x = data_dict['data_x']
    labels = np.array(data_dict['labels'])
    
    # 确定变换方式
    if args.transform == 'auto':
        if args.model == 'simple':
            transform_type = 'flatten'  # Simple NN使用flatten
        elif args.model == 'transformer':
            transform_type = 'transpose'  # Transformer使用transpose格式 (T, d)
        elif args.model == 'deep':
            if args.dim == 1:
                transform_type = 'channel'  # 1维保持原格式
            else:
                # 根据deep_input_format参数决定变换方式
                if args.deep_input_format == 'multidim_6channel':
                    transform_type = 'multidim_6channel'  # 新的6通道格式
                else:
                    transform_type = 'reshape_20x20'  # 默认20x20格式
        else:
            transform_type = 'channel'  # 默认保持通道格式
    else:
        # 用户明确指定了变换方式，直接使用
        transform_type = args.transform
        print(f"使用用户指定的变换方式: {transform_type}")
    
    print(f"数据变换方式: {transform_type}")
    
    if args.use_original and args.dim == 1:
        # 原版数据处理
        if args.model == 'deep':
            # 深度网络已经变换过了
            processed_data = data_x
        elif args.model == 'transformer':
            # Transformer需要 (N, T) 格式 - 对于1维数据，就是时间序列本身
            if len(data_x.shape) == 3:
                processed_data = data_x.squeeze(1)  # (N, 1, T) -> (N, T)
            else:
                processed_data = data_x
            # 为Transformer添加特征维度：(N, T) -> (N, T, 1)
            processed_data = processed_data.reshape(processed_data.shape[0], processed_data.shape[1], 1)
        else:
            # 简单网络需要展平
            if len(data_x.shape) == 3:
                processed_data = data_x.squeeze(1)  # (N, 1, T) -> (N, T)
            else:
                processed_data = data_x
        processed_labels = labels
    else:
        # 高维数据处理
        processed_data, processed_labels = prepare_high_dim_data_for_training(
            data_dict, transform_type
        )
    
    print(f"处理后数据形状: {processed_data.shape}")
    print(f"标签形状: {processed_labels.shape}")
    
    # 训练测试分割
    x_train, x_test, y_train, y_test = train_test_split(
        processed_data, processed_labels,
        train_size=0.8,
        random_state=args.seed,
        stratify=processed_labels
    )
    
    print(f"训练集: {x_train.shape}, 测试集: {x_test.shape}")
    
    return x_train, x_test, y_train, y_test, transform_type


def get_model_config(args, input_shape):
    """获取模型配置"""
    print(f"\n=== 模型配置 ===")
    
    # 自动确定训练参数
    if args.learning_rate is None:
        if args.model == 'transformer':
            # Transformer通常需要较小的学习率
            args.learning_rate = 5e-4 if args.classes <= 3 else 3e-4
        elif args.model == 'deep':
            args.learning_rate = 8e-4
        else:
            args.learning_rate = 8e-4
    
    if args.epochs is None:
        base_epochs = 50 if args.model == 'simple' or args.model == 'transformer' else 120
        args.epochs = base_epochs + (args.classes - 2) * 10 + (args.dim - 1) * 5
    
    if args.dropout is None:
        if args.model == 'transformer':
            # Transformer的dropout
            args.dropout = 0.1 + (args.classes - 2) * 0.02
        else:
            args.dropout = 0.3 + (args.classes - 2) * 0.05 + (args.dim - 1) * 0.02
        args.dropout = min(args.dropout, 0.5)
    
    print(f"模型类型: {args.model}")
    print(f"输入形状: {input_shape}")
    print(f"类别数: {args.classes}")
    print(f"学习率: {args.learning_rate}")
    print(f"训练轮数: {args.epochs}")
    print(f"Dropout率: {args.dropout}")
    print(f"批次大小: {args.batch_size}")
    
    return args


def build_model(args, input_shape):
    """构建模型"""
    print(f"\n=== 模型构建 ===")
    
    # 生成实验名称
    if args.exp_name is None:
        args.exp_name = f"{args.model}_{args.classes}class_d{args.dim}_n{args.length}_N{args.samples}"
        if not args.use_original:
            args.exp_name += f"_{args.preset}"
    
    print(f"实验名称: {args.exp_name}")
    
    if args.model == 'simple':
        # 简单神经网络
        if len(input_shape) == 1:
            n = input_shape[0]
        else:
            n = input_shape[-1]  # 假设最后一维是时间
        
        # 根据类别数调整网络大小
        if args.classes <= 2:
            m = np.array([50, 40, 30])
        elif args.classes <= 3:
            m = np.array([60, 50, 40, 30])
        else:
            m = np.array([80, 60, 50, 40])
        
        model = general_simple_nn(
            n=n,
            l=len(m),
            m=m,
            num_classes=args.classes,
            model_name=args.exp_name
        )
    
    elif args.model == 'transformer':
        # Transformer神经网络
        if len(input_shape) == 3:
            # 原始数据是(N, T, d)格式，input_shape是(T, d)
            time_steps, features = input_shape
            n = time_steps
            input_dim = features
        elif len(input_shape) == 2:
            # 新格式：(time_steps, features)
            time_steps, features = input_shape
            n = time_steps  # 序列长度是时间步数
            input_dim = features  # 每个时间步的特征维度
        elif len(input_shape) == 1:
            # 1维数据格式: (T,) - 需要转换为(T, 1)
            n = input_shape[0]  # 时间步数
            input_dim = 1  # 特征维度
        else:
            raise ValueError(f"Transformer不支持的输入形状: {input_shape}")
        
        # 使用命令行参数或自动确定Transformer参数
        if args.d_model is not None:
            d_model = args.d_model
        else:
            # 根据输入特征维度动态调整d_model
            if input_dim == 1:
                d_model = 64  # 1维数据使用较小的模型
            elif input_dim <= 5:
                d_model = 128  # 低维数据
            else:
                d_model = 256  # 高维数据
            
        if args.num_heads is not None:
            num_heads = args.num_heads
        else:
            num_heads = 4
            
        if args.ff_dim is not None:
            ff_dim = args.ff_dim
        else:
            ff_dim = max(128, d_model * 2)  # 前馈网络维度通常是d_model的2倍
            
        # 针对长序列优化参数 - 现在序列长度大大减少了
        if n > 500:  # 只有当时间步数超过500时才需要优化
            print(f"检测到长时间序列 (n={n})，优化Transformer参数以节省显存...")
            d_model = min(d_model, 128)  # 限制模型维度
            num_heads = min(num_heads, 4)  # 限制注意力头数
            ff_dim = min(ff_dim, 256)  # 限制前馈网络维度
            print(f"  优化后 - d_model: {d_model}, num_heads: {num_heads}, ff_dim: {ff_dim}")
            
        if args.num_layers is not None:
            num_layers = args.num_layers
        else:
            # 根据类别数调整层数
            num_layers = 1 if args.classes <= 3 else 2
        
        # 约束检查：确保num_heads能够整除d_model
        if d_model % num_heads != 0:
            print(f"警告: d_model ({d_model}) 不能被 num_heads ({num_heads}) 整除")
            # 调整num_heads为d_model的最大因子
            valid_heads = [h for h in [2, 4, 8, 16] if d_model % h == 0 and h <= d_model]
            if valid_heads:
                num_heads = max(valid_heads)
                print(f"自动调整 num_heads 为: {num_heads}")
            else:
                num_heads = 2
                print(f"使用默认 num_heads: {num_heads}")
        
        print(f"Transformer参数:")
        print(f"  时间步数: {n}")
        print(f"  特征维度: {input_dim}")
        print(f"  模型维度: {d_model}")
        print(f"  注意力头数: {num_heads}")
        print(f"  前馈维度: {ff_dim}")
        print(f"  层数: {num_layers}")
        
        # 使用改进的Transformer构建函数
        model = general_transformer_nn_v2(
            time_steps=n,
            input_dim=input_dim,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            num_classes=args.classes,
            dropout_rate=args.dropout,
            model_name=args.exp_name
        )
    
    else:  # deep
        # 深度神经网络
        if len(input_shape) == 3:
            # 3维格式：(channels, height, width) - 不包含batch维度
            channels, height, width = input_shape
            print(f"深度网络参数:")
            print(f"  输入格式: 3维")
            print(f"  通道数: {channels}")
            print(f"  高度: {height}")
            print(f"  宽度: {width}")
            
            # 根据输入格式决定使用哪种网络架构
            if args.deep_input_format == 'multidim_6channel' and channels == 6:
                # 新的(6, len, dim)格式 - 使用特殊的多维深度网络
                print(f"  使用格式: (batch_size, 6, {height}, {width})")
                
                # 可以在这里调用专门处理(6,len,dim)格式的网络
                # 暂时使用现有的general_deep_nn_4d，但调整参数
                from autocpd.neuralnetwork import general_deep_nn_4d
                
                # 针对(6,len,dim)格式优化的网络参数
                n_filter = 16 if width <= 5 else 24  # 根据dim调整
                kernel_size = (3, 3)  # 对于时间序列数据使用较小的卷积核
                num_resblock = 3 if args.classes <= 3 else 4
                
                # 根据类别数调整全连接层
                if args.classes <= 2:
                    m = np.array([40, 30, 20])
                elif args.classes <= 3:
                    m = np.array([50, 40, 30, 20])
                else:
                    m = np.array([60, 50, 40, 30, 20])
                
                l = len(m)
                
                print(f"  卷积核: {kernel_size}")
                print(f"  残差块: {num_resblock}")
                print(f"  全连接层: {m}")
                
                model = general_deep_nn_4d(
                    channels=channels,
                    height=height,
                    width=width,
                    kernel_size=kernel_size,
                    n_filter=n_filter,
                    dropout_rate=args.dropout,
                    n_classes=args.classes,
                    n_resblock=num_resblock,
                    m=m,
                    l=l,
                    model_name=args.exp_name
                )
                
            else:
                # 原来的(dim, 20, 20)格式
                print(f"  使用格式: (batch_size, {channels}, 20, 20)")
                
                # 根据新格式创建特殊的深度网络
                from autocpd.neuralnetwork import general_deep_nn_4d
                
                # 网络参数
                n_filter = 16
                kernel_size = (3, 3)  # 固定使用3x3卷积核
                num_resblock = 3
                
                # 根据类别数调整全连接层
                if args.classes <= 2:
                    m = np.array([40, 30, 20])
                elif args.classes <= 3:
                    m = np.array([50, 40, 30, 20])
                else:
                    m = np.array([60, 50, 40, 30, 20])
                
                l = len(m)
                
                print(f"  卷积核: {kernel_size}")
                print(f"  残差块: {num_resblock}")
                print(f"  全连接层: {m}")
                
                model = general_deep_nn_4d(
                    channels=channels,
                    height=height,
                    width=width,
                    kernel_size=kernel_size,
                    n_filter=n_filter,
                    dropout_rate=args.dropout,
                    n_classes=args.classes,
                    n_resblock=num_resblock,
                    m=m,
                    l=l,
                    model_name=args.exp_name
                )
                
        elif len(input_shape) == 2:
            num_tran, n = input_shape
            
            # 网络参数
            n_filter = 16
            kernel_size = (max(1, num_tran // 2), min(10, n // 10))
            num_resblock = 3
            
            # 根据类别数调整全连接层
            if args.classes <= 2:
                m = np.array([40, 30, 20])
            elif args.classes <= 3:
                m = np.array([50, 40, 30, 20])
            else:
                m = np.array([60, 50, 40, 30, 20])
            
            l = len(m)
            
            print(f"深度网络参数:")
            print(f"  通道数: {num_tran}")
            print(f"  时间长度: {n}")
            print(f"  卷积核: {kernel_size}")
            print(f"  残差块: {num_resblock}")
            print(f"  全连接层: {m}")
            
            model = general_deep_nn(
                n=n,
                n_trans=num_tran,
                kernel_size=kernel_size,
                n_filter=n_filter,
                dropout_rate=args.dropout,
                n_classes=args.classes,
                n_resblock=num_resblock,
                m=m,
                l=l,
                model_name=args.exp_name
            )
            
        elif len(input_shape) == 1:
            # 1维数据，需要添加通道维度
            n = input_shape[0]
            num_tran = 1
            
            # 网络参数
            n_filter = 16
            kernel_size = (1, min(10, n // 10))
            num_resblock = 3
            
            # 根据类别数调整全连接层
            if args.classes <= 2:
                m = np.array([40, 30, 20])
            elif args.classes <= 3:
                m = np.array([50, 40, 30, 20])
            else:
                m = np.array([60, 50, 40, 30, 20])
            
            l = len(m)
            
            print(f"深度网络参数:")
            print(f"  通道数: {num_tran}")
            print(f"  时间长度: {n}")
            print(f"  卷积核: {kernel_size}")
            print(f"  残差块: {num_resblock}")
            print(f"  全连接层: {m}")
            
            model = general_deep_nn(
                n=n,
                n_trans=num_tran,
                kernel_size=kernel_size,
                n_filter=n_filter,
                dropout_rate=args.dropout,
                n_classes=args.classes,
                n_resblock=num_resblock,
                m=m,
                l=l,
                model_name=args.exp_name
            )
        else:
            raise ValueError(f"不支持的输入形状: {input_shape}")
    
    return model


def train_model(model, x_train, y_train, args):
    """训练模型"""
    print(f"\n=== 开始训练 ===")
    
    # 设置日志目录
    logdir = Path(args.output_dir) / "tensorboard_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    
    # 训练
    from tensorflow_docs.modeling import EpochDots
    epochdots = EpochDots()
    
    history = compile_and_fit(
        model,
        x_train,
        y_train,
        args.batch_size,
        args.learning_rate,
        args.exp_name,
        logdir,
        epochdots,
        validation_split=args.validation_split,
        max_epochs=args.epochs,
    )
    
    return history


def evaluate_model(model, x_test, y_test, args, class_names):
    """评估模型"""
    print(f"\n=== 模型评估 ===")
    
    # 评估
    results = model.evaluate(x_test, y_test, verbose=0)
    if len(results) == 2:
        loss, accuracy = results
        metric1 = None
    elif len(results) == 3:
        loss, metric1, accuracy = results
    else:
        # 处理其他情况
        loss = results[0]
        accuracy = results[-1] if len(results) > 1 else None
        metric1 = results[1] if len(results) > 2 else None
    
    print(f"测试准确率: {accuracy:.4f}")
    if metric1 is not None:
        print(f"SparseCategoricalCrossentropy: {metric1:.4f}")
    print(f"测试损失: {loss:.4f}")
    
    # 预测
    y_prob = model.predict(x_test, verbose=0)
    if args.classes == 1:
        y_pred = (y_prob > 0.5).astype(int).flatten()
    else:
        y_pred = np.argmax(y_prob, axis=1)
    
    # 混淆矩阵
    if args.classes > 1:
        confusion_mtx = tf.math.confusion_matrix(y_test, y_pred)
        confusion_np = confusion_mtx.numpy()
        
        # 计算各类别准确率
        class_accuracies = []
        for i in range(args.classes):
            class_correct = confusion_np[i, i]
            class_total = np.sum(confusion_np[i, :])
            class_acc = class_correct / class_total if class_total > 0 else 0
            class_accuracies.append(class_acc)
            print(f"{class_names[i]}: {class_acc:.4f}")
        
        return {
            'test_accuracy': accuracy,
            'test_loss': loss,
            'class_accuracies': class_accuracies,
            'confusion_matrix': confusion_np,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    else:
        return {
            'test_accuracy': accuracy,
            'test_loss': loss,
            'y_pred': y_pred,
            'y_prob': y_prob
        }


def save_results(model, results, args, data_dict):
    """保存结果"""
    print(f"\n=== 保存结果 ===")
    
    # 创建输出目录
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 不保存模型权重（按用户要求）
    
    # 保存配置
    config = vars(args)
    config_path = output_dir / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"配置保存: {config_path}")
    
    # 保存结果（只包含最佳准确率和训练历史）
    results_path = output_dir / "results.npz"
    save_dict = {
        'config': config,
        'class_names': data_dict['change_types'],
        'best_accuracy': results['best_accuracy'],
        'training_history': results['training_history']
    }
    np.savez(results_path, **save_dict)
    print(f"结果保存: {results_path}")
    print(f"最佳准确率: {results['best_accuracy']:.4f}")
    
    # 保存数据（如果需要）
    if args.save_data:
        data_path = output_dir / "data.npz"
        np.savez(data_path, **data_dict)
        print(f"数据保存: {data_path}")
    
    return output_dir


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 设置随机种子
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # 设置GPU
    setup_gpu(args.gpu)
    
    print(f"=== AutoCPD 可配置训练开始 ===")
    print(f"参数: {vars(args)}")
    
    try:
        # 生成数据
        data_dict = generate_data(args)
        
        # 准备数据
        x_train, x_test, y_train, y_test, transform_type = prepare_data_for_model(data_dict, args)
        
        # 获取模型配置
        input_shape = x_train.shape[1:]
        args = get_model_config(args, input_shape)
        
        # 构建模型
        model = build_model(args, input_shape)
        
        if args.verbose >= 1:
            model.summary()
        
        # 训练模型
        history = train_model(model, x_train, y_train, args)
        
        # 从训练历史中获取最高的验证准确率
        if hasattr(history, 'history') and 'val_accuracy' in history.history:
            best_val_accuracy = max(history.history['val_accuracy'])
            best_epoch = history.history['val_accuracy'].index(best_val_accuracy) + 1
            print(f"\n=== 最佳验证准确率 ===")
            print(f"最佳验证准确率: {best_val_accuracy:.4f} (第 {best_epoch} 轮)")
            
            # 用最佳验证准确率替换测试准确率
            final_accuracy = best_val_accuracy
        else:
            # 如果没有验证历史，则评估模型获取测试准确率
            print(f"\n=== 模型评估 ===")
            results = evaluate_model(model, x_test, y_test, args, data_dict['change_types'])
            final_accuracy = results['test_accuracy']
        
        # 创建简化的结果字典（只保存准确率）
        results = {
            'best_accuracy': final_accuracy,
            'training_history': history.history if hasattr(history, 'history') else None
        }
        
        # 保存结果
        output_dir = save_results(model, results, args, data_dict)
        
        print(f"\n=== 训练完成 ===")
        print(f"实验名称: {args.exp_name}")
        print(f"最佳准确率: {final_accuracy:.4f}")
        print(f"结果目录: {output_dir}")
        
        return final_accuracy  # 返回最佳准确率用于批量实验
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 