"""
Author         : AI Assistant
Date           : 2024-01-XX
Description    : 测试深度网络在多维数据上的表现
                输入格式: (batch_size, 6, len, dim)
                其中: 6=变换数量, len=400(时间序列长度), dim=数据维度

实验设置:
- 分类数: 3, 5
- 维度: 1, 5, 8
- 网络: 深度CNN
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_docs.modeling as tfdoc_model
import tensorflow_docs.plots as tfdoc_plot
from sklearn.model_selection import train_test_split
from datetime import datetime
import pandas as pd
import sys

# 添加AutoCPD到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autocpd.neuralnetwork import compile_and_fit
from autocpd.utils import DataSetGen, Transform2D2TR
from autocpd.high_dim_utils import HighDimDataSetGen, get_preset_config
from tensorflow.keras import layers, models


def create_multidim_deep_network(input_shape, n_classes, n_filter=16, dropout_rate=0.3, 
                                n_resblock=3, model_name="multidim_deep_nn"):
    """
    创建适用于(6, len, dim)输入格式的深度网络
    
    Parameters:
    -----------
    input_shape : tuple
        输入形状 (6, len, dim)
    n_classes : int
        分类数量
    n_filter : int
        卷积滤波器数量
    dropout_rate : float
        Dropout率
    n_resblock : int
        残差块数量
    model_name : str
        模型名称
    """
    n_trans, length, dim = input_shape
    
    # 输入层
    input_layer = layers.Input(shape=input_shape, name="Input")
    
    # 初始卷积层
    x = layers.Conv2D(n_filter, (2, 2), padding="same")(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # 残差块
    for i in range(n_resblock):
        residual = x
        
        # 第一个卷积
        x = layers.Conv2D(n_filter, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # 第二个卷积
        x = layers.Conv2D(n_filter, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        
        # 残差连接
        if residual.shape[-1] != x.shape[-1]:
            residual = layers.Conv2D(n_filter, (1, 1), padding="same")(residual)
        x = layers.Add()([x, residual])
        x = layers.ReLU()(x)
        
        # 适度降采样
        if i % 2 == 1:
            x = layers.MaxPooling2D((1, 2))(x)
    
    # 全局平均池化
    x = layers.GlobalAveragePooling2D()(x)
    
    # 全连接层
    fc_sizes = [80, 60, 40, 30] if n_classes >= 5 else [60, 40, 30]
    
    for size in fc_sizes:
        x = layers.Dense(size, activation="relu", kernel_regularizer="l2")(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # 输出层
    output_layer = layers.Dense(n_classes)(x)
    
    model = models.Model(input_layer, output_layer, name=model_name)
    return model


def transform_to_multidim_format(data_x, dim):
    """
    将数据转换为(N, 6, len, dim)格式
    
    Parameters:
    -----------
    data_x : np.ndarray
        输入数据
    dim : int
        目标维度
        
    Returns:
    --------
    np.ndarray
        转换后的数据 (N, 6, len, dim)
    """
    if dim == 1:
        # 1维数据：使用Transform2D2TR，然后扩展维度
        if len(data_x.shape) == 2:
            # (N, len) -> 通过Transform2D2TR -> (N, 6, len) -> (N, 6, len, 1)
            transformed = Transform2D2TR(data_x, rescale=True, times=3)
            return np.expand_dims(transformed, axis=-1)
        else:
            # 如果已经是3维
            return np.expand_dims(data_x, axis=-1)
    else:
        # 多维数据：模拟6种变换
        N, d, length = data_x.shape
        result = np.zeros((N, 6, length, dim))
        
        # 6种变换：
        # 1. 原始数据
        result[:, 0, :, :] = data_x.transpose(0, 2, 1)  # (N, d, len) -> (N, len, d)
        
        # 2. 平方数据
        result[:, 1, :, :] = np.square(data_x).transpose(0, 2, 1)
        
        # 3. 立方数据
        result[:, 2, :, :] = np.power(data_x, 3).transpose(0, 2, 1)
        
        # 4. 移动平均 (窗口=3)
        for i in range(N):
            for j in range(d):
                smoothed = np.convolve(data_x[i, j], np.ones(3)/3, mode='same')
                result[i, 3, :, j] = smoothed
        
        # 5. 差分数据
        diff_data = np.diff(data_x, axis=2, prepend=data_x[:, :, :1])
        result[:, 4, :, :] = diff_data.transpose(0, 2, 1)
        
        # 6. 累积和数据
        cumsum_data = np.cumsum(data_x, axis=2)
        result[:, 5, :, :] = cumsum_data.transpose(0, 2, 1)
        
        return result


def generate_experiment_data(samples_per_class, length, dim, classes, preset='basic'):
    """
    生成实验数据
    
    Parameters:
    -----------
    samples_per_class : int
        每类样本数
    length : int
        时间序列长度
    dim : int
        数据维度
    classes : int
        分类数量
    preset : str
        预设配置
        
    Returns:
    --------
    tuple
        (data_x, data_y, class_names)
    """
    print(f"\n生成数据: {classes}分类, {dim}维, 长度{length}, 每类{samples_per_class}样本")
    
    if dim == 1:
        # 1维数据：使用原版DataSetGen
        mean_arg = np.array([0.7, 5, -5, 1.2, 0.6])
        var_arg = np.array([0, 0.7, 0.3, 0.4, 0.2])
        slope_arg = np.array([0.5, 0.025, -0.025, 0.03, 0.015])
        
        dataset = DataSetGen(samples_per_class, length, mean_arg, var_arg, slope_arg, n_trim=20)
        data_x = dataset["data_x"]
        
        # 选择指定数量的类别
        if classes == 3:
            # 删除前两类，使用后3类
            data_x = np.delete(data_x, np.arange(0, 2 * samples_per_class), 0)
            labels = [0, 1, 2]
            class_names = ['Variance Change', 'No Slope Change', 'Slope Change']
        elif classes == 5:
            labels = [0, 1, 2, 3, 4]
            class_names = ['No Mean Change', 'Mean Change', 'Variance Change', 'No Slope Change', 'Slope Change']
        else:
            # 使用前classes类
            data_x = data_x[:classes * samples_per_class]
            labels = list(range(classes))
            all_names = ['No Mean Change', 'Mean Change', 'Variance Change', 'No Slope Change', 'Slope Change']
            class_names = all_names[:classes]
        
        data_y = np.repeat(labels, samples_per_class)
        
    else:
        # 多维数据：使用HighDimDataSetGen
        config = get_preset_config(preset, d=dim)
        
        # 根据classes调整配置
        if classes == 3:
            # 只使用3种变点类型
            config['correlation_changes']['enabled'] = False  # 关闭相关性变化
            config['structural_changes']['enabled'] = False  # 关闭结构变化
            class_names = ['No Change', 'Mean Change', 'Variance Change']
        elif classes == 5:
            # 使用所有5种类型
            class_names = ['No Change', 'Mean Change', 'Variance Change', 'Correlation Change', 'Trend Change']
        
        # 生成数据
        dataset_dict = HighDimDataSetGen(
            N_sub=samples_per_class,
            n=length,  # 直接使用目标长度
            d=dim,
            mean_changes=config['mean_changes'],
            var_changes=config['var_changes'],
            correlation_changes=config['correlation_changes'],
            trend_changes=config['trend_changes'],
            structural_changes=config['structural_changes'],
            n_trim=0,  # 不进行trim
            noise_std=1.0,
            seed=2022
        )
        
        data_x = dataset_dict['data_x']
        data_y = np.array(dataset_dict['labels'])
        class_names = ['No Change'] + [name.replace('_', ' ').title() for name in dataset_dict['change_types'][1:]]
    
    return data_x, data_y, class_names


def run_experiment(classes, dim, samples_per_class=800, length=400, epochs=80, batch_size=64):
    """
    运行单个实验
    
    Parameters:
    -----------
    classes : int
        分类数量
    dim : int
        数据维度
    samples_per_class : int
        每类样本数
    length : int
        时间序列长度
    epochs : int
        训练轮数
    batch_size : int
        批次大小
        
    Returns:
    --------
    dict
        实验结果
    """
    print(f"\n{'='*60}")
    print(f"实验: {classes}分类, {dim}维度")
    print(f"{'='*60}")
    
    # 生成数据
    data_x, data_y, class_names = generate_experiment_data(
        samples_per_class, length, dim, classes
    )
    
    # 转换为多维格式
    data_x_transformed = transform_to_multidim_format(data_x, dim)
    print(f"转换后数据形状: {data_x_transformed.shape}")
    
    # 数据分割
    x_train, x_test, y_train, y_test = train_test_split(
        data_x_transformed, data_y, 
        train_size=0.8, 
        random_state=42,
        stratify=data_y
    )
    
    print(f"训练集: {x_train.shape}, 测试集: {x_test.shape}")
    
    # 创建模型
    input_shape = x_train.shape[1:]  # (6, length, dim)
    model_name = f"deep_multidim_{classes}c_{dim}d"
    
    # 根据复杂度调整参数
    n_filter = 16 if dim <= 5 else 24
    dropout_rate = 0.3 + (classes - 3) * 0.05 + (dim - 1) * 0.02
    dropout_rate = min(dropout_rate, 0.5)
    n_resblock = 3 if classes <= 3 else 4
    
    model = create_multidim_deep_network(
        input_shape=input_shape,
        n_classes=classes,
        n_filter=n_filter,
        dropout_rate=dropout_rate,
        n_resblock=n_resblock,
        model_name=model_name
    )
    
    print(f"\n模型参数:")
    print(f"输入形状: {input_shape}")
    print(f"滤波器数: {n_filter}")
    print(f"Dropout率: {dropout_rate:.3f}")
    print(f"残差块数: {n_resblock}")
    
    model.summary()
    
    # 设置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = Path("logs_multidim_deep", f"{model_name}_{timestamp}")
    logdir.mkdir(parents=True, exist_ok=True)
    
    # 训练模型
    learning_rate = 8e-4
    epochdots = tfdoc_model.EpochDots()
    
    history = compile_and_fit(
        model,
        x_train,
        y_train,
        batch_size,
        learning_rate,
        model_name,
        logdir,
        epochdots,
        validation_split=0.25,
        max_epochs=epochs,
    )
    
    # 评估模型
    eval_results = model.evaluate(x_test, y_test, verbose=0)
    if isinstance(eval_results, list):
        test_loss = eval_results[0]
        test_accuracy = eval_results[1] if len(eval_results) > 1 else None
    else:
        test_loss = eval_results
        test_accuracy = None
    
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    
    print(f"\n测试结果:")
    if test_accuracy is not None:
        print(f"测试准确率: {test_accuracy:.4f}")
    else:
        # 计算准确率从预测结果
        test_accuracy = np.mean(y_pred == y_test)
        print(f"测试准确率: {test_accuracy:.4f} (手动计算)")
    print(f"测试损失: {test_loss:.4f}")
    
    # 生成混淆矩阵
    confusion_mtx = tf.math.confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_mtx,
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
        annot=True,
        fmt="g",
    )
    plt.xlabel("Prediction")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix: {classes}分类, {dim}维")
    
    cm_path = logdir / f"{model_name}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    
    # 绘制训练历史
    plotter = tfdoc_plot.HistoryPlotter(metric="accuracy", smoothing_std=10)
    plt.figure(figsize=(10, 6))
    plotter.plot({model_name: history})
    plt.title(f"Training History: {classes}分类, {dim}维")
    
    acc_path = logdir / f"{model_name}_training_history.png"
    plt.savefig(acc_path)
    plt.close()
    
    # 保存模型
    model_path = logdir / "model"
    model.save(model_path)
    
    # 保存混淆矩阵数据
    np.save(logdir / "confusion_matrix.npy", confusion_mtx)
    
    return {
        'classes': classes,
        'dim': dim,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'model_name': model_name,
        'logdir': str(logdir),
        'confusion_matrix': confusion_mtx.numpy()
    }


def main():
    """主函数：运行所有实验"""
    print("多维深度网络实验")
    print("输入格式: (batch_size, 6, len, dim)")
    print("实验设置: 3/5分类 × 1/5/8维")
    
    # 设置随机种子
    np.random.seed(2022)
    tf.random.set_seed(2022)
    
    # 实验配置
    class_numbers = [3, 5]
    dimensions = [1, 5, 8]
    
    # 实验参数
    samples_per_class = 800
    length = 400
    epochs = 80
    batch_size = 64
    
    results = []
    
    print(f"\n实验参数:")
    print(f"每类样本数: {samples_per_class}")
    print(f"时间序列长度: {length}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    
    # 运行所有实验
    total_experiments = len(class_numbers) * len(dimensions)
    current_exp = 0
    
    for classes in class_numbers:
        for dim in dimensions:
            current_exp += 1
            print(f"\n进度: {current_exp}/{total_experiments}")
            
            try:
                result = run_experiment(
                    classes=classes,
                    dim=dim,
                    samples_per_class=samples_per_class,
                    length=length,
                    epochs=epochs,
                    batch_size=batch_size
                )
                results.append(result)
                
                print(f"✓ 实验完成: {classes}分类, {dim}维 - 准确率: {result['test_accuracy']:.4f}")
                
            except Exception as e:
                print(f"✗ 实验失败: {classes}分类, {dim}维 - 错误: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # 保存和展示结果
    if results:
        results_df = pd.DataFrame(results)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = Path("logs_multidim_deep") / f"experiment_results_{timestamp}.csv"
        results_df.to_csv(results_path, index=False)
        
        print(f"\n{'='*60}")
        print("实验结果汇总")
        print(f"{'='*60}")
        
        print("\n准确率结果:")
        pivot_table = results_df.pivot(index='dim', columns='classes', values='test_accuracy')
        print(pivot_table)
        
        # 可视化结果
        plt.figure(figsize=(12, 8))
        
        # 准确率热力图
        plt.subplot(2, 2, 1)
        sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Test Accuracy'})
        plt.title('测试准确率')
        plt.xlabel('分类数')
        plt.ylabel('维度')
        
        # 准确率条形图
        plt.subplot(2, 2, 2)
        results_df['exp_label'] = results_df['classes'].astype(str) + 'c_' + results_df['dim'].astype(str) + 'd'
        sorted_results = results_df.sort_values('test_accuracy', ascending=True)
        
        bars = plt.barh(range(len(sorted_results)), sorted_results['test_accuracy'])
        plt.yticks(range(len(sorted_results)), sorted_results['exp_label'])
        plt.xlabel('测试准确率')
        plt.title('准确率排行')
        plt.grid(axis='x', alpha=0.3)
        
        # 按分类数分组
        plt.subplot(2, 2, 3)
        class_acc = results_df.groupby('classes')['test_accuracy'].agg(['mean', 'std']).reset_index()
        plt.bar(class_acc['classes'], class_acc['mean'], yerr=class_acc['std'], capsize=5)
        plt.xlabel('分类数')
        plt.ylabel('平均准确率')
        plt.title('按分类数分组的准确率')
        plt.grid(axis='y', alpha=0.3)
        
        # 按维度分组
        plt.subplot(2, 2, 4)
        dim_acc = results_df.groupby('dim')['test_accuracy'].agg(['mean', 'std']).reset_index()
        plt.bar(dim_acc['dim'], dim_acc['mean'], yerr=dim_acc['std'], capsize=5)
        plt.xlabel('维度')
        plt.ylabel('平均准确率')
        plt.title('按维度分组的准确率')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        summary_path = Path("logs_multidim_deep") / f"experiment_summary_{timestamp}.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n结果已保存:")
        print(f"详细结果: {results_path}")
        print(f"汇总图表: {summary_path}")
        
        # 显示最佳结果
        best_result = results_df.loc[results_df['test_accuracy'].idxmax()]
        print(f"\n🏆 最佳结果:")
        print(f"配置: {best_result['classes']}分类, {best_result['dim']}维")
        print(f"准确率: {best_result['test_accuracy']:.4f}")
        print(f"模型: {best_result['model_name']}")


if __name__ == "__main__":
    main() 