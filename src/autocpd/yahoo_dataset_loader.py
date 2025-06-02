"""
Yahoo时间序列异常检测数据集加载器
支持加载A1-A4 Benchmark数据集用于变点检测任务

数据集结构:
- A1Benchmark: 真实生产流量数据
- A2Benchmark: 合成时间序列（异常值）
- A3Benchmark: 合成时间序列（仅异常值）
- A4Benchmark: 合成时间序列（异常值+变点）

Usage:
    loader = YahooDatasetLoader('data')
    data_dict = loader.load_for_changepoint_detection(
        benchmark='A4', 
        n_files=10, 
        segment_length=200,
        overlap_ratio=0.1
    )
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import glob


class YahooDatasetLoader:
    """Yahoo时间序列异常检测数据集加载器"""
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化数据加载器
        
        Parameters:
        -----------
        data_dir : str
            数据目录路径，包含A1-A4Benchmark文件夹
        """
        self.data_dir = Path(data_dir)
        self.benchmarks = {
            'A1': self.data_dir / "A1Benchmark",  # 真实数据
            'A2': self.data_dir / "A2Benchmark",  # 合成数据（异常值）
            'A3': self.data_dir / "A3Benchmark",  # 合成数据（仅异常值）
            'A4': self.data_dir / "A4Benchmark"   # 合成数据（异常值+变点）
        }
        
        # 验证数据目录存在
        for benchmark, path in self.benchmarks.items():
            if not path.exists():
                warnings.warn(f"Benchmark {benchmark} directory not found: {path}")
    
    def get_available_files(self, benchmark: str) -> List[Path]:
        """获取指定benchmark的所有可用文件"""
        if benchmark not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(self.benchmarks.keys())}")
        
        benchmark_dir = self.benchmarks[benchmark]
        if not benchmark_dir.exists():
            return []
        
        if benchmark == 'A1':
            pattern = "real_*.csv"
        elif benchmark == 'A2':
            pattern = "synthetic_*.csv"
        elif benchmark in ['A3', 'A4']:
            pattern = f"{benchmark}Benchmark-TS*.csv"
        else:
            pattern = "*.csv"
        
        files = list(benchmark_dir.glob(pattern))
        # 排除可能的汇总文件
        files = [f for f in files if not f.name.endswith('_all.csv')]
        return sorted(files)
    
    def load_single_file(self, file_path: Path, benchmark: str) -> pd.DataFrame:
        """加载单个时间序列文件"""
        try:
            df = pd.read_csv(file_path)
            
            # 根据benchmark类型处理不同的列格式
            if benchmark == 'A1':
                # A1Benchmark: timestamp, value, is_anomaly
                df.columns = ['timestamp', 'value', 'is_anomaly']
                df['changepoint'] = 0  # A1没有变点标注
            elif benchmark == 'A2':
                # A2Benchmark: timestamp, value, is_anomaly
                df.columns = ['timestamp', 'value', 'is_anomaly']
                df['changepoint'] = 0  # A2没有变点标注
            elif benchmark in ['A3', 'A4']:
                # A3/A4Benchmark: timestamps, value, anomaly, changepoint, trend, noise, seasonality1-3
                expected_cols = ['timestamps', 'value', 'anomaly', 'changepoint', 
                               'trend', 'noise', 'seasonality1', 'seasonality2', 'seasonality3']
                if len(df.columns) >= 4:
                    df = df.iloc[:, :4]  # 只保留前4列
                    df.columns = ['timestamp', 'value', 'is_anomaly', 'changepoint']
                else:
                    raise ValueError(f"Unexpected format in {file_path}")
            
            return df
            
        except Exception as e:
            warnings.warn(f"Error loading {file_path}: {e}")
            return None
    
    def segment_time_series(self, 
                          df: pd.DataFrame, 
                          segment_length: int = 200,
                          overlap_ratio: float = 0.1,
                          min_changepoint_ratio: float = 0.05) -> List[Dict]:
        """
        将长时间序列分割成固定长度的段，用于变点检测
        
        Parameters:
        -----------
        df : pd.DataFrame
            时间序列数据
        segment_length : int
            每个段的长度
        overlap_ratio : float
            段之间的重叠比例 (0-1)
        min_changepoint_ratio : float
            段中最少变点比例，用于确定标签
            
        Returns:
        --------
        List[Dict]
            每个字典包含: {'data': np.array, 'label': int, 'has_changepoint': bool, 'changepoint_positions': list}
        """
        segments = []
        values = df['value'].values
        changepoints = df['changepoint'].values if 'changepoint' in df.columns else np.zeros(len(df))
        
        step_size = int(segment_length * (1 - overlap_ratio))
        
        for start_idx in range(0, len(values) - segment_length + 1, step_size):
            end_idx = start_idx + segment_length
            
            segment_data = values[start_idx:end_idx]
            segment_changepoints = changepoints[start_idx:end_idx]
            
            # 检查是否包含足够的变点
            changepoint_ratio = np.sum(segment_changepoints) / segment_length
            has_changepoint = changepoint_ratio >= min_changepoint_ratio
            
            # 找到变点位置
            changepoint_positions = np.where(segment_changepoints == 1)[0].tolist()
            
            segments.append({
                'data': segment_data,
                'label': 1 if has_changepoint else 0,  # 二分类：有变点=1，无变点=0
                'has_changepoint': has_changepoint,
                'changepoint_positions': changepoint_positions,
                'changepoint_ratio': changepoint_ratio,
                'start_idx': start_idx,
                'end_idx': end_idx
            })
        
        return segments
    
    def load_for_changepoint_detection(self,
                                     benchmark: str = 'A4',
                                     n_files: Optional[int] = None,
                                     segment_length: int = 200,
                                     overlap_ratio: float = 0.1,
                                     train_ratio: float = 0.8,
                                     normalize: str = 'standard',
                                     min_changepoint_ratio: float = 0.01,
                                     balance_classes: bool = True,
                                     seed: int = 42) -> Dict:
        """
        加载Yahoo数据集用于变点检测任务
        
        Parameters:
        -----------
        benchmark : str
            基准数据集 ('A1', 'A2', 'A3', 'A4')
        n_files : Optional[int]
            加载文件数量，None表示加载所有
        segment_length : int
            时间序列段长度
        overlap_ratio : float
            段重叠比例
        train_ratio : float
            训练集比例
        normalize : str
            归一化方法 ('standard', 'minmax', 'none')
        min_changepoint_ratio : float
            最小变点比例阈值
        balance_classes : bool
            是否平衡类别
        seed : int
            随机种子
            
        Returns:
        --------
        Dict
            包含训练和测试数据的字典
        """
        np.random.seed(seed)
        
        # 获取文件列表
        files = self.get_available_files(benchmark)
        if not files:
            raise ValueError(f"No files found for benchmark {benchmark}")
        
        if n_files is not None:
            files = files[:n_files]
        
        print(f"Loading {len(files)} files from {benchmark}Benchmark...")
        
        # 加载所有时间序列并分割
        all_segments = []
        successful_files = 0
        
        for file_path in files:
            df = self.load_single_file(file_path, benchmark)
            if df is not None and len(df) >= segment_length:
                segments = self.segment_time_series(
                    df, 
                    segment_length=segment_length,
                    overlap_ratio=overlap_ratio,
                    min_changepoint_ratio=min_changepoint_ratio
                )
                all_segments.extend(segments)
                successful_files += 1
        
        if not all_segments:
            raise ValueError("No valid segments extracted from files")
        
        print(f"Successfully loaded {successful_files} files, extracted {len(all_segments)} segments")
        
        # 提取数据和标签
        X = np.array([seg['data'] for seg in all_segments])
        y = np.array([seg['label'] for seg in all_segments])
        
        # 统计类别分布
        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == 0)
        print(f"Class distribution: No changepoint={n_negative}, Has changepoint={n_positive}")
        
        # 添加变点统计信息
        changepoint_ratios = [seg['changepoint_ratio'] for seg in all_segments]
        print(f"Changepoint ratio statistics:")
        print(f"  Min: {np.min(changepoint_ratios):.4f}")
        print(f"  Max: {np.max(changepoint_ratios):.4f}")
        print(f"  Mean: {np.mean(changepoint_ratios):.4f}")
        print(f"  Std: {np.std(changepoint_ratios):.4f}")
        print(f"  Above threshold ({min_changepoint_ratio}): {np.sum(np.array(changepoint_ratios) >= min_changepoint_ratio)}")
        
        # 平衡类别（如果需要）
        if balance_classes and n_positive > 0 and n_negative > 0:  # 确保两个类别都有样本
            min_samples = min(n_positive, n_negative)
            pos_indices = np.where(y == 1)[0]
            neg_indices = np.where(y == 0)[0]
            
            # 随机采样
            pos_selected = np.random.choice(pos_indices, min_samples, replace=False)
            neg_selected = np.random.choice(neg_indices, min_samples, replace=False)
            
            selected_indices = np.concatenate([pos_selected, neg_selected])
            np.random.shuffle(selected_indices)
            
            X = X[selected_indices]
            y = y[selected_indices]
            
            print(f"Balanced dataset: {len(X)} samples ({min_samples} per class)")
        elif balance_classes:
            # 如果某个类别为空，发出警告但继续处理
            if n_positive == 0:
                warnings.warn(f"No positive samples found with threshold {min_changepoint_ratio}. "
                            f"Consider lowering the threshold or using balance_classes=False.")
            if n_negative == 0:
                warnings.warn(f"No negative samples found. This is unusual for time series data.")
            print(f"Cannot balance classes: positive={n_positive}, negative={n_negative}")
            print(f"Using unbalanced dataset: {len(X)} samples")
        
        # 检查是否有数据可用于训练
        if len(X) == 0:
            raise ValueError(f"No data available after processing. "
                           f"Try lowering min_changepoint_ratio (current: {min_changepoint_ratio}) "
                           f"or setting balance_classes=False")
        
        # 归一化
        if normalize == 'standard':
            scaler = StandardScaler()
            X_reshaped = X.reshape(-1, 1)
            X_normalized = scaler.fit_transform(X_reshaped).reshape(X.shape)
        elif normalize == 'minmax':
            scaler = MinMaxScaler()
            X_reshaped = X.reshape(-1, 1)
            X_normalized = scaler.fit_transform(X_reshaped).reshape(X.shape)
        elif normalize == 'none':
            X_normalized = X
            scaler = None
        else:
            raise ValueError(f"Unknown normalization method: {normalize}")
        
        # 分割训练和测试集
        n_samples = len(X_normalized)
        n_train = int(n_samples * train_ratio)
        
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        X_train = X_normalized[train_indices]
        X_test = X_normalized[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        # 构建返回字典
        result = {
            'data_x': X_normalized,  # 全部数据
            'labels': y.tolist(),
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'n_classes': 2,  # 二分类
            'dimensions': 1,  # 一维时间序列
            'change_types': ['no_changepoint', 'has_changepoint'],
            'segment_length': segment_length,
            'benchmark': benchmark,
            'n_files': successful_files,
            'total_segments': len(all_segments),
            'scaler': scaler,
            'metadata': {
                'overlap_ratio': overlap_ratio,
                'min_changepoint_ratio': min_changepoint_ratio,
                'normalize': normalize,
                'balance_classes': balance_classes,
                'train_ratio': train_ratio
            }
        }
        
        print(f"Dataset prepared:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Test: {len(X_test)} samples") 
        print(f"  Segment length: {segment_length}")
        print(f"  Classes: {result['change_types']}")
        
        return result
    
    def load_for_multi_class_detection(self,
                                     benchmark: str = 'A4',
                                     n_files: Optional[int] = None,
                                     segment_length: int = 200,
                                     overlap_ratio: float = 0.1,
                                     train_ratio: float = 0.8,
                                     normalize: str = 'standard',
                                     changepoint_thresholds: List[float] = [0.01, 0.05, 0.1],
                                     seed: int = 42) -> Dict:
        """
        加载数据用于多分类变点检测（基于变点密度分类）
        
        Parameters:
        -----------
        changepoint_thresholds : List[float]
            变点密度阈值，用于定义不同类别
            例如: [0.01, 0.05, 0.1] -> 4个类别: [0, 0.01), [0.01, 0.05), [0.05, 0.1), [0.1, 1]
        """
        np.random.seed(seed)
        
        # 获取文件列表
        files = self.get_available_files(benchmark)
        if not files:
            raise ValueError(f"No files found for benchmark {benchmark}")
        
        if n_files is not None:
            files = files[:n_files]
        
        print(f"Loading {len(files)} files from {benchmark}Benchmark...")
        
        # 加载所有时间序列并分割
        all_segments = []
        successful_files = 0
        
        for file_path in files:
            df = self.load_single_file(file_path, benchmark)
            if df is not None and len(df) >= segment_length:
                segments = self.segment_time_series(
                    df, 
                    segment_length=segment_length,
                    overlap_ratio=overlap_ratio,
                    min_changepoint_ratio=0.0  # 不过滤
                )
                all_segments.extend(segments)
                successful_files += 1
        
        if not all_segments:
            raise ValueError("No valid segments extracted from files")
        
        # 基于变点密度分类
        changepoint_ratios = [seg['changepoint_ratio'] for seg in all_segments]
        thresholds = sorted(changepoint_thresholds)
        
        labels = []
        class_names = []
        
        # 定义类别
        class_names.append(f"no_changepoint (0-{thresholds[0]:.3f})")
        for i in range(len(thresholds)):
            if i == len(thresholds) - 1:
                class_names.append(f"high_changepoint ({thresholds[i]:.3f}-1.0)")
            else:
                class_names.append(f"medium_changepoint ({thresholds[i]:.3f}-{thresholds[i+1]:.3f})")
        
        # 分配标签
        for ratio in changepoint_ratios:
            label = 0
            for i, threshold in enumerate(thresholds):
                if ratio >= threshold:
                    label = i + 1
                else:
                    break
            labels.append(label)
        
        # 转换为numpy数组
        X = np.array([seg['data'] for seg in all_segments])
        y = np.array(labels)
        
        # 统计类别分布
        for i in range(len(class_names)):
            count = np.sum(y == i)
            print(f"Class {i} ({class_names[i]}): {count} samples")
        
        # 归一化和数据分割逻辑与二分类相同
        if normalize == 'standard':
            scaler = StandardScaler()
            X_reshaped = X.reshape(-1, 1)
            X_normalized = scaler.fit_transform(X_reshaped).reshape(X.shape)
        elif normalize == 'minmax':
            scaler = MinMaxScaler()
            X_reshaped = X.reshape(-1, 1)
            X_normalized = scaler.fit_transform(X_reshaped).reshape(X.shape)
        elif normalize == 'none':
            X_normalized = X
            scaler = None
        else:
            raise ValueError(f"Unknown normalization method: {normalize}")
        
        # 分割训练和测试集
        n_samples = len(X_normalized)
        n_train = int(n_samples * train_ratio)
        
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        X_train = X_normalized[train_indices]
        X_test = X_normalized[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        # 构建返回字典
        result = {
            'data_x': X_normalized,
            'labels': y.tolist(),
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'n_classes': len(class_names),
            'dimensions': 1,
            'change_types': class_names,
            'segment_length': segment_length,
            'benchmark': benchmark,
            'n_files': successful_files,
            'total_segments': len(all_segments),
            'scaler': scaler,
            'metadata': {
                'overlap_ratio': overlap_ratio,
                'changepoint_thresholds': thresholds,
                'normalize': normalize,
                'train_ratio': train_ratio
            }
        }
        
        print(f"Multi-class dataset prepared:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Test: {len(X_test)} samples") 
        print(f"  Classes: {len(class_names)}")
        
        return result


def demo_yahoo_loader():
    """演示如何使用Yahoo数据集加载器"""
    # 初始化加载器
    loader = YahooDatasetLoader('data')
    
    # 检查可用文件
    for benchmark in ['A1', 'A4']:
        files = loader.get_available_files(benchmark)
        print(f"{benchmark}Benchmark: {len(files)} files available")
        if files:
            print(f"  Sample files: {[f.name for f in files[:3]]}")
    
    print("\n" + "="*50)
    
    # 二分类变点检测 - 优化参数
    print("Loading for binary changepoint detection...")
    binary_data = loader.load_for_changepoint_detection(
        benchmark='A4',
        n_files=10,  # 增加文件数量获取更多样本
        segment_length=500,  # 增加段长度以包含更多变点
        overlap_ratio=0.5,  # 增加重叠以获取更多样本
        normalize='standard',
        min_changepoint_ratio=0.002,  # 根据实际数据调整阈值
        balance_classes=True  # 平衡类别
    )
    
    print(f"Binary classification data shape: {binary_data['data_x'].shape}")
    print(f"Classes: {binary_data['change_types']}")
    
    print("\n" + "="*50)
    
    # 多分类变点检测 - 调整阈值
    print("Loading for multi-class changepoint detection...")
    multi_data = loader.load_for_multi_class_detection(
        benchmark='A4',
        n_files=10,
        segment_length=500,  # 增加段长度
        overlap_ratio=0.5,
        changepoint_thresholds=[0.001, 0.003, 0.006]  # 根据实际数据分布调整阈值
    )
    
    print(f"Multi-class data shape: {multi_data['data_x'].shape}")
    print(f"Classes: {multi_data['change_types']}")
    
    print("\n" + "="*50)
    
    # 添加数据质量分析
    print("Data quality analysis:")
    print(f"Binary data - Train/Test split: {len(binary_data['X_train'])}/{len(binary_data['X_test'])}")
    print(f"Binary data - Class balance in train: "
          f"Class 0: {np.sum(binary_data['y_train'] == 0)}, "
          f"Class 1: {np.sum(binary_data['y_train'] == 1)}")
    
    if multi_data['n_classes'] > 1:
        for i in range(multi_data['n_classes']):
            count = np.sum(multi_data['y_train'] == i)
            print(f"Multi-class train - Class {i}: {count} samples")
    
    # 建议进一步的参数调整
    print("\n" + "="*50)
    print("Parameter tuning suggestions:")
    print("1. For better changepoint detection:")
    print("   - Consider segment_length=300-800 for more changepoints per segment")
    print("   - Use overlap_ratio=0.3-0.7 for more diverse samples")
    print("   - Adjust min_changepoint_ratio based on your specific needs")
    print("2. For multi-class detection:")
    print("   - Use thresholds like [0.001, 0.002, 0.004] for Yahoo A4 data")
    print("   - Consider using percentile-based thresholds")
    print("3. Consider using A1 (real data) if synthetic data doesn't fit your use case")


if __name__ == "__main__":
    demo_yahoo_loader() 