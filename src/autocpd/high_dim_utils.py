"""
High-dimensional Multi-class Change Point Data Generation
Author: Modified for high-dimensional scenarios
Date: 2024-01-XX
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings


def HighDimDataSetGen(
    N_sub: int,
    n: int, 
    d: int = 1,
    mean_changes: Optional[Dict] = None,
    var_changes: Optional[Dict] = None,
    correlation_changes: Optional[Dict] = None,
    trend_changes: Optional[Dict] = None,
    structural_changes: Optional[Dict] = None,
    n_trim: int = 10,
    noise_std: float = 1.0,
    correlation_base: float = 0.3,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    生成高维多类型变点数据集
    
    Parameters:
    -----------
    N_sub : int
        每类样本数量
    n : int  
        时间序列长度
    d : int
        数据维度 (默认1维，可扩展到高维)
    mean_changes : dict, optional
        均值变化参数 {'enabled': bool, 'magnitude': float, 'dimensions': list}
    var_changes : dict, optional
        方差变化参数 {'enabled': bool, 'magnitude': float, 'dimensions': list}
    correlation_changes : dict, optional
        相关性变化参数 {'enabled': bool, 'magnitude': float}
    trend_changes : dict, optional
        趋势变化参数 {'enabled': bool, 'magnitude': float, 'dimensions': list}
    structural_changes : dict, optional
        结构变化参数 {'enabled': bool, 'type': str, 'magnitude': float}
    n_trim : int
        边界修剪大小
    noise_std : float
        噪声标准差
    correlation_base : float
        基础相关系数
    seed : int, optional
        随机种子
        
    Returns:
    --------
    dict
        包含生成数据的字典 {'data_x': ndarray, 'labels': list, 'change_points': list}
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # 默认参数设置
    default_mean = {'enabled': True, 'magnitude': 2.0, 'dimensions': 'all'}
    default_var = {'enabled': True, 'magnitude': 0.5, 'dimensions': 'all'}
    default_corr = {'enabled': True, 'magnitude': 0.4}
    default_trend = {'enabled': True, 'magnitude': 0.02, 'dimensions': 'all'}
    default_struct = {'enabled': True, 'type': 'regime', 'magnitude': 1.5}
    
    mean_changes = mean_changes or default_mean
    var_changes = var_changes or default_var  
    correlation_changes = correlation_changes or default_corr
    trend_changes = trend_changes or default_trend
    structural_changes = structural_changes or default_struct
    
    # 确定变点类型
    change_types = []
    if mean_changes['enabled']:
        change_types.append('mean')
    if var_changes['enabled']:
        change_types.append('variance') 
    if d > 1 and correlation_changes['enabled']:
        change_types.append('correlation')
    if trend_changes['enabled']:
        change_types.append('trend')
    if d > 1 and structural_changes['enabled']:
        change_types.append('structural')
    
    n_classes = len(change_types) + 1  # +1 for no-change class
    
    print(f"生成 {n_classes} 类高维变点数据:")
    print(f"  维度: {d}")
    print(f"  变点类型: {change_types}")
    print(f"  每类样本数: {N_sub}")
    
    data_list = []
    labels = []
    change_points = []
    
    # 类别 0: 无变点 (基线)
    print("生成类别 0: 无变点")
    no_change_data = _generate_no_change_data(N_sub, n, d, noise_std, correlation_base, n_trim)
    data_list.append(no_change_data)
    labels.extend([0] * N_sub)
    change_points.extend([None] * N_sub)
    
    # 其他类别: 各种变点类型
    for i, change_type in enumerate(change_types):
        class_idx = i + 1
        print(f"生成类别 {class_idx}: {change_type} 变点")
        
        if change_type == 'mean':
            change_data = _generate_mean_change_data(
                N_sub, n, d, mean_changes, noise_std, correlation_base, n_trim
            )
        elif change_type == 'variance':
            change_data = _generate_variance_change_data(
                N_sub, n, d, var_changes, noise_std, correlation_base, n_trim
            )
        elif change_type == 'correlation':
            change_data = _generate_correlation_change_data(
                N_sub, n, d, correlation_changes, noise_std, correlation_base, n_trim
            )
        elif change_type == 'trend':
            change_data = _generate_trend_change_data(
                N_sub, n, d, trend_changes, noise_std, correlation_base, n_trim
            )
        elif change_type == 'structural':
            change_data = _generate_structural_change_data(
                N_sub, n, d, structural_changes, noise_std, correlation_base, n_trim
            )
        
        data_list.append(change_data)
        labels.extend([class_idx] * N_sub)
        change_points.extend([n//2] * N_sub)  # 变点在中间位置
    
    # 合并所有数据
    data_x = np.concatenate(data_list, axis=0)
    
    print(f"数据生成完成:")
    print(f"  总形状: {data_x.shape}")
    print(f"  类别分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    return {
        'data_x': data_x,
        'labels': labels,
        'change_points': change_points,
        'change_types': ['no_change'] + change_types,
        'n_classes': n_classes,
        'dimensions': d
    }


def _generate_no_change_data(N_sub: int, n: int, d: int, noise_std: float, 
                           correlation_base: float, n_trim: int) -> np.ndarray:
    """生成无变点的基线数据"""
    data = []
    
    for _ in range(N_sub):
        if d == 1:
            # 1维情况
            ts = np.random.normal(0, noise_std, n)
        else:
            # 多维情况
            # 生成相关矩阵
            corr_matrix = _generate_correlation_matrix(d, correlation_base)
            cov_matrix = corr_matrix * (noise_std ** 2)
            
            # 生成多维时间序列
            ts = np.random.multivariate_normal(np.zeros(d), cov_matrix, n).T
        
        # 修剪边界
        if n_trim > 0:
            ts = ts[..., n_trim:-n_trim]
        
        data.append(ts)
    
    return np.array(data)


def _generate_mean_change_data(N_sub: int, n: int, d: int, mean_changes: Dict,
                             noise_std: float, correlation_base: float, n_trim: int) -> np.ndarray:
    """生成均值变化数据"""
    data = []
    magnitude = mean_changes['magnitude']
    dims = mean_changes.get('dimensions', 'all')
    
    for _ in range(N_sub):
        change_point = n // 2
        
        if d == 1:
            # 1维情况
            ts1 = np.random.normal(0, noise_std, change_point)
            ts2 = np.random.normal(magnitude, noise_std, n - change_point)
            ts = np.concatenate([ts1, ts2])
        else:
            # 多维情况
            corr_matrix = _generate_correlation_matrix(d, correlation_base)
            cov_matrix = corr_matrix * (noise_std ** 2)
            
            # 确定变化的维度
            if dims == 'all':
                change_dims = list(range(d))
            elif isinstance(dims, (list, tuple)):
                change_dims = dims
            else:
                change_dims = [0]  # 默认第一维
            
            # 前半段
            mean1 = np.zeros(d)
            ts1 = np.random.multivariate_normal(mean1, cov_matrix, change_point).T
            
            # 后半段 - 在指定维度上添加均值变化
            mean2 = np.zeros(d)
            for dim in change_dims:
                if dim < d:
                    mean2[dim] = magnitude * np.random.choice([-1, 1])
            
            ts2 = np.random.multivariate_normal(mean2, cov_matrix, n - change_point).T
            ts = np.concatenate([ts1, ts2], axis=-1)
        
        # 修剪边界
        if n_trim > 0:
            ts = ts[..., n_trim:-n_trim]
            
        data.append(ts)
    
    return np.array(data)


def _generate_variance_change_data(N_sub: int, n: int, d: int, var_changes: Dict,
                                 noise_std: float, correlation_base: float, n_trim: int) -> np.ndarray:
    """生成方差变化数据"""
    data = []
    magnitude = var_changes['magnitude']
    dims = var_changes.get('dimensions', 'all')
    
    for _ in range(N_sub):
        change_point = n // 2
        
        if d == 1:
            # 1维情况
            ts1 = np.random.normal(0, noise_std, change_point)
            new_std = noise_std * (1 + magnitude)
            ts2 = np.random.normal(0, new_std, n - change_point)
            ts = np.concatenate([ts1, ts2])
        else:
            # 多维情况
            corr_matrix = _generate_correlation_matrix(d, correlation_base)
            
            # 前半段
            cov_matrix1 = corr_matrix * (noise_std ** 2)
            ts1 = np.random.multivariate_normal(np.zeros(d), cov_matrix1, change_point).T
            
            # 后半段 - 在指定维度上改变方差
            cov_matrix2 = cov_matrix1.copy()
            if dims == 'all':
                change_dims = list(range(d))
            elif isinstance(dims, (list, tuple)):
                change_dims = dims
            else:
                change_dims = [0]
            
            for dim in change_dims:
                if dim < d:
                    new_var = (noise_std * (1 + magnitude)) ** 2
                    cov_matrix2[dim, dim] = new_var
                    # 调整协方差
                    for j in range(d):
                        if j != dim:
                            cov_matrix2[dim, j] = cov_matrix2[j, dim] = \
                                corr_matrix[dim, j] * np.sqrt(cov_matrix2[dim, dim] * cov_matrix2[j, j])
            
            ts2 = np.random.multivariate_normal(np.zeros(d), cov_matrix2, n - change_point).T
            ts = np.concatenate([ts1, ts2], axis=-1)
        
        # 修剪边界
        if n_trim > 0:
            ts = ts[..., n_trim:-n_trim]
            
        data.append(ts)
    
    return np.array(data)


def _generate_correlation_change_data(N_sub: int, n: int, d: int, corr_changes: Dict,
                                    noise_std: float, correlation_base: float, n_trim: int) -> np.ndarray:
    """生成相关性变化数据 (仅适用于多维)"""
    if d == 1:
        warnings.warn("相关性变化需要多维数据 (d > 1)")
        return _generate_no_change_data(N_sub, n, d, noise_std, correlation_base, n_trim)
    
    data = []
    magnitude = corr_changes['magnitude']
    
    for _ in range(N_sub):
        change_point = n // 2
        
        # 前半段 - 基础相关性
        corr_matrix1 = _generate_correlation_matrix(d, correlation_base)
        cov_matrix1 = corr_matrix1 * (noise_std ** 2)
        ts1 = np.random.multivariate_normal(np.zeros(d), cov_matrix1, change_point).T
        
        # 后半段 - 改变相关性
        new_corr = correlation_base + magnitude * np.random.choice([-1, 1])
        new_corr = np.clip(new_corr, -0.9, 0.9)  # 确保合理范围
        corr_matrix2 = _generate_correlation_matrix(d, new_corr)
        cov_matrix2 = corr_matrix2 * (noise_std ** 2)
        ts2 = np.random.multivariate_normal(np.zeros(d), cov_matrix2, n - change_point).T
        
        ts = np.concatenate([ts1, ts2], axis=-1)
        
        # 修剪边界
        if n_trim > 0:
            ts = ts[..., n_trim:-n_trim]
            
        data.append(ts)
    
    return np.array(data)


def _generate_trend_change_data(N_sub: int, n: int, d: int, trend_changes: Dict,
                              noise_std: float, correlation_base: float, n_trim: int) -> np.ndarray:
    """生成趋势变化数据"""
    data = []
    magnitude = trend_changes['magnitude']
    dims = trend_changes.get('dimensions', 'all')
    
    for _ in range(N_sub):
        change_point = n // 2
        
        if d == 1:
            # 1维情况
            # 前半段 - 无趋势
            ts1 = np.random.normal(0, noise_std, change_point)
            # 后半段 - 有趋势
            trend = np.linspace(0, magnitude * (n - change_point), n - change_point)
            ts2 = trend + np.random.normal(0, noise_std, n - change_point)
            ts = np.concatenate([ts1, ts2])
        else:
            # 多维情况
            corr_matrix = _generate_correlation_matrix(d, correlation_base)
            cov_matrix = corr_matrix * (noise_std ** 2)
            
            # 确定变化的维度
            if dims == 'all':
                change_dims = list(range(d))
            elif isinstance(dims, (list, tuple)):
                change_dims = dims
            else:
                change_dims = [0]
            
            # 前半段 - 无趋势
            ts1 = np.random.multivariate_normal(np.zeros(d), cov_matrix, change_point).T
            
            # 后半段 - 在指定维度添加趋势
            ts2_base = np.random.multivariate_normal(np.zeros(d), cov_matrix, n - change_point).T
            
            for dim in change_dims:
                if dim < d:
                    trend_magnitude = magnitude * np.random.choice([-1, 1])
                    trend = np.linspace(0, trend_magnitude * (n - change_point), n - change_point)
                    ts2_base[dim] += trend
            
            ts = np.concatenate([ts1, ts2_base], axis=-1)
        
        # 修剪边界
        if n_trim > 0:
            ts = ts[..., n_trim:-n_trim]
            
        data.append(ts)
    
    return np.array(data)


def _generate_structural_change_data(N_sub: int, n: int, d: int, struct_changes: Dict,
                                   noise_std: float, correlation_base: float, n_trim: int) -> np.ndarray:
    """生成结构变化数据 (仅适用于多维)"""
    if d == 1:
        warnings.warn("结构变化需要多维数据 (d > 1)")
        return _generate_trend_change_data(N_sub, n, d, struct_changes, noise_std, correlation_base, n_trim)
    
    data = []
    magnitude = struct_changes['magnitude']
    struct_type = struct_changes.get('type', 'regime')
    
    for _ in range(N_sub):
        change_point = n // 2
        
        if struct_type == 'regime':
            # 制度切换：改变动力学结构
            # 前半段 - AR(1) 过程
            ts1 = _generate_ar_process(change_point, d, 0.3, noise_std, correlation_base)
            
            # 后半段 - AR(1) 过程with不同参数
            new_ar_coef = 0.3 + magnitude * np.random.choice([-1, 1]) * 0.4
            new_ar_coef = np.clip(new_ar_coef, -0.8, 0.8)
            ts2 = _generate_ar_process(n - change_point, d, new_ar_coef, noise_std, correlation_base)
            
        elif struct_type == 'coupling':
            # 耦合变化：改变维度间的依赖关系
            corr_matrix1 = _generate_correlation_matrix(d, correlation_base)
            cov_matrix1 = corr_matrix1 * (noise_std ** 2)
            ts1 = np.random.multivariate_normal(np.zeros(d), cov_matrix1, change_point).T
            
            # 引入新的耦合结构
            coupling_matrix = np.eye(d) + magnitude * 0.1 * np.random.randn(d, d)
            ts2_base = np.random.multivariate_normal(np.zeros(d), cov_matrix1, n - change_point).T
            ts2 = coupling_matrix @ ts2_base
            
        else:
            # 默认：简单的结构变化
            ts1 = _generate_no_change_data(1, change_point, d, noise_std, correlation_base, 0)[0]
            ts2 = _generate_no_change_data(1, n - change_point, d, noise_std * (1 + magnitude), correlation_base, 0)[0]
        
        ts = np.concatenate([ts1, ts2], axis=-1)
        
        # 修剪边界
        if n_trim > 0:
            ts = ts[..., n_trim:-n_trim]
            
        data.append(ts)
    
    return np.array(data)


def _generate_correlation_matrix(d: int, base_corr: float) -> np.ndarray:
    """生成相关矩阵"""
    if d == 1:
        return np.array([[1.0]])
    
    # 生成随机相关矩阵
    corr_matrix = np.eye(d)
    for i in range(d):
        for j in range(i+1, d):
            corr = base_corr + 0.2 * np.random.randn()
            corr = np.clip(corr, -0.9, 0.9)
            corr_matrix[i, j] = corr_matrix[j, i] = corr
    
    # 确保正定性
    eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
    eigenvals = np.maximum(eigenvals, 0.01)  # 确保正定
    corr_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    # 重新归一化对角线
    diag_sqrt = np.sqrt(np.diag(corr_matrix))
    corr_matrix = corr_matrix / diag_sqrt[:, None] / diag_sqrt[None, :]
    
    return corr_matrix


def _generate_ar_process(n: int, d: int, ar_coef: float, noise_std: float, correlation_base: float) -> np.ndarray:
    """生成多维AR(1)过程"""
    corr_matrix = _generate_correlation_matrix(d, correlation_base)
    cov_matrix = corr_matrix * (noise_std ** 2)
    
    # 初始化
    if d == 1:
        ts = np.zeros(n)
        ts[0] = np.random.normal(0, noise_std)
        for t in range(1, n):
            ts[t] = ar_coef * ts[t-1] + np.random.normal(0, noise_std)
        return ts.reshape(1, -1)
    else:
        ts = np.zeros((d, n))
        ts[:, 0] = np.random.multivariate_normal(np.zeros(d), cov_matrix)
        
        for t in range(1, n):
            innovation = np.random.multivariate_normal(np.zeros(d), cov_matrix)
            ts[:, t] = ar_coef * ts[:, t-1] + innovation
        
        return ts


# 便捷函数：预设配置
def get_preset_config(preset_name: str, d: int = 1) -> Dict:
    """获取预设的变点配置"""
    
    configs = {
        'basic': {
            'mean_changes': {'enabled': True, 'magnitude': 2.0, 'dimensions': 'all'},
            'var_changes': {'enabled': True, 'magnitude': 0.5, 'dimensions': 'all'},
            'correlation_changes': {'enabled': False},
            'trend_changes': {'enabled': False},
            'structural_changes': {'enabled': False}
        },
        
        'full': {
            'mean_changes': {'enabled': True, 'magnitude': 2.0, 'dimensions': 'all'},
            'var_changes': {'enabled': True, 'magnitude': 0.5, 'dimensions': 'all'},
            'correlation_changes': {'enabled': d > 1, 'magnitude': 0.4},
            'trend_changes': {'enabled': True, 'magnitude': 0.02, 'dimensions': 'all'},
            'structural_changes': {'enabled': d > 1, 'type': 'regime', 'magnitude': 1.0}
        },
        
        'mean_var_only': {
            'mean_changes': {'enabled': True, 'magnitude': 2.5, 'dimensions': 'all'},
            'var_changes': {'enabled': True, 'magnitude': 0.8, 'dimensions': 'all'},
            'correlation_changes': {'enabled': False},
            'trend_changes': {'enabled': False},
            'structural_changes': {'enabled': False}
        },
        
        'correlation_focus': {
            'mean_changes': {'enabled': False},
            'var_changes': {'enabled': False},
            'correlation_changes': {'enabled': d > 1, 'magnitude': 0.6},
            'trend_changes': {'enabled': True, 'magnitude': 0.03, 'dimensions': 'all'},
            'structural_changes': {'enabled': d > 1, 'type': 'coupling', 'magnitude': 1.5}
        }
    }
    
    return configs.get(preset_name, configs['basic'])


def prepare_high_dim_data_for_training(data_dict: Dict, transform_type: str = 'flatten') -> Tuple[np.ndarray, np.ndarray]:
    """
    为训练准备高维数据
    
    Parameters:
    -----------
    data_dict : dict
        HighDimDataSetGen的输出
    transform_type : str
        数据变换类型: 'flatten', 'channel', 'pca'
        
    Returns:
    --------
    tuple
        (transformed_data, labels)
    """
    data_x = data_dict['data_x']
    labels = np.array(data_dict['labels'])
    d = data_dict['dimensions']
    
    if d == 1:
        # 1维数据，去除多余维度
        if len(data_x.shape) == 3:
            data_x = data_x.squeeze(1)  # (N, 1, T) -> (N, T)
        return data_x, labels
    
    if transform_type == 'flatten':
        # 展平：(N, d, T) -> (N, d*T)
        N, d, T = data_x.shape
        data_x = data_x.reshape(N, d * T)
        
    elif transform_type == 'channel':
        # 保持通道格式：(N, d, T) - 适合CNN
        pass
        
    elif transform_type == 'pca':
        # PCA降维 + 展平
        N, d, T = data_x.shape
        data_reshaped = data_x.reshape(N, d * T)
        
        from sklearn.decomposition import PCA
        n_components = min(50, d * T // 2)  # 自适应组件数
        pca = PCA(n_components=n_components)
        data_x = pca.fit_transform(data_reshaped)
        
        print(f"PCA降维: {d*T} -> {data_x.shape[1]} (解释方差比: {pca.explained_variance_ratio_.sum():.3f})")
    
    return data_x, labels 