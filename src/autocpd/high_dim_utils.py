"""
High-dimensional Multi-class Change Point Data Generation
Author: Modified for high-dimensional scenarios
Date: 2024-01-XX
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
from sklearn.decomposition import PCA
from scipy.stats import cauchy


def _generate_noise_by_type(n: int, d: int, noise_type: str, noise_std: float, 
                           ar_coef: float = 0.7, cauchy_scale: float = 0.3,
                           correlation_base: float = 0.3) -> np.ndarray:
    """
    根据噪声类型生成噪声
    
    Parameters:
    -----------
    n : int
        时间序列长度
    d : int
        维度
    noise_type : str
        噪声类型
    noise_std : float
        噪声强度
    ar_coef : float
        AR(1)系数
    cauchy_scale : float
        柯西分布尺度参数
    correlation_base : float
        基础相关系数
        
    Returns:
    --------
    np.ndarray
        生成的噪声，形状为 (d, n)
    """
    
    if noise_type == 'gaussian':
        # 高斯白噪声
        if d == 1:
            return np.random.normal(0, noise_std, (1, n))
        else:
            corr_matrix = _generate_correlation_matrix(d, correlation_base)
            cov_matrix = corr_matrix * (noise_std ** 2)
            return np.random.multivariate_normal(np.zeros(d), cov_matrix, n).T
            
    elif noise_type == 'ar1':
        # AR(1)噪声
        return _generate_ar_noise(n, d, ar_coef, noise_std, correlation_base)
        
    elif noise_type == 'cauchy':
        # 柯西噪声（重尾）
        if d == 1:
            noise = cauchy.rvs(loc=0, scale=cauchy_scale, size=(1, n))
            # 标准化到指定强度
            noise = noise * noise_std / np.std(noise)
            return noise
        else:
            # 多维柯西噪声（简化处理，各维度独立）
            noise = np.zeros((d, n))
            for i in range(d):
                noise[i] = cauchy.rvs(loc=0, scale=cauchy_scale, size=n)
                noise[i] = noise[i] * noise_std / np.std(noise[i])
            return noise
            
    elif noise_type == 'ar_random':
        # 随机AR系数噪声
        random_coef = np.random.uniform(0, 0.9)
        return _generate_ar_noise(n, d, random_coef, noise_std, correlation_base)
        
    else:
        # 默认返回高斯噪声
        warnings.warn(f"未知噪声类型: {noise_type}, 使用高斯噪声")
        return _generate_noise_by_type(n, d, 'gaussian', noise_std, ar_coef, cauchy_scale, correlation_base)


def _generate_ar_noise(n: int, d: int, ar_coef: float, noise_std: float, correlation_base: float) -> np.ndarray:
    """生成AR(1)噪声"""
    if d == 1:
        noise = np.zeros((1, n))
        innovation = np.random.normal(0, noise_std, n)
        noise[0, 0] = innovation[0]
        for t in range(1, n):
            noise[0, t] = ar_coef * noise[0, t-1] + innovation[t]
        return noise
    else:
        corr_matrix = _generate_correlation_matrix(d, correlation_base)
        cov_matrix = corr_matrix * (noise_std ** 2)
        
        noise = np.zeros((d, n))
        noise[:, 0] = np.random.multivariate_normal(np.zeros(d), cov_matrix)
        
        for t in range(1, n):
            innovation = np.random.multivariate_normal(np.zeros(d), cov_matrix)
            noise[:, t] = ar_coef * noise[:, t-1] + innovation
            
        return noise


def HighDimDataSetGen(
    N_sub: int,
    n: int, 
    d: int = 1,
    mean_changes: Optional[Dict] = None,
    var_changes: Optional[Dict] = None,
    correlation_changes: Optional[Dict] = None,
    trend_changes: Optional[Dict] = None,
    structural_changes: Optional[Dict] = None,
    n_trim: int = 0,
    noise_std: float = 1.0,
    noise_type: str = 'gaussian',
    ar_coef: float = 0.7,
    cauchy_scale: float = 0.3,
    correlation_base: float = 0.3,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    生成高维多类型变点数据集，支持多种噪声类型
    
    Parameters:
    -----------
    N_sub : int
        每类样本数量
    n : int  
        时间序列长度（最终输出长度）
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
        边界修剪大小（已废弃，保留参数兼容性）
    noise_std : float
        噪声标准差
    noise_type : str
        噪声类型: 'gaussian', 'ar1', 'cauchy', 'ar_random'
    ar_coef : float
        AR(1)噪声的自回归系数
    cauchy_scale : float
        柯西噪声的尺度参数
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
    
    print(f"生成时间序列长度: {n}")
    print(f"噪声类型: {noise_type}, 噪声强度: {noise_std}")
    
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
    
    # 传递噪声参数到生成函数
    noise_params = {
        'noise_std': noise_std,
        'noise_type': noise_type,
        'ar_coef': ar_coef,
        'cauchy_scale': cauchy_scale
    }
    
    # 类别 0: 无变点 (基线)
    print("生成类别 0: 无变点")
    no_change_data = _generate_no_change_data(N_sub, n, d, correlation_base, **noise_params)
    data_list.append(no_change_data)
    labels.extend([0] * N_sub)
    change_points.extend([None] * N_sub)
    
    # 其他类别: 各种变点类型
    for i, change_type in enumerate(change_types):
        class_idx = i + 1
        print(f"生成类别 {class_idx}: {change_type} 变点")
        
        if change_type == 'mean':
            change_data = _generate_mean_change_data(
                N_sub, n, d, mean_changes, correlation_base, **noise_params
            )
        elif change_type == 'variance':
            change_data = _generate_variance_change_data(
                N_sub, n, d, var_changes, correlation_base, **noise_params
            )
        elif change_type == 'correlation':
            change_data = _generate_correlation_change_data(
                N_sub, n, d, correlation_changes, correlation_base, **noise_params
            )
        elif change_type == 'trend':
            change_data = _generate_trend_change_data(
                N_sub, n, d, trend_changes, correlation_base, **noise_params
            )
        elif change_type == 'structural':
            change_data = _generate_structural_change_data(
                N_sub, n, d, structural_changes, correlation_base, **noise_params
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
        'dimensions': d,
        'noise_type': noise_type,
        'noise_level': noise_std
    }


def _generate_no_change_data(N_sub: int, n: int, d: int, correlation_base: float, **noise_params) -> np.ndarray:
    """生成无变点的基线数据"""
    data = []
    
    for _ in range(N_sub):
        # 使用新的噪声生成方法
        noise = _generate_noise_by_type(
            n, d, 
            noise_params['noise_type'], 
            noise_params['noise_std'],
            noise_params.get('ar_coef', 0.7),
            noise_params.get('cauchy_scale', 0.3),
            correlation_base
        )
        
        if d == 1:
            ts = noise.flatten()
        else:
            ts = noise
        
        data.append(ts)
    
    return np.array(data)


def _generate_mean_change_data(N_sub: int, n: int, d: int, mean_changes: Dict,
                             correlation_base: float, **noise_params) -> np.ndarray:
    """生成均值变化数据"""
    data = []
    magnitude = mean_changes['magnitude']
    dims = mean_changes.get('dimensions', 'all')
    
    for _ in range(N_sub):
        change_point = n // 2
        
        if d == 1:
            # 1维情况 - 前半段无偏移，后半段有偏移
            noise1 = _generate_noise_by_type(
                change_point, d, 
                noise_params['noise_type'], 
                noise_params['noise_std'],
                noise_params.get('ar_coef', 0.7),
                noise_params.get('cauchy_scale', 0.3),
                correlation_base
            )
            noise2 = _generate_noise_by_type(
                n - change_point, d, 
                noise_params['noise_type'], 
                noise_params['noise_std'],
                noise_params.get('ar_coef', 0.7),
                noise_params.get('cauchy_scale', 0.3),
                correlation_base
            )
            ts1 = noise1.flatten()
            ts2 = noise2.flatten() + magnitude
            ts = np.concatenate([ts1, ts2])
        else:
            # 多维情况
            # 确定变化的维度
            if dims == 'all':
                change_dims = list(range(d))
            elif isinstance(dims, (list, tuple)):
                change_dims = dims
            else:
                change_dims = [0]
            
            # 生成基础噪声
            noise1 = _generate_noise_by_type(
                change_point, d, 
                noise_params['noise_type'], 
                noise_params['noise_std'],
                noise_params.get('ar_coef', 0.7),
                noise_params.get('cauchy_scale', 0.3),
                correlation_base
            )
            noise2 = _generate_noise_by_type(
                n - change_point, d, 
                noise_params['noise_type'], 
                noise_params['noise_std'],
                noise_params.get('ar_coef', 0.7),
                noise_params.get('cauchy_scale', 0.3),
                correlation_base
            )
            
            # 前半段
            ts1 = noise1
            
            # 后半段 - 在指定维度添加均值偏移
            ts2 = noise2.copy()
            for dim in change_dims:
                if dim < d:
                    mean_shift = magnitude * np.random.choice([-1, 1])
                    ts2[dim] += mean_shift
            
            ts = np.concatenate([ts1, ts2], axis=-1)
            
        data.append(ts)
    
    return np.array(data)


def _generate_variance_change_data(N_sub: int, n: int, d: int, var_changes: Dict,
                                 correlation_base: float, **noise_params) -> np.ndarray:
    """生成方差变化数据"""
    data = []
    magnitude = var_changes['magnitude']
    dims = var_changes.get('dimensions', 'all')
    
    for _ in range(N_sub):
        change_point = n // 2
        
        if d == 1:
            # 1维情况
            # 前半段 - 原始噪声强度
            noise1 = _generate_noise_by_type(
                change_point, d, 
                noise_params['noise_type'], 
                noise_params['noise_std'],
                noise_params.get('ar_coef', 0.7),
                noise_params.get('cauchy_scale', 0.3),
                correlation_base
            )
            # 后半段 - 增加方差
            new_std = noise_params['noise_std'] * (1 + magnitude)
            noise2 = _generate_noise_by_type(
                n - change_point, d, 
                noise_params['noise_type'], 
                new_std,
                noise_params.get('ar_coef', 0.7),
                noise_params.get('cauchy_scale', 0.3),
                correlation_base
            )
            ts1 = noise1.flatten()
            ts2 = noise2.flatten()
            ts = np.concatenate([ts1, ts2])
        else:
            # 多维情况 - 更复杂，需要修改协方差矩阵
            # 对于非高斯噪声，简化处理
            if noise_params['noise_type'] != 'gaussian':
                # 前半段
                noise1 = _generate_noise_by_type(
                    change_point, d, 
                    noise_params['noise_type'], 
                    noise_params['noise_std'],
                    noise_params.get('ar_coef', 0.7),
                    noise_params.get('cauchy_scale', 0.3),
                    correlation_base
                )
                
                # 后半段 - 增加方差
                change_dims = list(range(d)) if dims == 'all' else (dims if isinstance(dims, (list, tuple)) else [0])
                noise2 = _generate_noise_by_type(
                    n - change_point, d, 
                    noise_params['noise_type'], 
                    noise_params['noise_std'],
                    noise_params.get('ar_coef', 0.7),
                    noise_params.get('cauchy_scale', 0.3),
                    correlation_base
                )
                
                # 在指定维度上放大方差
                for dim in change_dims:
                    if dim < d:
                        noise2[dim] *= (1 + magnitude)
                
                ts = np.concatenate([noise1, noise2], axis=-1)
            else:
                # 高斯噪声的精确处理
                corr_matrix = _generate_correlation_matrix(d, correlation_base)
                
                # 前半段
                cov_matrix1 = corr_matrix * (noise_params['noise_std'] ** 2)
                ts1 = np.random.multivariate_normal(np.zeros(d), cov_matrix1, change_point).T
                
                # 后半段 - 在指定维度修改方差
                cov_matrix2 = cov_matrix1.copy()
                change_dims = list(range(d)) if dims == 'all' else (dims if isinstance(dims, (list, tuple)) else [0])
                
                for dim in change_dims:
                    if dim < d:
                        new_var = (noise_params['noise_std'] * (1 + magnitude)) ** 2
                        cov_matrix2[dim, dim] = new_var
                        # 调整协方差
                        for other_dim in range(d):
                            if other_dim != dim:
                                scale_factor = np.sqrt(new_var / cov_matrix1[dim, dim])
                                cov_matrix2[dim, other_dim] *= scale_factor
                                cov_matrix2[other_dim, dim] = cov_matrix2[dim, other_dim]
                
                ts2 = np.random.multivariate_normal(np.zeros(d), cov_matrix2, n - change_point).T
                ts = np.concatenate([ts1, ts2], axis=-1)
            
        data.append(ts)
    
    return np.array(data)


def _generate_correlation_change_data(N_sub: int, n: int, d: int, corr_changes: Dict,
                                    correlation_base: float, **noise_params) -> np.ndarray:
    """生成相关性变化数据 (仅适用于多维)"""
    if d == 1:
        warnings.warn("相关性变化需要多维数据 (d > 1)")
        return _generate_no_change_data(N_sub, n, d, correlation_base, **noise_params)
    
    data = []
    magnitude = corr_changes['magnitude']
    
    for _ in range(N_sub):
        change_point = n // 2
        
        # 前半段 - 基础相关性
        corr_matrix1 = _generate_correlation_matrix(d, correlation_base)
        cov_matrix1 = corr_matrix1 * (noise_params['noise_std'] ** 2)
        ts1 = np.random.multivariate_normal(np.zeros(d), cov_matrix1, change_point).T
        
        # 后半段 - 改变相关性
        new_corr = correlation_base + magnitude * np.random.choice([-1, 1])
        new_corr = np.clip(new_corr, -0.9, 0.9)  # 确保合理范围
        corr_matrix2 = _generate_correlation_matrix(d, new_corr)
        cov_matrix2 = corr_matrix2 * (noise_params['noise_std'] ** 2)
        ts2 = np.random.multivariate_normal(np.zeros(d), cov_matrix2, n - change_point).T
        
        ts = np.concatenate([ts1, ts2], axis=-1)
            
        data.append(ts)
    
    return np.array(data)


def _generate_trend_change_data(N_sub: int, n: int, d: int, trend_changes: Dict,
                              correlation_base: float, **noise_params) -> np.ndarray:
    """生成趋势变化数据"""
    data = []
    magnitude = trend_changes['magnitude']
    dims = trend_changes.get('dimensions', 'all')
    
    for _ in range(N_sub):
        change_point = n // 2
        
        if d == 1:
            # 1维情况
            # 前半段 - 无趋势
            ts1 = np.random.normal(0, noise_params['noise_std'], change_point)
            # 后半段 - 有趋势
            trend = np.linspace(0, magnitude * (n - change_point), n - change_point)
            ts2 = trend + np.random.normal(0, noise_params['noise_std'], n - change_point)
            ts = np.concatenate([ts1, ts2])
        else:
            # 多维情况
            corr_matrix = _generate_correlation_matrix(d, correlation_base)
            cov_matrix = corr_matrix * (noise_params['noise_std'] ** 2)
            
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
            
        data.append(ts)
    
    return np.array(data)


def _generate_structural_change_data(N_sub: int, n: int, d: int, struct_changes: Dict,
                                   correlation_base: float, **noise_params) -> np.ndarray:
    """生成结构变化数据 (仅适用于多维)"""
    if d == 1:
        warnings.warn("结构变化需要多维数据 (d > 1)")
        return _generate_trend_change_data(N_sub, n, d, struct_changes, correlation_base, **noise_params)
    
    data = []
    magnitude = struct_changes['magnitude']
    struct_type = struct_changes.get('type', 'regime')
    
    for _ in range(N_sub):
        change_point = n // 2
        
        if struct_type == 'regime':
            # 制度切换：改变动力学结构
            # 前半段 - AR(1) 过程
            ts1 = _generate_ar_process(change_point, d, 0.3, noise_params['noise_std'], correlation_base)
            
            # 后半段 - AR(1) 过程with不同参数
            new_ar_coef = 0.3 + magnitude * np.random.choice([-1, 1]) * 0.4
            new_ar_coef = np.clip(new_ar_coef, -0.8, 0.8)
            ts2 = _generate_ar_process(n - change_point, d, new_ar_coef, noise_params['noise_std'], correlation_base)
            
        elif struct_type == 'coupling':
            # 耦合变化：改变维度间的依赖关系
            corr_matrix1 = _generate_correlation_matrix(d, correlation_base)
            cov_matrix1 = corr_matrix1 * (noise_params['noise_std'] ** 2)
            ts1 = np.random.multivariate_normal(np.zeros(d), cov_matrix1, change_point).T
            
            # 引入新的耦合结构
            coupling_matrix = np.eye(d) + magnitude * 0.1 * np.random.randn(d, d)
            ts2_base = np.random.multivariate_normal(np.zeros(d), cov_matrix1, n - change_point).T
            ts2 = coupling_matrix @ ts2_base
            
        else:
            # 默认：简单的结构变化
            ts1 = _generate_no_change_data(1, change_point, d, correlation_base, **noise_params)[0]
            ts2 = _generate_no_change_data(1, n - change_point, d, noise_params['noise_std'] * (1 + magnitude), correlation_base, **noise_params)[0]
        
        ts = np.concatenate([ts1, ts2], axis=-1)
            
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
            'mean_changes': {'enabled': True, 'magnitude': 1.5 if d <= 3 else 1.0, 'dimensions': 'all' if d <= 3 else [0, 1]},
            'var_changes': {'enabled': True, 'magnitude': 0.4 if d <= 3 else 0.3, 'dimensions': 'all' if d <= 3 else [0, 2]},
            'correlation_changes': {'enabled': d > 1, 'magnitude': 0.3},
            'trend_changes': {'enabled': d > 1, 'magnitude': 0.015, 'dimensions': [0] if d > 1 else 'all'},
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
        数据变换类型: 'flatten', 'channel', 'reshape_20x20', 'multidim_6channel', 'pca', 'transpose'
        
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
        
        # 对于Transformer，需要添加特征维度
        if transform_type == 'transpose':
            data_x = data_x.reshape(data_x.shape[0], data_x.shape[1], 1)  # (N, T) -> (N, T, 1)
        
        return data_x, labels
    
    if transform_type == 'flatten':
        # 展平：(N, d, T) -> (N, d*T)
        N, d, T = data_x.shape
        data_x = data_x.reshape(N, d * T)
        
    elif transform_type == 'transpose':
        # 转置：(N, d, T) -> (N, T, d) - 适合Transformer
        N, d, T = data_x.shape
        data_x = data_x.transpose(0, 2, 1)  # (N, d, T) -> (N, T, d)
        print(f"转置为Transformer格式: (N={N}, d={d}, T={T}) -> (N={N}, T={T}, d={d})")
        
    elif transform_type == 'channel':
        # 保持通道格式：(N, d, T) - 适合CNN
        pass
        
    elif transform_type == 'pca':
        # PCA降维：(N, d, T) -> (N, T*n_components) 通过对维度进行PCA降维
        N, d, T = data_x.shape
        
        # 将数据重组为 (N*T, d) 进行PCA
        data_reshaped = data_x.transpose(0, 2, 1).reshape(N * T, d)  # (N, d, T) -> (N*T, d)
        
        # 应用PCA降维
        pca = PCA(n_components=3)
        data_pca = pca.fit_transform(data_reshaped)  # (N*T, d) -> (N*T, n_components)
        
        # 重组回合适的格式
        n_components = data_pca.shape[1]
        if n_components == 1:
            # 如果降到1维，输出 (N, T)
            data_x = data_pca.reshape(N, T)
            print(f"PCA降维完成: (N={N}, d={d}, T={T}) -> (N={N}, T={T})")
        else:
            # 如果保留多维，输出 (N, T*n_components) 用于flatten
            data_x = data_pca.reshape(N, T * n_components)
            print(f"PCA降维完成: (N={N}, d={d}, T={T}) -> (N={N}, T={T * n_components})")
        
        print(f"PCA参数: n_components={n_components}")
        print(f"PCA解释方差比: {pca.explained_variance_ratio_}")
        print(f"PCA累计解释方差比: {pca.explained_variance_ratio_.sum():.4f}")
        
    elif transform_type == 'reshape_20x20':
        # 专门为深度CNN重新整形：(N, d, T) -> (N, d, 20, 20)
        N, d, T = data_x.shape
        
        # 确保时间序列长度是400（20x20）
        if T != 400:
            raise ValueError(f"时间序列长度必须是400，当前是{T}")
        
        # 重新整形为20x20格式
        data_x = data_x.reshape(N, d, 20, 20)
        print(f"重新整形为深度网络格式: (N={N}, channels={d}, height=20, width=20)")
    
    elif transform_type == 'multidim_6channel':
        # 新的多维6通道格式：(N, d, T) -> (N, 6, T, d)
        # 前3个通道：原始数据重复3遍
        # 后3个通道：平方数据重复3遍
        N, d, T = data_x.shape
        
        # 生成6种变换
        result = np.zeros((N, 6, T, d))
        
        # 变换1-3: 原始数据（重复3遍）
        original_data_transposed = data_x.transpose(0, 2, 1)  # (N, d, T) -> (N, T, d)
        result[:, 0, :, :] = original_data_transposed
        result[:, 1, :, :] = original_data_transposed
        result[:, 2, :, :] = original_data_transposed
        
        # # 变换4-6: 平方数据（重复3遍）
        # squared_data_transposed = np.square(data_x).transpose(0, 2, 1)  # (N, d, T) -> (N, T, d)
        # result[:, 3, :, :] = squared_data_transposed
        # result[:, 4, :, :] = squared_data_transposed
        # result[:, 5, :, :] = squared_data_transposed
 
        ############################################################33
        result[:, 3, :, :] = original_data_transposed
        result[:, 4, :, :] = original_data_transposed
        result[:, 5, :, :] = original_data_transposed
        
        data_x = result
        print(f"重新整形为多维6通道格式: (N={N}, channels=6, length={T}, dim={d})")
        print(f"  通道0-2: 原始数据重复3遍")
        print(f"  通道3-5: 平方数据重复3遍")
    
    else:
        raise ValueError(f"不支持的变换类型: {transform_type}")
    
    return data_x, labels 