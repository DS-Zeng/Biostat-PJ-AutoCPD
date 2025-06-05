#!/usr/bin/env python3
"""
AutoCPD 噪声鲁棒性批量实验脚本
系统比较不同噪声类型对模型性能的影响
Author: Noise Robustness Evaluation
Date: 2024-01-XX

实验设计:
- 噪声类型: Gaussian, AR(1), Cauchy, AR_random
- 噪声强度: 0.5, 1.0, 1.5, 2.0
- 模型: simple, deep, transformer  
- 分类数: 3, 5
- 维度: 1, 5, 8
"""

import subprocess
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class NoiseExperiment:
    """噪声鲁棒性实验管理类"""
    
    def __init__(self, output_dir="noise_experiment_results", base_args=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_dir / "noise_experiment_results.csv"
        self.summary_file = self.output_dir / "noise_experiment_summary.txt"
        
        # 基础参数
        self.base_args = base_args or {
            'samples': 1200,
            'length': 400,
            'epochs': None,  # 自动确定
            'batch_size': 64,
            'seed': 2022,
            'verbose': 1,
            'validation_split': 0.2,
            'preset': 'basic',
            'gpu': 'auto'
        }
        
        # 噪声实验配置
        self.experiment_configs = self._generate_noise_experiment_configs()
        
        # 结果存储
        self.results = []
        
    def _generate_noise_experiment_configs(self):
        """生成所有噪声实验配置"""
        configs = []
        
        # 噪声类型配置
        noise_configs = {
            'gaussian': {
                'type': 'gaussian',
                'description': 'Gaussian white noise',
                'params': {}
            },
            # 'ar1_weak': {
            #     'type': 'ar1', 
            #     'description': 'AR(1) noise with ρ=0.3',
            #     'params': {'ar_coef': 0.3}
            # },
            # 'ar1_moderate': {
            #     'type': 'ar1',
            #     'description': 'AR(1) noise with ρ=0.7', 
            #     'params': {'ar_coef': 0.7}
            # },
            # 'cauchy': {
            #     'type': 'cauchy',
            #     'description': 'Heavy-tailed Cauchy noise',
            #     'params': {'scale': 0.3}
            # },
            # 'ar_random': {
            #     'type': 'ar_random',
            #     'description': 'AR(1) with random coefficients',
            #     'params': {}
            # }
        }
        
        # 噪声强度
        # noise_levels = [0.5, 1.0, 1.5, 2.0]
        noise_levels = [1.5]
        
        # 模型和数据配置
        # models = ['transformer', 'simple', 'deep']  # 优先测试transformer
        models = ['deep']
        class_numbers = [3, 5]
        dimensions = [1, 5, 8]
        
        for noise_name, noise_config in noise_configs.items():
            for noise_level in noise_levels:
                for model in models:
                    for classes in class_numbers:
                        for dim in dimensions:
                            # 根据维度选择合适的preset
                            if dim == 1:
                                preset = 'basic'
                            elif dim <= 5:
                                preset = 'basic'
                            else:
                                preset = 'mean_var_only'
                            
                            config = {
                                'model': model,
                                'classes': classes,
                                'dim': dim,
                                'noise_type': noise_config['type'],
                                'noise_level': noise_level,
                                'noise_params': noise_config['params'],
                                'noise_description': noise_config['description'],
                                'preset': preset,
                                'exp_id': f"{model}_{classes}c_{dim}d_{noise_name}_n{noise_level}"
                            }
                            configs.append(config)
        
        return configs
    
    def run_single_experiment(self, config):
        """运行单个噪声实验"""
        print(f"\n{'='*80}")
        print(f"噪声实验: {config['exp_id']}")
        print(f"模型: {config['model']}, 分类: {config['classes']}, 维度: {config['dim']}")
        print(f"噪声: {config['noise_description']}, 强度: {config['noise_level']}")
        print(f"{'='*80}")
        
        # 构建命令
        cmd = [
            sys.executable, "train_configurable.py",
            "--model", config['model'],
            "--classes", str(config['classes']),
            "--dim", str(config['dim']),
            "--preset", config['preset'],
            "--output_dir", str(self.output_dir / "individual_results")
        ]
        
        # 添加噪声参数
        cmd.extend(["--noise_type", config['noise_type']])
        cmd.extend(["--noise_level", str(config['noise_level'])])
        
        # 添加噪声特定参数
        for param_name, param_value in config['noise_params'].items():
            cmd.extend([f"--{param_name}", str(param_value)])
        
        # 根据模型类型添加特定参数
        if config['model'] == 'simple':
            cmd.extend(["--transform", "flatten"])
        elif config['model'] == 'deep':
            cmd.extend(["--deep_input_format", "multidim_6channel"])
        elif config['model'] == 'transformer':
            cmd.extend(["--transform", "transpose"])
            print(f"  → 使用新的Transformer架构 (transpose格式)")
        
        # 添加base_args
        for key, value in self.base_args.items():
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{key}")
                else:
                    cmd.extend([f"--{key}", str(value)])
        
        # 设置实验名称
        exp_name = f"noise_{config['exp_id']}_{datetime.now().strftime('%H%M%S')}"
        cmd.extend(["--exp_name", exp_name])
        
        print(f"命令: {' '.join(cmd)}")
        
        # 运行实验
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=36000  # 10小时超时
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                # 解析结果
                accuracy = self._extract_accuracy_from_output(result.stdout)
                status = "成功"
                error_msg = None
            else:
                accuracy = None
                status = "失败"
                error_msg = result.stderr
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            accuracy = None
            status = "超时"
            error_msg = "实验超时"
            
        except Exception as e:
            duration = time.time() - start_time
            accuracy = None
            status = "错误"
            error_msg = str(e)
        
        # 记录结果
        result_record = {
            'exp_id': config['exp_id'],
            'model': config['model'],
            'classes': config['classes'],
            'dimension': config['dim'],
            'noise_type': config['noise_type'],
            'noise_level': config['noise_level'],
            'noise_description': config['noise_description'],
            'preset': config['preset'],
            'best_accuracy': accuracy,
            'duration_minutes': duration / 60,
            'status': status,
            'error_msg': error_msg,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(result_record)
        
        # 打印结果
        if accuracy is not None:
            print(f"✓ 最佳准确率: {accuracy:.4f}, 时间: {duration/60:.1f}分钟")
        else:
            print(f"✗ 失败: {status}, 时间: {duration/60:.1f}分钟")
            if error_msg:
                print(f"错误: {error_msg[:200]}...")
        
        return result_record
    
    def _extract_accuracy_from_output(self, output):
        """从输出中提取准确率"""
        lines = output.split('\n')
        best_val_acc = None
        
        # 寻找最佳验证准确率
        for line in lines:
            if 'Best validation accuracy:' in line:
                try:
                    best_val_acc = float(line.split(':')[1].strip())
                    break
                except:
                    pass
            elif 'val_accuracy:' in line and 'loss:' in line:
                try:
                    # 解析类似 "val_accuracy: 0.8500" 的行
                    parts = line.split('val_accuracy:')
                    if len(parts) > 1:
                        acc_part = parts[1].strip().split()[0]
                        best_val_acc = float(acc_part)
                except:
                    pass
        
        return best_val_acc
    
    def run_all_experiments(self):
        """运行所有噪声实验"""
        print(f"开始运行 {len(self.experiment_configs)} 个噪声实验")
        print(f"结果将保存到: {self.output_dir}")
        
        for i, config in enumerate(self.experiment_configs):
            print(f"\n进度: {i+1}/{len(self.experiment_configs)}")
            result = self.run_single_experiment(config)
            
            # 定期保存部分结果
            if (i + 1) % 5 == 0:
                self._save_partial_results()
        
        # 保存最终结果并生成报告
        self._generate_final_report()
    
    def _save_partial_results(self):
        """保存部分结果"""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(self.results_file, index=False)
            print(f"已保存 {len(self.results)} 个实验结果到 {self.results_file}")
    
    def _generate_final_report(self):
        """生成最终报告"""
        if not self.results:
            print("没有实验结果可生成报告")
            return
        
        # 保存详细结果
        df = pd.DataFrame(self.results) 
        df.to_csv(self.results_file, index=False)
        
        # 生成汇总报告
        self._generate_noise_summary_report(df)
        
        # 生成可视化
        self._generate_noise_visualizations(df)
        
        # 打印主要结果
        self._print_noise_main_results(df)
        
        print(f"\n{'='*60}")
        print("噪声实验完成!")
        print(f"详细结果: {self.results_file}")
        print(f"汇总报告: {self.summary_file}")
        print(f"可视化图表: {self.output_dir}/visualizations/")
        print(f"{'='*60}")
    
    def _generate_noise_summary_report(self, df):
        """生成噪声实验汇总报告"""
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write("AutoCPD 噪声鲁棒性实验汇总报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总实验数: {len(df)}\n")
            f.write(f"成功实验数: {len(df[df['status'] == '成功'])}\n")
            f.write(f"失败实验数: {len(df[df['status'] != '成功'])}\n\n")
            
            # 按噪声类型汇总
            f.write("按噪声类型汇总:\n")
            f.write("-" * 30 + "\n")
            noise_summary = df[df['status'] == '成功'].groupby('noise_type')['best_accuracy'].agg(['mean', 'std', 'count'])
            f.write(noise_summary.to_string())
            f.write("\n\n")
            
            # 按模型类型汇总
            f.write("按模型类型汇总:\n") 
            f.write("-" * 30 + "\n")
            model_summary = df[df['status'] == '成功'].groupby('model')['best_accuracy'].agg(['mean', 'std', 'count'])
            f.write(model_summary.to_string())
            f.write("\n\n")
            
            # 按噪声强度汇总
            f.write("按噪声强度汇总:\n")
            f.write("-" * 30 + "\n")
            level_summary = df[df['status'] == '成功'].groupby('noise_level')['best_accuracy'].agg(['mean', 'std', 'count'])
            f.write(level_summary.to_string())
            f.write("\n\n")
            
            # 最佳配置
            if len(df[df['status'] == '成功']) > 0:
                best_row = df[df['status'] == '成功'].loc[df['best_accuracy'].idxmax()]
                f.write("最佳配置:\n")
                f.write("-" * 20 + "\n")
                f.write(f"实验ID: {best_row['exp_id']}\n")
                f.write(f"模型: {best_row['model']}\n")
                f.write(f"噪声类型: {best_row['noise_type']}\n")
                f.write(f"噪声强度: {best_row['noise_level']}\n")
                f.write(f"准确率: {best_row['best_accuracy']:.4f}\n")
                f.write(f"维度: {best_row['dimension']}\n")
                f.write(f"分类数: {best_row['classes']}\n")
    
    def _generate_noise_visualizations(self, df):
        """生成噪声实验可视化图表"""
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        success_df = df[df['status'] == '成功'].copy()
        if len(success_df) == 0:
            print("没有成功的实验结果，跳过可视化")
            return
        
        plt.style.use('default')
        
        # 1. 噪声类型对比
        fig, ax = plt.subplots(figsize=(12, 8))
        noise_acc = success_df.groupby(['noise_type', 'model'])['best_accuracy'].mean().unstack()
        noise_acc.plot(kind='bar', ax=ax)
        ax.set_title('不同噪声类型下的模型性能对比')
        ax.set_xlabel('噪声类型')
        ax.set_ylabel('平均准确率')
        ax.legend(title='模型')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(viz_dir / "noise_type_comparison.png", dpi=300)
        plt.close()
        
        # 2. 噪声强度影响
        fig, ax = plt.subplots(figsize=(12, 8))
        for model in success_df['model'].unique():
            model_data = success_df[success_df['model'] == model]
            noise_effect = model_data.groupby('noise_level')['best_accuracy'].mean()
            ax.plot(noise_effect.index, noise_effect.values, marker='o', label=model)
        
        ax.set_title('噪声强度对模型性能的影响')
        ax.set_xlabel('噪声强度')
        ax.set_ylabel('平均准确率')
        ax.legend(title='模型')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / "noise_level_effect.png", dpi=300)
        plt.close()
        
        # 3. 热力图: 模型 vs 噪声类型
        fig, ax = plt.subplots(figsize=(10, 8))
        pivot_data = success_df.pivot_table(
            values='best_accuracy', 
            index='model', 
            columns='noise_type', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax)
        ax.set_title('模型-噪声类型性能热力图')
        plt.tight_layout()
        plt.savefig(viz_dir / "model_noise_heatmap.png", dpi=300)
        plt.close()
        
        # 4. 维度影响分析
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for i, model in enumerate(['simple', 'deep', 'transformer']):
            model_data = success_df[success_df['model'] == model]
            if len(model_data) > 0:
                dim_effect = model_data.groupby(['dimension', 'noise_type'])['best_accuracy'].mean().unstack()
                dim_effect.plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'{model.capitalize()} 模型 - 维度vs噪声')
                axes[i].set_xlabel('维度')
                axes[i].set_ylabel('平均准确率')
                axes[i].legend(title='噪声类型', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "dimension_noise_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化图表已保存到: {viz_dir}")
    
    def _print_noise_main_results(self, df):
        """打印噪声实验主要结果"""
        print(f"\n{'='*60}")
        print("噪声鲁棒性实验主要结果")
        print(f"{'='*60}")
        
        success_df = df[df['status'] == '成功']
        if len(success_df) == 0:
            print("没有成功的实验结果")
            return
        
        # 整体统计
        print(f"成功实验: {len(success_df)}/{len(df)}")
        print(f"平均准确率: {success_df['best_accuracy'].mean():.4f} ± {success_df['best_accuracy'].std():.4f}")
        print(f"最高准确率: {success_df['best_accuracy'].max():.4f}")
        print(f"最低准确率: {success_df['best_accuracy'].min():.4f}")
        
        # 按噪声类型排名
        print(f"\n噪声类型鲁棒性排名:")
        noise_ranking = success_df.groupby('noise_type')['best_accuracy'].mean().sort_values(ascending=False)
        for i, (noise_type, acc) in enumerate(noise_ranking.items()):
            print(f"  {i+1}. {noise_type}: {acc:.4f}")
        
        # 按模型类型排名
        print(f"\n模型鲁棒性排名:")
        model_ranking = success_df.groupby('model')['best_accuracy'].mean().sort_values(ascending=False)
        for i, (model, acc) in enumerate(model_ranking.items()):
            print(f"  {i+1}. {model}: {acc:.4f}")
        
        # 最佳组合
        best_combo = success_df.groupby(['model', 'noise_type'])['best_accuracy'].mean().sort_values(ascending=False).head(5)
        print(f"\n最佳模型-噪声组合 (Top 5):")
        for i, ((model, noise_type), acc) in enumerate(best_combo.items()):
            print(f"  {i+1}. {model} + {noise_type}: {acc:.4f}")


def main():
    """主函数"""
    print("AutoCPD 噪声鲁棒性批量实验")
    print("=" * 50)
    
    # 创建实验管理器
    experiment = NoiseExperiment(
        output_dir="noise_experiment_results",
        base_args={
            'samples': 1500,  # 减少样本数加快实验
            'length': 400,
            'batch_size': 64,
            'seed': 2022,
            'verbose': 1,
            'validation_split': 0.2,
            'gpu': '0'
        }
    )
    
    print(f"计划运行 {len(experiment.experiment_configs)} 个噪声实验")
    
    # 询问用户是否继续
    response = input("是否继续? (y/n): ")
    if response.lower() != 'y':
        print("实验已取消")
        return
    
    # 运行所有实验
    experiment.run_all_experiments()


if __name__ == "__main__":
    main() 