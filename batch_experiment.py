#!/usr/bin/env python3
"""
AutoCPD 批量实验脚本
系统比较 Simple NN, Deep NN, Transformer 在不同设置下的表现
Author: Comprehensive Evaluation
Date: 2024-01-XX

实验设计:
- 模型: simple, deep, transformer
- 分类数: 3, 5
- 维度: 1, 5, 10
- 总计: 3 × 2 × 3 = 18 个实验
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


class BatchExperiment:
    """批量实验管理类"""
    
    def __init__(self, output_dir="batch_results", base_args=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_dir / "experiment_results.csv"
        self.summary_file = self.output_dir / "experiment_summary.txt"
        
        # 基础参数
        self.base_args = base_args or {
            'samples': 1500,
            'length': 400,
            'epochs': None,  # 自动确定
            'batch_size': 64,
            'seed': 2022,
            'verbose': 1,  # 减少输出
            'validation_split': 0.2,
            'preset': 'basic',
            'gpu': 'auto'
        }
        
        # 实验配置
        self.experiment_configs = self._generate_experiment_configs()
        
        # 结果存储
        self.results = []
        
    def _generate_experiment_configs(self):
        """生成所有实验配置"""
        configs = []
        
        # models = ['simple', 'deep', 'transformer']  # 启用所有三种模型
        models = ['transformer']
        class_numbers = [3, 5]
        dimensions = [1, 5, 8]
        
        for model in models:
            for classes in class_numbers:
                for dim in dimensions:
                    # 根据维度选择合适的preset
                    if dim == 1:
                        preset = 'basic'
                        use_original = True  # 1维使用原版数据
                    elif dim <= 5:
                        preset = 'basic'
                        use_original = False
                    else:
                        preset = 'mean_var_only'  # 高维简化配置
                        use_original = False
                    
                    config = {
                        'model': model,
                        'classes': classes,
                        'dim': dim,
                        'preset': preset,
                        'use_original': use_original,
                        'exp_id': f"{model}_{classes}c_{dim}d"
                    }
                    configs.append(config)
        
        return configs
    
    def run_single_experiment(self, config):
        """运行单个实验"""
        print(f"\n{'='*60}")
        print(f"实验: {config['exp_id']}")
        print(f"模型: {config['model']}, 分类: {config['classes']}, 维度: {config['dim']}")
        print(f"{'='*60}")
        
        # 构建命令
        cmd = [
            sys.executable, "train_configurable.py",
            "--model", config['model'],
            "--classes", str(config['classes']),
            "--dim", str(config['dim']),
            "--preset", config['preset'],
            "--output_dir", str(self.output_dir / "individual_results")
        ]
        
        # 根据模型类型添加特定参数
        if config['model'] == 'simple':
            # Simple模型使用flatten
            cmd.extend(["--transform", "flatten"])
        elif config['model'] == 'deep':
            # 深度模型使用6通道格式
            cmd.extend(["--deep_input_format", "multidim_6channel"])
            cmd.extend(["--transform", "multidim_6channel"])
        elif config['model'] == 'transformer':
            # 新的Transformer使用transpose格式（对所有维度都适用）
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
        
        # 添加use_original标志
        if config['use_original']:
            cmd.append("--use_original")
        
        # 设置实验名称
        exp_name = f"batch_{config['exp_id']}_{datetime.now().strftime('%H%M%S')}"
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
            'preset': config['preset'],
            'use_original': config['use_original'],
            'best_accuracy': accuracy,  # 最佳验证准确率
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
        """从输出中提取准确率（最佳验证准确率）"""
        lines = output.split('\n')
        
        # 首先尝试提取最佳准确率（新格式）
        for line in lines:
            if "最佳准确率:" in line:
                try:
                    # 提取类似 "最佳准确率: 0.8750" 的数字
                    parts = line.split("最佳准确率:")
                    if len(parts) > 1:
                        accuracy_str = parts[1].strip()
                        return float(accuracy_str)
                except:
                    continue
        
        # 备用：尝试提取最佳验证准确率（从详细输出中）
        for line in lines:
            if "最佳验证准确率:" in line:
                try:
                    # 提取类似 "最佳验证准确率: 0.8750 (第 5 轮)" 的数字
                    parts = line.split("最佳验证准确率:")
                    if len(parts) > 1:
                        accuracy_part = parts[1].split("(")[0].strip()  # 去掉轮次信息
                        return float(accuracy_part)
                except:
                    continue
        
        # 最后备用：提取测试准确率（旧格式兼容）
        for line in lines:
            if "测试准确率:" in line:
                try:
                    parts = line.split("测试准确率:")
                    if len(parts) > 1:
                        accuracy_str = parts[1].strip()
                        return float(accuracy_str)
                except:
                    continue
        
        return None
    
    def run_all_experiments(self):
        """运行所有实验"""
        total_experiments = len(self.experiment_configs)
        print(f"开始批量实验: 共 {total_experiments} 个实验")
        print(f"输出目录: {self.output_dir}")
        
        start_time = time.time()
        
        for i, config in enumerate(self.experiment_configs, 1):
            print(f"\n进度: {i}/{total_experiments}")
            result = self.run_single_experiment(config)
            
            # 实时保存结果
            self._save_partial_results()
            
        total_duration = time.time() - start_time
        print(f"\n所有实验完成! 总时间: {total_duration/3600:.1f}小时")
        
        # 生成最终报告
        self._generate_final_report()
    
    def _save_partial_results(self):
        """保存部分结果"""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(self.results_file, index=False)
    
    def _generate_final_report(self):
        """生成最终报告"""
        print(f"\n{'='*60}")
        print("生成最终报告...")
        print(f"{'='*60}")
        
        if not self.results:
            print("没有结果可报告")
            return
        
        df = pd.DataFrame(self.results)
        
        # 保存完整结果
        df.to_csv(self.results_file, index=False)
        print(f"详细结果保存: {self.results_file}")
        
        # 生成摘要
        self._generate_summary_report(df)
        
        # 生成可视化
        self._generate_visualizations(df)
        
        # 打印主要结果
        self._print_main_results(df)
    
    def _generate_summary_report(self, df):
        """生成文字摘要报告"""
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write("AutoCPD 批量实验报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 实验概览
            f.write("实验概览:\n")
            f.write(f"总实验数: {len(df)}\n")
            f.write(f"成功实验: {len(df[df['status'] == '成功'])}\n")
            f.write(f"失败实验: {len(df[df['status'] != '成功'])}\n")
            f.write(f"平均训练时间: {df['duration_minutes'].mean():.1f}分钟\n\n")
            
            # 成功实验的准确率统计
            successful_df = df[df['status'] == '成功'].copy()
            if len(successful_df) > 0:
                f.write("准确率统计 (仅成功实验):\n")
                f.write(f"平均准确率: {successful_df['best_accuracy'].mean():.4f}\n")
                f.write(f"最高准确率: {successful_df['best_accuracy'].max():.4f}\n")
                f.write(f"最低准确率: {successful_df['best_accuracy'].min():.4f}\n")
                f.write(f"标准差: {successful_df['best_accuracy'].std():.4f}\n\n")
                
                # 按模型分组
                f.write("按模型分组的平均准确率:\n")
                model_stats = successful_df.groupby('model')['best_accuracy'].agg(['mean', 'std', 'count'])
                for model in model_stats.index:
                    stats = model_stats.loc[model]
                    f.write(f"  {model:12}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})\n")
                f.write("\n")
                
                # 按分类数分组
                f.write("按分类数分组的平均准确率:\n")
                class_stats = successful_df.groupby('classes')['best_accuracy'].agg(['mean', 'std', 'count'])
                for classes in class_stats.index:
                    stats = class_stats.loc[classes]
                    f.write(f"  {classes}分类: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})\n")
                f.write("\n")
                
                # 按维度分组
                f.write("按维度分组的平均准确率:\n")
                dim_stats = successful_df.groupby('dimension')['best_accuracy'].agg(['mean', 'std', 'count'])
                for dim in dim_stats.index:
                    stats = dim_stats.loc[dim]
                    f.write(f"  {dim}维: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})\n")
                f.write("\n")
                
                # 最佳实验
                best_exp = successful_df.loc[successful_df['best_accuracy'].idxmax()]
                f.write("最佳实验:\n")
                f.write(f"  实验ID: {best_exp['exp_id']}\n")
                f.write(f"  模型: {best_exp['model']}\n")
                f.write(f"  分类数: {best_exp['classes']}\n")
                f.write(f"  维度: {best_exp['dimension']}\n")
                f.write(f"  准确率: {best_exp['best_accuracy']:.4f}\n")
                f.write(f"  训练时间: {best_exp['duration_minutes']:.1f}分钟\n\n")
            
            # 失败实验
            failed_df = df[df['status'] != '成功']
            if len(failed_df) > 0:
                f.write("失败实验:\n")
                for _, row in failed_df.iterrows():
                    f.write(f"  {row['exp_id']}: {row['status']}\n")
        
        print(f"摘要报告保存: {self.summary_file}")
    
    def _generate_visualizations(self, df):
        """生成可视化图表"""
        successful_df = df[df['status'] == '成功'].copy()
        if len(successful_df) == 0:
            print("没有成功的实验，跳过可视化")
            return
        
        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. 总体准确率比较
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AutoCPD 方法比较实验结果', fontsize=16, fontweight='bold')
        
        # 1.1 按模型分组的准确率
        ax1 = axes[0, 0]
        model_acc = successful_df.groupby('model')['best_accuracy'].mean().sort_values(ascending=True)
        model_acc.plot(kind='barh', ax=ax1, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_title('各模型平均准确率', fontweight='bold')
        ax1.set_xlabel('准确率')
        ax1.grid(axis='x', alpha=0.3)
        
        # 1.2 按分类数分组的准确率
        ax2 = axes[0, 1]
        sns.boxplot(data=successful_df, x='classes', y='best_accuracy', ax=ax2)
        ax2.set_title('不同分类数的准确率分布', fontweight='bold')
        ax2.set_xlabel('分类数')
        ax2.set_ylabel('准确率')
        ax2.grid(axis='y', alpha=0.3)
        
        # 1.3 按维度分组的准确率
        ax3 = axes[1, 0]
        sns.boxplot(data=successful_df, x='dimension', y='best_accuracy', ax=ax3)
        ax3.set_title('不同维度的准确率分布', fontweight='bold')
        ax3.set_xlabel('数据维度')
        ax3.set_ylabel('准确率')
        ax3.grid(axis='y', alpha=0.3)
        
        # 1.4 训练时间vs准确率
        ax4 = axes[1, 1]
        colors = {'simple': 'blue', 'deep': 'red', 'transformer': 'green'}
        for model in successful_df['model'].unique():
            model_data = successful_df[successful_df['model'] == model]
            ax4.scatter(model_data['duration_minutes'], model_data['best_accuracy'], 
                       label=model, alpha=0.7, color=colors.get(model, 'gray'))
        ax4.set_title('训练时间 vs 准确率', fontweight='bold')
        ax4.set_xlabel('训练时间 (分钟)')
        ax4.set_ylabel('准确率')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        overview_path = self.output_dir / "experiment_overview.png"
        plt.savefig(overview_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"总览图保存: {overview_path}")
        
        # 2. 详细热力图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('各模型在不同设置下的准确率热力图', fontsize=16, fontweight='bold')
        
        models = ['simple', 'deep', 'transformer']
        for i, model in enumerate(models):
            model_data = successful_df[successful_df['model'] == model]
            if len(model_data) > 0:
                # 创建透视表
                pivot_table = model_data.pivot_table(
                    values='best_accuracy', 
                    index='dimension', 
                    columns='classes',
                    aggfunc='mean'
                )
                
                # 绘制热力图
                sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd',
                           ax=axes[i], cbar_kws={'label': '准确率'})
                axes[i].set_title(f'{model.title()} 网络', fontweight='bold')
                axes[i].set_xlabel('分类数')
                axes[i].set_ylabel('数据维度')
        
        plt.tight_layout()
        heatmap_path = self.output_dir / "accuracy_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"热力图保存: {heatmap_path}")
        
        # 3. 准确率排行榜
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 创建实验标识符
        successful_df['exp_label'] = (successful_df['model'] + '_' + 
                                    successful_df['classes'].astype(str) + 'c_' + 
                                    successful_df['dimension'].astype(str) + 'd')
        
        # 按准确率排序
        sorted_df = successful_df.sort_values('best_accuracy', ascending=True)
        
        # 绘制条形图
        bars = ax.barh(range(len(sorted_df)), sorted_df['best_accuracy'])
        
        # 设置颜色
        for i, (_, row) in enumerate(sorted_df.iterrows()):
            if row['model'] == 'simple':
                bars[i].set_color('skyblue')
            elif row['model'] == 'deep':
                bars[i].set_color('lightcoral')
            else:  # transformer
                bars[i].set_color('lightgreen')
        
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df['exp_label'])
        ax.set_xlabel('准确率')
        ax.set_title('所有实验准确率排行榜', fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='skyblue', label='Simple NN'),
                          Patch(facecolor='lightcoral', label='Deep NN'),
                          Patch(facecolor='lightgreen', label='Transformer')]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        ranking_path = self.output_dir / "accuracy_ranking.png"
        plt.savefig(ranking_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"排行榜保存: {ranking_path}")
    
    def _print_main_results(self, df):
        """打印主要结果到控制台"""
        successful_df = df[df['status'] == '成功'].copy()
        
        print(f"\n{'='*60}")
        print("主要实验结果")
        print(f"{'='*60}")
        
        if len(successful_df) == 0:
            print("没有成功的实验！")
            return
        
        # 总体统计
        print(f"成功实验数: {len(successful_df)}/{len(df)}")
        print(f"平均准确率: {successful_df['best_accuracy'].mean():.4f}")
        print(f"最高准确率: {successful_df['best_accuracy'].max():.4f}")
        print(f"平均训练时间: {successful_df['duration_minutes'].mean():.1f}分钟")
        
        # 最佳实验
        best_exp = successful_df.loc[successful_df['best_accuracy'].idxmax()]
        print(f"\n🏆 最佳实验:")
        print(f"   {best_exp['exp_id']} - 准确率: {best_exp['best_accuracy']:.4f}")
        
        # 按模型排行
        print(f"\n📊 模型排行:")
        model_ranking = successful_df.groupby('model')['best_accuracy'].mean().sort_values(ascending=False)
        for i, (model, acc) in enumerate(model_ranking.items(), 1):
            print(f"   {i}. {model:12}: {acc:.4f}")
        
        # 最佳配置
        print(f"\n🎯 各模型最佳配置:")
        for model in successful_df['model'].unique():
            model_data = successful_df[successful_df['model'] == model]
            best_model_exp = model_data.loc[model_data['best_accuracy'].idxmax()]
            print(f"   {model:12}: {best_model_exp['classes']}分类, "
                  f"{best_model_exp['dimension']}维 - {best_model_exp['best_accuracy']:.4f}")
        
        print(f"\n{'='*60}")


def main():
    """主函数"""
    print("AutoCPD 批量实验脚本")
    print("比较 Simple NN, Deep NN, Transformer 在不同设置下的表现")
    
    # 创建批量实验对象
    batch_exp = BatchExperiment(
        output_dir="batch_experiment_results_transformer", #####################
        base_args={
            'samples': 1500,  # 样本数量
            'length': 400,   # 时间序列长度
            'batch_size': 32,
            'seed': 2022,
            'verbose': 0,    # 减少输出
            'validation_split': 0.2,
            'gpu': '0'
        }
    )
    
    print(f"\n实验配置:")
    print(f"- 模型: simple, deep (6通道), transformer (transpose)")
    print(f"- 分类数: 3, 5") 
    print(f"- 维度: 1, 5, 8")
    print(f"- 样本数: 200, 序列长度: 200")
    print(f"- 总实验数: {len(batch_exp.experiment_configs)}")
    
    # 询问用户是否继续
    try:
        response = input(f"\n是否开始批量实验? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("实验取消")
            return
    except KeyboardInterrupt:
        print("\n实验取消")
        return
    
    # 运行批量实验
    try:
        batch_exp.run_all_experiments()
    except KeyboardInterrupt:
        print("\n实验被中断")
        batch_exp._generate_final_report()
    except Exception as e:
        print(f"\n实验过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        batch_exp._generate_final_report()


if __name__ == "__main__":
    main() 