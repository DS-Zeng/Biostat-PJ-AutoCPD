#!/usr/bin/env python3
"""
AutoCPD Transformer GPU 批量实验脚本
专门在GPU上运行Transformer实验，对比不同配置下的表现
Author: AI Assistant
Date: 2024-01-XX

实验设计:
- 模型: transformer (仅)
- 分类数: 3, 5
- 维度: 1, 5, 8
- 总计: 1 × 2 × 3 = 6 个Transformer实验
- 运行环境: GPU加速
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
import tensorflow as tf
warnings.filterwarnings('ignore')


class TransformerGPUBatchExperiment:
    """GPU Transformer批量实验管理类"""
    
    def __init__(self, output_dir="batch_experiment_results", base_args=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_dir / "transformer_gpu_experiment_results.csv"
        self.summary_file = self.output_dir / "transformer_gpu_experiment_summary.txt"
        
        # GPU优化的基础参数
        self.base_args = base_args or {
            'samples': 1500,     # 保持相同样本数
            'length': 400,       # 保持相同序列长度
            'epochs': None,      # 自动确定（GPU上可以训练更多轮）
            'batch_size': 128,   # GPU上可以使用更大批次
            'seed': 2022,
            'verbose': 1,        # 显示详细输出
            'validation_split': 0.2,
            'preset': 'basic',
            'gpu': '0'           # 明确使用GPU:0
        }
        
        # 实验配置 - 仅Transformer
        self.experiment_configs = self._generate_transformer_configs()
        
        # 结果存储
        self.results = []
        
        # 验证GPU可用性
        self._check_gpu_availability()
        
    def _check_gpu_availability(self):
        """检查GPU可用性"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if len(gpus) > 0:
                print(f"✅ 检测到 {len(gpus)} 块GPU")
                print(f"将使用GPU:{self.base_args['gpu']}")
                
                # 设置GPU内存增长
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU内存增长已启用")
            else:
                print("⚠️  未检测到GPU，将回退到CPU模式")
                self.base_args['gpu'] = 'cpu'
        except Exception as e:
            print(f"GPU检查失败: {e}")
            self.base_args['gpu'] = 'cpu'
        
    def _generate_transformer_configs(self):
        """生成仅Transformer的实验配置"""
        configs = []
        
        # 仅Transformer模型
        model = 'transformer'
        class_numbers = [3, 5]
        dimensions = [1, 5, 8]
        
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
                    'exp_id': f"transformer_{classes}c_{dim}d_gpu"
                }
                configs.append(config)
        
        return configs
    
    def run_single_experiment(self, config):
        """运行单个Transformer实验"""
        print(f"\n{'='*70}")
        print(f"🚀 GPU Transformer实验: {config['exp_id']}")
        print(f"分类: {config['classes']}, 维度: {config['dim']}")
        print(f"{'='*70}")
        
        # 构建命令
        cmd = [
            sys.executable, "train_configurable.py",
            "--model", config['model'],
            "--classes", str(config['classes']),
            "--dim", str(config['dim']),
            "--preset", config['preset'],
            "--output_dir", str(self.output_dir / "individual_results")
        ]
        
        # 添加GPU优化参数
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
        exp_name = f"gpu_transformer_{config['classes']}c_{config['dim']}d_{datetime.now().strftime('%H%M%S')}"
        cmd.extend(["--exp_name", exp_name])
        
        print(f"🔧 命令: {' '.join(cmd)}")
        
        # 运行实验
        start_time = time.time()
        try:
            print("⏱️  开始GPU训练...")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=7200  # 2小时超时（GPU训练应该很快）
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
                print(f"❌ 标准错误: {result.stderr[:500]}...")
                
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
            'best_accuracy': accuracy,
            'duration_minutes': duration / 60,
            'status': status,
            'error_msg': error_msg,
            'gpu_used': self.base_args['gpu'],
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(result_record)
        
        # 打印结果
        if accuracy is not None:
            print(f"✅ 成功! 最佳准确率: {accuracy:.4f}, 时间: {duration/60:.1f}分钟")
        else:
            print(f"❌ 失败: {status}, 时间: {duration/60:.1f}分钟")
            if error_msg:
                print(f"错误信息: {error_msg[:300]}...")
        
        return result_record
    
    def _extract_accuracy_from_output(self, output):
        """从输出中提取准确率"""
        lines = output.split('\n')
        
        # 尝试提取最佳准确率
        for line in lines:
            if "最佳准确率:" in line:
                try:
                    parts = line.split("最佳准确率:")
                    if len(parts) > 1:
                        accuracy_str = parts[1].strip()
                        return float(accuracy_str)
                except:
                    continue
        
        # 备用：提取最佳验证准确率
        for line in lines:
            if "最佳验证准确率:" in line:
                try:
                    parts = line.split("最佳验证准确率:")
                    if len(parts) > 1:
                        accuracy_part = parts[1].split("(")[0].strip()
                        return float(accuracy_part)
                except:
                    continue
        
        # 最后备用：提取测试准确率
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
        """运行所有Transformer实验"""
        total_experiments = len(self.experiment_configs)
        print(f"\n🎯 开始GPU Transformer批量实验")
        print(f"📊 总实验数: {total_experiments}")
        print(f"💾 输出目录: {self.output_dir}")
        print(f"🔥 GPU设备: {self.base_args['gpu']}")
        
        start_time = time.time()
        
        for i, config in enumerate(self.experiment_configs, 1):
            print(f"\n📈 进度: {i}/{total_experiments}")
            result = self.run_single_experiment(config)
            
            # 实时保存结果
            self._save_partial_results()
            
        total_duration = time.time() - start_time
        print(f"\n🎉 所有Transformer实验完成!")
        print(f"⏱️  总时间: {total_duration/3600:.1f}小时")
        
        # 生成最终报告
        self._generate_final_report()
    
    def _save_partial_results(self):
        """保存部分结果"""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(self.results_file, index=False)
    
    def _generate_final_report(self):
        """生成最终报告"""
        print(f"\n{'='*70}")
        print("📝 生成GPU Transformer实验报告...")
        print(f"{'='*70}")
        
        if not self.results:
            print("没有结果可报告")
            return
        
        df = pd.DataFrame(self.results)
        
        # 保存完整结果
        df.to_csv(self.results_file, index=False)
        print(f"💾 详细结果保存: {self.results_file}")
        
        # 生成摘要
        self._generate_summary_report(df)
        
        # 生成可视化
        self._generate_visualizations(df)
        
        # 打印主要结果
        self._print_main_results(df)
    
    def _generate_summary_report(self, df):
        """生成文字摘要报告"""
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write("AutoCPD GPU Transformer 批量实验报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"GPU设备: {self.base_args['gpu']}\n\n")
            
            # 实验概览
            f.write("实验概览:\n")
            f.write(f"总实验数: {len(df)}\n")
            f.write(f"成功实验: {len(df[df['status'] == '成功'])}\n")
            f.write(f"失败实验: {len(df[df['status'] != '成功'])}\n")
            f.write(f"平均训练时间: {df['duration_minutes'].mean():.1f}分钟\n\n")
            
            # 成功实验的准确率统计
            successful_df = df[df['status'] == '成功'].copy()
            if len(successful_df) > 0:
                f.write("Transformer准确率统计 (仅成功实验):\n")
                f.write(f"平均准确率: {successful_df['best_accuracy'].mean():.4f}\n")
                f.write(f"最高准确率: {successful_df['best_accuracy'].max():.4f}\n")
                f.write(f"最低准确率: {successful_df['best_accuracy'].min():.4f}\n")
                f.write(f"标准差: {successful_df['best_accuracy'].std():.4f}\n\n")
                
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
                f.write("最佳Transformer实验:\n")
                f.write(f"  实验ID: {best_exp['exp_id']}\n")
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
        
        print(f"📄 摘要报告保存: {self.summary_file}")
    
    def _generate_visualizations(self, df):
        """生成可视化图表"""
        successful_df = df[df['status'] == '成功'].copy()
        if len(successful_df) == 0:
            print("没有成功的实验，跳过可视化")
            return
        
        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("viridis")
        
        # 1. Transformer性能总览
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('GPU Transformer 实验结果分析', fontsize=16, fontweight='bold')
        
        # 1.1 按分类数分组的准确率
        ax1 = axes[0, 0]
        class_acc = successful_df.groupby('classes')['best_accuracy'].mean()
        class_acc.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('不同分类数的平均准确率', fontweight='bold')
        ax1.set_xlabel('分类数')
        ax1.set_ylabel('准确率')
        ax1.tick_params(axis='x', rotation=0)
        ax1.grid(axis='y', alpha=0.3)
        
        # 1.2 按维度分组的准确率
        ax2 = axes[0, 1]
        dim_acc = successful_df.groupby('dimension')['best_accuracy'].mean()
        dim_acc.plot(kind='bar', ax=ax2, color='lightcoral')
        ax2.set_title('不同维度的平均准确率', fontweight='bold')
        ax2.set_xlabel('数据维度')
        ax2.set_ylabel('准确率')
        ax2.tick_params(axis='x', rotation=0)
        ax2.grid(axis='y', alpha=0.3)
        
        # 1.3 训练时间分布
        ax3 = axes[1, 0]
        ax3.hist(successful_df['duration_minutes'], bins=8, alpha=0.7, color='lightgreen')
        ax3.set_title('训练时间分布', fontweight='bold')
        ax3.set_xlabel('训练时间 (分钟)')
        ax3.set_ylabel('实验数量')
        ax3.grid(axis='y', alpha=0.3)
        
        # 1.4 准确率vs训练时间散点图
        ax4 = axes[1, 1]
        scatter = ax4.scatter(successful_df['duration_minutes'], successful_df['best_accuracy'], 
                             c=successful_df['dimension'], cmap='viridis', alpha=0.7, s=100)
        ax4.set_title('准确率 vs 训练时间', fontweight='bold')
        ax4.set_xlabel('训练时间 (分钟)')
        ax4.set_ylabel('准确率')
        ax4.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='数据维度')
        
        plt.tight_layout()
        overview_path = self.output_dir / "transformer_gpu_overview.png"
        plt.savefig(overview_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 总览图保存: {overview_path}")
        
        # 2. 热力图
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        pivot_table = successful_df.pivot_table(
            values='best_accuracy', 
            index='dimension', 
            columns='classes',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlOrRd',
                   ax=ax, cbar_kws={'label': '准确率'})
        ax.set_title('GPU Transformer: 不同配置下的准确率热力图', fontweight='bold', pad=20)
        ax.set_xlabel('分类数')
        ax.set_ylabel('数据维度')
        
        plt.tight_layout()
        heatmap_path = self.output_dir / "transformer_gpu_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"🔥 热力图保存: {heatmap_path}")
        
    def _print_main_results(self, df):
        """打印主要结果到控制台"""
        successful_df = df[df['status'] == '成功'].copy()
        
        print(f"\n{'='*70}")
        print("🎯 GPU Transformer 实验主要结果")
        print(f"{'='*70}")
        
        if len(successful_df) == 0:
            print("❌ 没有成功的实验！")
            return
        
        # 总体统计
        print(f"✅ 成功实验数: {len(successful_df)}/{len(df)}")
        print(f"📊 平均准确率: {successful_df['best_accuracy'].mean():.4f}")
        print(f"🏆 最高准确率: {successful_df['best_accuracy'].max():.4f}")
        print(f"⏱️  平均训练时间: {successful_df['duration_minutes'].mean():.1f}分钟")
        
        # 最佳实验
        best_exp = successful_df.loc[successful_df['best_accuracy'].idxmax()]
        print(f"\n🥇 最佳Transformer实验:")
        print(f"   {best_exp['exp_id']} - 准确率: {best_exp['best_accuracy']:.4f}")
        print(f"   配置: {best_exp['classes']}分类, {best_exp['dimension']}维")
        print(f"   训练时间: {best_exp['duration_minutes']:.1f}分钟")
        
        # 按分类数和维度的最佳配置
        print(f"\n📈 各配置最佳结果:")
        for classes in sorted(successful_df['classes'].unique()):
            for dim in sorted(successful_df['dimension'].unique()):
                subset = successful_df[(successful_df['classes'] == classes) & 
                                     (successful_df['dimension'] == dim)]
                if len(subset) > 0:
                    best_acc = subset['best_accuracy'].max()
                    avg_time = subset['duration_minutes'].mean()
                    print(f"   {classes}分类-{dim}维: {best_acc:.4f} (平均时间: {avg_time:.1f}分钟)")
        
        print(f"\n{'='*70}")


def main():
    """主函数"""
    print("🚀 AutoCPD GPU Transformer 批量实验脚本")
    print("专门在GPU上运行Transformer实验，提升训练速度")
    
    # 创建GPU Transformer批量实验对象
    batch_exp = TransformerGPUBatchExperiment(
        output_dir="batch_experiment_results",  # 保持原有路径
        base_args={
            'samples': 1500,     # 保持样本数
            'length': 400,       # 保持序列长度
            'batch_size': 128,   # GPU上使用更大批次
            'seed': 2022,
            'verbose': 1,        # 显示训练过程
            'validation_split': 0.2,
            'gpu': '0'           # 使用第一块GPU
        }
    )
    
    print(f"\n📋 实验配置:")
    print(f"🤖 模型: transformer (仅)")
    print(f"📊 分类数: 3, 5") 
    print(f"📏 维度: 1, 5, 8")
    print(f"🔢 总实验数: {len(batch_exp.experiment_configs)}")
    print(f"🔥 GPU设备: {batch_exp.base_args['gpu']}")
    
    # 询问用户是否继续
    try:
        response = input(f"\n🤔 是否开始GPU Transformer批量实验? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ 实验取消")
            return
    except KeyboardInterrupt:
        print("\n❌ 实验取消")
        return
    
    # 运行批量实验
    try:
        batch_exp.run_all_experiments()
    except KeyboardInterrupt:
        print("\n⚠️  实验被中断")
        batch_exp._generate_final_report()
    except Exception as e:
        print(f"\n❌ 实验过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        batch_exp._generate_final_report()


if __name__ == "__main__":
    main() 