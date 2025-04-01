import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv('/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouzhejian-240108120128/Mol/e3_diffusion_for_molecules/model_rebuttal.csv')  # 替换为你的文件路径


# 设置全局样式
sns.set(style="whitegrid", context="talk", font_scale=1.1)
# 提取模型编号作为横坐标
df['Model Number'] = df['Model Name'].str.extract(r'(\d+)_generative_model.npy').astype(int)

# 指标和颜色
metrics = ['Mol Stability', 'Atom Stability', 'Validity', 'Uniqueness', 'Novelty']
colors = sns.color_palette("Set2", len(metrics))  # 柔和配色

# 提取模型编号
df['Model Number'] = df['Model Name'].str.extract(r'(\d+)_generative_model.npy').astype(int)

# 指标和颜色
metrics = ['Mol Stability', 'Atom Stability', 'Validity', 'Uniqueness', 'Novelty']
colors = sns.color_palette("Set2", len(metrics))

# 你的基准线值
baselines = {
    'Mol Stability': 82.0 / 100,
    'Atom Stability': 98.7 / 100,
    'Validity': 91.9 / 100,
    'Uniqueness': 90.7 / 100,
    'Novelty': 65.7 / 100,
}

# 绘图
# 逐组画图
for path, group in df.groupby('Model Path'):
    group = group.sort_values('Model Number')

    # 创建图和轴对象，留出右边空间
    fig, ax = plt.subplots(figsize=(15, 6))

    for i, metric in enumerate(metrics):
        ax.plot(group['Model Number'], group[metric],
                label=metric,
                color=colors[i],
                linewidth=2.5,
                marker='o',
                markersize=6)

        # 添加基线（虚线）
        ax.axhline(baselines[metric], color=colors[i], linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'The {metric} of EDM')

    # 标题与坐标轴
    model_name = path.split('/')[-1]
    ax.set_title(f'Training Metric Trends across Epochs (EDPO Fine-Tuning Phase & Reward : Force QM)', fontsize=12, weight='bold')
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)

    # 坐标轴字体
    ax.tick_params(axis='both', labelsize=12)

    # 网格线
    ax.grid(alpha=0.3)

    # 图例放在图外右侧
    ax.legend(title='Metric & Baseline', fontsize=11, title_fontsize=12,
              loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)

    # 自动调整布局以防图例遮挡
    plt.tight_layout(rect=[0, 0, 0.82, 1])  # 留出右侧空间

    # 保存图像
    filename = f'{model_name}_metrics_with_baseline_outside_legend.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Saved with outside legend: {filename}")