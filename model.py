# ==============================================================
# GA-CV-SHAP Workflow ( Final Model + Explainability)
# 增加ga筛选出的2特征值模型和4特征值模型对比
# Author: [ccl]
# Date: [2025-10-21]
# ==============================================================

# Required Libraries
# pip install numpy pandas scikit-learn lightgbm shap deap matplotlib seaborn openpyxl

import os
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold,
    train_test_split, cross_val_score
)
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay
)
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from deap import base, creator, tools, algorithms
import random, warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="The NumPy global RNG was seeded by calling `np.random.seed`")
from copy import deepcopy
import matplotlib

# --- ANSI 转义码 (用于控制台颜色) ---
RED_START = '\033[91m'
COLOR_RESET = '\033[0m'

# 全局设置
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 8
DESIRED_DPI = 600
matplotlib.rcParams['savefig.dpi'] = DESIRED_DPI
# [--- 请将以下5行代码添加到您的全局设置中 ---]

matplotlib.rcParams['axes.linewidth'] = 0.5       # 设置坐标轴边框（Spines）宽度
matplotlib.rcParams['xtick.major.width'] = 0.5  # 设置 X 轴主刻度线宽度
matplotlib.rcParams['ytick.major.width'] = 0.5  # 设置 Y 轴主刻度线宽度
matplotlib.rcParams['xtick.minor.width'] = 0.5  # 设置 X 轴次刻度线宽度
matplotlib.rcParams['ytick.minor.width'] = 0.5  # 设置 Y 轴次刻度线宽度

# [--- 添加结束 ---]
# ========== 保存路径 ==========
SAVE_DIR = "/Volumes/T7/pycharm/python subject/体系一/Model_Results-改尺寸-10.27-shap"
os.makedirs(SAVE_DIR, exist_ok=True)


# ==============================================================
# 辅助函数：创建 Figure 和 Axes，直接控制坐标轴区域的物理尺寸 (英寸)
# ==============================================================
def _create_axes_with_plot_area_in(plot_width_in, plot_height_in,
                                   left_margin_in=0.6, right_margin_in=0.2,
                                   top_margin_in=0.4, bottom_margin_in=0.5,
                                   dpi=DESIRED_DPI):
    """
    创建一个 Figure 和 Axes。Axes 的物理尺寸直接由 plot_width_in 和 plot_height_in 控制 (英寸)。
    边距用于容纳标题、坐标轴标签等 (英寸)。

    参数:
        plot_width_in (float): 绘图区域 (Axes) 的所需宽度（英寸）。
        plot_height_in (float): 绘图区域 (Axes) 的所需高度（英寸）。
        left_margin_in (float): Figure 左边距（英寸），用于 Y 轴标签等。
        right_margin_in (float): Figure 右边距（英寸）。
        top_margin_in (float): Figure 上边距（英寸），用于标题等。
        bottom_margin_in (float): Figure 下边距（英寸），用于 X 轴标签等。
        dpi (int): 输出图像的 DPI。

    返回:
        tuple: (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    figure_width_inches = plot_width_in + left_margin_in + right_margin_in
    figure_height_inches = plot_height_in + top_margin_in + bottom_margin_in

    fig = plt.figure(figsize=(figure_width_inches, figure_height_inches), dpi=dpi)

    axes_left = left_margin_in / figure_width_inches
    axes_bottom = bottom_margin_in / figure_height_inches
    axes_width = plot_width_in / figure_width_inches
    axes_height = plot_height_in / figure_height_inches

    ax = fig.add_axes([axes_left, axes_bottom, axes_width, axes_height])

    return fig, ax


# ==============================================================
# 1. 数据导入与手动特征列选择
# ==============================================================

# TODO: 修改为你的 Excel 文件路径
file_path = "ALL.csv"

# 读取数据
df = pd.read_csv(file_path)

print("✅ 数据加载成功！前5行预览：")
print(df.head(), "\n")

# TODO: 指定你的特征列名和标签列名
feature_cols = ['pH', 'Glucose', 'Lactate', 'Urea']  # ← 手动指定
label_col = 'Fatigued'  # ← 手动指定

# 从 DataFrame 提取特征与标签
X = df[feature_cols].values
y = df[label_col].values

# 检查数据维度
print(f"特征数: {X.shape[1]}, 样本数: {X.shape[0]}, 标签唯一值: {np.unique(y)}\n")

# 固定随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ==============================================================
# 2. 数据划分（Lock-box策略）
# ==============================================================

X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
)
print(f"开发集样本: {X_dev.shape[0]}, 测试集样本: {X_test.shape[0]}\n")

import pandas as pd

print(f"!!! 关键诊断：开发集 (y_dev) 的实际类别分布：\n{pd.Series(y_dev).value_counts()}\n")


# ==============================================================
# 3. 定义 嵌套CV + AUC评估函数（适应度）
# ==============================================================

def nested_cv_auc(X_subset, y_subset, random_state=0):
    outer_cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=2, random_state=random_state)
    aucs = []
    for train_idx, val_idx in outer_cv.split(X_subset, y_subset):
        X_train, X_val = X_subset[train_idx], X_subset[val_idx]
        y_train, y_val = y_subset[train_idx], y_subset[val_idx]

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, random_state=random_state)
        )
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]

        try:
            aucs.append(roc_auc_score(y_val, preds))
        except ValueError:
            aucs.append(0.5)

    return np.mean(aucs)


# ==============================================================
# 4. 定义 GA 搜索（遗传算法特征选择）
# ==============================================================

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def run_ga(X_dev, y_dev, random_state=42, n_gen=80, pop_size=60, mutation_prob=0.01):
    n_features = X_dev.shape[1]

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_individual(individual):
        if sum(individual) == 0:
            return (0,)
        selected_idx = [i for i, bit in enumerate(individual) if bit == 1]
        X_sel = X_dev[:, selected_idx]
        score = nested_cv_auc(X_sel, y_dev, random_state)
        return (score,)

    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.8)
    toolbox.register("mutate", tools.mutFlipBit, indpb=mutation_prob)
    toolbox.register("select", tools.selTournament, tournsize=3)

    random.seed(random_state)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(2)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=mutation_prob,
                        ngen=n_gen, stats=stats, halloffame=hof, verbose=False)

    best_ind = hof[0]
    return np.array(best_ind), hof[0].fitness.values[0]


# ==============================================================
# 5. 多次GA运行 + 稳定性分析
# ==============================================================

n_runs = 10
best_sets = []
scores = []

for run_seed in range(n_runs):
    print(f"Running GA {run_seed + 1}/{n_runs} ...")
    best_mask, best_score = run_ga(X_dev, y_dev, random_state=run_seed)
    best_sets.append(best_mask)
    scores.append(best_score)

best_sets = np.array(best_sets)
feature_freq = best_sets.mean(axis=0)

print(f"!!! 诊断：特征选择的实际频率分布：")
print(pd.Series(feature_freq, index=feature_cols))

# 选择稳定特征（≥0.6频率）
stable_features = np.where(feature_freq >= 0.6)[0]
print(f"稳定特征数量: {len(stable_features)}")
print("稳定特征列表:", [feature_cols[i] for i in stable_features], "\n")

# ==============================================================
# 配置板块：GA特征稳定性图（Selection Frequency）
# ==============================================================
# 设定此图的【坐标轴区域】的物理尺寸（英寸）
GA_FEATURE_STABILITY_PLOT_CONFIG = {
    "plot_width_in": 1.6,  # 绘图区域宽度（英寸），约 12cm
    "plot_height_in": 1.2,  # 绘图区域高度（英寸），约 6cm
    "left_margin_in": 0.8,  # Figure 左边距（英寸）
    "right_margin_in": 0.2,  # Figure 右边距（英寸）
    "top_margin_in": 0.4,  # Figure 上边距（英寸）
    "bottom_margin_in": 0.7  # Figure 下边距（英寸），为 X 轴标签预留
}
# ==============================================================

# 可视化特征选择频率
fig_ga_freq, ax_ga_freq = _create_axes_with_plot_area_in(
    plot_width_in=GA_FEATURE_STABILITY_PLOT_CONFIG["plot_width_in"],
    plot_height_in=GA_FEATURE_STABILITY_PLOT_CONFIG["plot_height_in"],
    left_margin_in=GA_FEATURE_STABILITY_PLOT_CONFIG["left_margin_in"],
    right_margin_in=GA_FEATURE_STABILITY_PLOT_CONFIG["right_margin_in"],
    top_margin_in=GA_FEATURE_STABILITY_PLOT_CONFIG["top_margin_in"],
    bottom_margin_in=GA_FEATURE_STABILITY_PLOT_CONFIG["bottom_margin_in"]
)
sns.barplot(x=feature_cols, y=feature_freq, lw=0.5, edgecolor='black', ax=ax_ga_freq)
ax_ga_freq.set_ylabel("Selection Frequency")
ax_ga_freq.set_title("Feature Stability Across GA Runs")
ax_ga_freq.tick_params(axis='x', rotation=45)

fig_ga_freq.savefig(os.path.join(SAVE_DIR, "GA_Feature_Stability.png"), bbox_inches='tight')
plt.close(fig_ga_freq)

# ==============================================================
# 6. [替换] 定义用于对比的两个模型
# ==============================================================

# 模型 A: GA-Optimal (来自步骤5的稳定特征)
ga_features_idx = stable_features
ga_feature_names = [feature_cols[i] for i in ga_features_idx]
print(f"定义 模型 A (GA-Optimal, {len(ga_feature_names)} features): {ga_feature_names}")

# 模型 B: Full-Model (全部特征)
full_features_idx = np.arange(len(feature_cols))
full_feature_names = [feature_cols[i] for i in full_features_idx]
print(f"定义 模型 B (Full-Model, {len(full_feature_names)} features): {full_feature_names}\n")

# 定义模型参数 (必须与步骤3的GA评估时一致)
model_pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=0.1, random_state=RANDOM_SEED)
)

models_to_test = {
    "GA-Optimal": (ga_features_idx, ga_feature_names),
    "Full-Model": (full_features_idx, full_feature_names)
}

final_results = {}
final_models = {}

# ==============================================================
# 7. [替换] 训练并测试两个模型 (手动缩放并保存 dict)
# ==============================================================
print("\n=== 开始对比评估 (在 Test Set, n=12) ===")

final_results = {}
final_models = {}

for model_name, (indices, names) in models_to_test.items():

    # --- 6. 训练 (手动缩放) ---
    X_train_final = X_dev[:, indices]

    scaler_final = StandardScaler()
    X_train_final_scaled = scaler_final.fit_transform(X_train_final)

    model_final = LogisticRegression(C=0.1, random_state=RANDOM_SEED)
    model_final.fit(X_train_final_scaled, y_dev)

    final_models[model_name] = {"model": model_final, "scaler": scaler_final}

    # --- 7. 评估 (手动缩放) ---
    X_test_final = X_test[:, indices]
    X_test_final_scaled = final_models[model_name]["scaler"].transform(X_test_final)

    y_pred_prob = final_models[model_name]["model"].predict_proba(X_test_final_scaled)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)

    auc_score = roc_auc_score(y_test, y_pred_prob)

    # (Bootstrap CI)
    boot_aucs = []
    for _ in range(1000):
        idx = resample(range(len(y_test)))
        # 检查重采样后的样本是否包含至少两个类别，否则跳过
        if len(np.unique(np.array(y_test)[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(np.array(y_test)[idx], np.array(y_pred_prob)[idx]))

    # 确保 boot_aucs 不为空，以避免百分位数计算错误
    if boot_aucs:
        ci_lower, ci_upper = np.percentile(boot_aucs, [2.5, 97.5])
    else:
        ci_lower, ci_upper = np.nan, np.nan  # 如果没有有效AUC，则设为NaN

    print(f"--- 模型: {model_name} ---")
    print(f"  测试集 AUC: {auc_score:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
    print(f"  测试集 F1-Score: {f1_score(y_test, y_pred):.3f}")

    final_results[model_name] = auc_score

# ==============================================================
# 配置板块：ROC 曲线对比图
# ==============================================================
# 设定此图的【坐标轴区域】的物理尺寸（英寸）
ROC_CURVE_PLOT_CONFIG = {
    "plot_width_in": 1.6,  # 绘图区域宽度（英寸），约 10cm
    "plot_height_in": 1.2,  # 绘图区域高度（英寸），约 9cm
    "left_margin_in": 0.6,  # Figure 左边距（英寸）
    "right_margin_in": 0.2,  # Figure 右边距（英寸）
    "top_margin_in": 0.4,  # Figure 上边距（英寸）
    "bottom_margin_in": 0.5  # Figure 下边距（英寸）
}
# ==============================================================

# 绘制ROC曲线
fig_roc, ax_roc = _create_axes_with_plot_area_in(
    plot_width_in=ROC_CURVE_PLOT_CONFIG["plot_width_in"],
    plot_height_in=ROC_CURVE_PLOT_CONFIG["plot_height_in"],
    left_margin_in=ROC_CURVE_PLOT_CONFIG["left_margin_in"],
    right_margin_in=ROC_CURVE_PLOT_CONFIG["right_margin_in"],
    top_margin_in=ROC_CURVE_PLOT_CONFIG["top_margin_in"],
    bottom_margin_in=ROC_CURVE_PLOT_CONFIG["bottom_margin_in"]
)

for model_name, (indices, _) in models_to_test.items():
    model_data = final_models[model_name]
    model_obj = model_data["model"]
    scaler_obj = model_data["scaler"]
    X_test_scaled = scaler_obj.transform(X_test[:, indices])
    y_pred_prob = model_obj.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, lw=0.5, label=f'{model_name} (AUC = {roc_auc:.3f})')

ax_roc.plot([0, 1], [0, 1], color='navy', lw=0.5, linestyle='--')
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate (FPR)')
ax_roc.set_ylabel('True Positive Rate (TPR)')
ax_roc.set_title('ROC Curve Comparison on Test Set')
ax_roc.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

fig_roc.savefig(os.path.join(SAVE_DIR, "ROC_Curve_Comparison.png"), bbox_inches='tight')
plt.show()

# ==============================================================
# 配置板块：混淆矩阵图
# ==============================================================
# 设定此图的【坐标轴区域】的物理尺寸（英寸）
CONFUSION_MATRIX_PLOT_CONFIG = {
    "plot_width_in": 1.2,  # 绘图区域宽度（英寸）
    "plot_height_in": 1.2,  # 绘图区域高度（英寸）
    "left_margin_in": 0.8,  # Figure 左边距（英寸），为 Y 轴标签预留
    "right_margin_in": 0.2,  # Figure 右边距（英寸）
    "top_margin_in": 0.4,  # Figure 上边距（英寸）
    "bottom_margin_in": 0.7  # Figure 下边距（英寸），为 X 轴标签预留
}
# ==============================================================

# 绘制 Full-Model 的混淆矩阵
fig_cm, ax_cm = _create_axes_with_plot_area_in(
    plot_width_in=CONFUSION_MATRIX_PLOT_CONFIG["plot_width_in"],
    plot_height_in=CONFUSION_MATRIX_PLOT_CONFIG["plot_height_in"],
    left_margin_in=CONFUSION_MATRIX_PLOT_CONFIG["left_margin_in"],
    right_margin_in=CONFUSION_MATRIX_PLOT_CONFIG["right_margin_in"],
    top_margin_in=CONFUSION_MATRIX_PLOT_CONFIG["top_margin_in"],
    bottom_margin_in=CONFUSION_MATRIX_PLOT_CONFIG["bottom_margin_in"]
)

model_full_data = final_models["Full-Model"]
model_full_obj = model_full_data["model"]
scaler_full_obj = model_full_data["scaler"]
X_test_full_scaled_cm = scaler_full_obj.transform(X_test[:, full_features_idx])
y_pred_full_cm = (model_full_obj.predict_proba(X_test_full_scaled_cm)[:, 1] > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred_full_cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues, ax=ax_cm, values_format='d', colorbar=False)

ax_cm.set_title('Confusion Matrix of Full-Model (n=12)')
ax_cm.set_xlabel('Predicted Label')
ax_cm.set_ylabel('True Label')

fig_cm.savefig(os.path.join(SAVE_DIR, "Confusion_Matrix_Full_Model.png"), bbox_inches='tight')
plt.show()

# ==============================================================
# 8. SHAP 可解释性分析 (训练集 + 测试集)
# ==============================================================
print("\n=== 开始 SHAP 可解释性分析 (Train & Test) ===")

model_to_explain_data = final_models["Full-Model"]
model_to_explain = model_to_explain_data["model"]
scaler_to_explain = model_to_explain_data["scaler"]

X_train_shap_unscaled = X_dev[:, full_features_idx]
X_test_shap_unscaled = X_test[:, full_features_idx]
feature_names_shap = full_feature_names  # 修改为 feature_names_shap，避免与全局feature_cols混淆

# --- 使用缩放后的数据 ---
X_train_shap_scaled = scaler_to_explain.transform(X_train_shap_unscaled)
background_scaled = shap.sample(X_train_shap_scaled, 30, random_state=RANDOM_SEED)

# --- 使用 LinearExplainer ---
explainer = shap.LinearExplainer(model_to_explain, background_scaled)
print("✅ SHAP LinearExplainer 创建成功。")


# ==============================================================
# 辅助函数: 统一可视化风格与保存逻辑 (SHAP 图)
# [版本 3: 修正了 Beeswarm, Bar, Decision 图的尺寸控制]
# ==============================================================

def plot_all_shap_figures(shap_values, X_data, feature_names_for_plot, dataset_name="Train", save_dir=None):
    print(f"\n--- 绘制 {dataset_name}集 SHAP 可视化 ---")

    def _save_shap_plot(filename, fig_object):
        if save_dir:
            path = os.path.join(save_dir, f"SHAP_{dataset_name}_{filename}.png")
            try:
                fig_object.savefig(path, bbox_inches='tight')
                print(f"  > 已保存: {path}")
            except Exception as e:
                print(f"  ⚠️ 保存图表失败 ({filename}): {e}")
            plt.close(fig_object)

    # (Beeswarm, Bar, Heatmap 部分保持不变)
    # ==============================================================
    # 配置板块：SHAP Beeswarm Plot
    # [!] 已修改：改为控制【坐标轴区域】尺寸
    # ==============================================================
    SHAP_BEESWARM_PLOT_CONFIG = {
        "plot_width_in": 2.2,  # 绘图区域宽度（英寸）
        "plot_height_in": 1.4,  # 绘图区域高度（英寸）
        "left_margin_in": 0.6,  # Figure 左边距（英寸），为 Y 轴标签 (Lactate) 预留
        "right_margin_in": 0.7,  # Figure 右边距（英寸）
        "top_margin_in": 0.4,  # Figure 上边距（英寸），为标题预留
        "bottom_margin_in": 0.5  # Figure 下边距（英寸），为 X 轴标签 (SHAP value) 预留
    }
    # ==============================================================
    # 1. 从配置中读取尺寸
    plot_width_in = SHAP_BEESWARM_PLOT_CONFIG["plot_width_in"]
    plot_height_in = SHAP_BEESWARM_PLOT_CONFIG["plot_height_in"]
    left_margin = SHAP_BEESWARM_PLOT_CONFIG["left_margin_in"]
    right_margin = SHAP_BEESWARM_PLOT_CONFIG["right_margin_in"]
    top_margin = SHAP_BEESWARM_PLOT_CONFIG["top_margin_in"]
    bottom_margin = SHAP_BEESWARM_PLOT_CONFIG["bottom_margin_in"]

    # 2. 计算总 Figure 尺寸
    figure_width_inches = plot_width_in + left_margin + right_margin
    figure_height_inches = plot_height_in + top_margin + bottom_margin

    # 3. 计算 *主绘图区* Axes 相对位置 ([left, bottom, width, height])
    axes_left = left_margin / figure_width_inches
    axes_bottom = bottom_margin / figure_height_inches
    axes_width = plot_width_in / figure_width_inches
    axes_height = plot_height_in / figure_height_inches

    # 4. [关键] 先让 SHAP 绘图 (带 color_bar=True)
    shap.summary_plot(shap_values, X_data, feature_names=feature_names_for_plot, show=False, color_bar=True)  # [修改!]

    # 5. [关键] 获取 SHAP 创建的 Figure 和 *所有* Axes
    fig_beeswarm = plt.gcf()
    all_axes = fig_beeswarm.get_axes()

    # 6. [关键] 将 Figure 尺寸重设为我们计算的
    fig_beeswarm.set_size_inches(figure_width_inches, figure_height_inches)

    # 7. [关键] 识别并重设 *主图* 位置
    ax_beeswarm = all_axes[0]
    ax_beeswarm.set_position([axes_left, axes_bottom, axes_width, axes_height])

    # 8. [关键] 识别并重设 *Color Bar* 位置
    ax_colorbar = all_axes[1]

    # (计算 color bar 的位置: 放在主图右侧，留 0.1 英寸空隙，宽度 0.15 英寸)
    cbar_gap_inches = 0.1
    cbar_width_inches = 0.15

    cbar_left_rel = (plot_width_in + left_margin + cbar_gap_inches) / figure_width_inches
    cbar_bottom_rel = axes_bottom
    cbar_width_rel = cbar_width_inches / figure_width_inches
    cbar_height_rel = axes_height

    ax_colorbar.set_position([cbar_left_rel, cbar_bottom_rel, cbar_width_rel, cbar_height_rel])

    # 9. 统一重设主图字体 (SHAP 可能会设置自己的大字体)
    ax_beeswarm.set_title(f"({dataset_name}) SHAP Beeswarm Plot", fontsize=8)
    ax_beeswarm.set_xlabel("SHAP value (Impact on prediction)", fontsize=8)
    ax_beeswarm.tick_params(axis='both', which='major', labelsize=8)
    for label in ax_beeswarm.get_yticklabels():
        label.set_fontsize(8)

    # 10. 统一重设 Color Bar 字体
    # (SHAP v0.41+ 可能会自动设置标题 "Feature value")
    try:
        ax_colorbar.set_title(ax_colorbar.get_title().get_text(), fontsize=8)
    except AttributeError:
        pass  # 兼容旧版 SHAP
    ax_colorbar.tick_params(labelsize=8)
    ax_colorbar.set_ylabel(ax_colorbar.get_ylabel(), fontsize=8)  # 修复可能的 "Feature value" 标签

    _save_shap_plot("Beeswarm_Plot", fig_beeswarm)
    # [!!!] 已删除 plt.show()

    # ==============================================================
    # 配置板块：SHAP Bar Plot
    # [!] 已修改：改为控制【坐标轴区域】尺寸
    # ==============================================================
    SHAP_BAR_PLOT_CONFIG = {
        "plot_width_in": 1.6,  # 绘图区域宽度（英寸）
        "plot_height_in": 1.2,  # 绘图区域高度（英寸）
        "left_margin_in": 0.8,  # Figure 左边距（英寸），为 Y 轴标签 (Lactate) 预留
        "right_margin_in": 0.2,  # Figure 右边距（英寸）
        "top_margin_in": 0.4,  # Figure 上边距（英寸），为标题预留
        "bottom_margin_in": 0.5  # Figure 下边距（英寸），为 X 轴标签预留
    }
    # ==============================================================
    # 1. 从配置中读取尺寸
    plot_width_in = SHAP_BAR_PLOT_CONFIG["plot_width_in"]
    plot_height_in = SHAP_BAR_PLOT_CONFIG["plot_height_in"]
    left_margin = SHAP_BAR_PLOT_CONFIG["left_margin_in"]
    right_margin = SHAP_BAR_PLOT_CONFIG["right_margin_in"]
    top_margin = SHAP_BAR_PLOT_CONFIG["top_margin_in"]
    bottom_margin = SHAP_BAR_PLOT_CONFIG["bottom_margin_in"]

    # 2. 计算总 Figure 尺寸
    figure_width_inches = plot_width_in + left_margin + right_margin
    figure_height_inches = plot_height_in + top_margin + bottom_margin

    # 3. 计算 Axes 相对位置 ([left, bottom, width, height])
    axes_left = left_margin / figure_width_inches
    axes_bottom = bottom_margin / figure_height_inches
    axes_width = plot_width_in / figure_width_inches
    axes_height = plot_height_in / figure_height_inches

    # 4. [关键] 先让 SHAP 绘图
    shap.plots.bar(shap_values, max_display=10, show=False)

    # 5. [关键] 获取 SHAP 创建的 Figure 和 Axes
    fig_bar = plt.gcf()
    ax_bar = plt.gca()

    # 6. [关键] 将 Figure 尺寸重设为我们计算的
    fig_bar.set_size_inches(figure_width_inches, figure_height_inches)

    # 7. [关键] 将 Axes 位置重设为我们计算的
    ax_bar.set_position([axes_left, axes_bottom, axes_width, axes_height])

    # 8. 统一重设字体
    ax_bar.set_title(f"({dataset_name}) SHAP Feature Importance", fontsize=8)
    ax_bar.tick_params(axis='both', which='major', labelsize=8)
    # 确保 Y 轴标签字体也是 8
    for label in ax_bar.get_yticklabels():
        label.set_fontsize(8)
    # 确保 X 轴标签字体也是 8
    ax_bar.set_xlabel(ax_bar.get_xlabel(), fontsize=8)

    # 9. (布局和保存)
    _save_shap_plot("Bar_Plot", fig_bar)
    # [!!!] 已删除 plt.show()

    # ==============================================================
    # ==============================================================
    # 配置板块：SHAP Heatmap Plot
    # [!] 已修改：改为动态布局，以同时处理 3 轴 (Train) 和 2 轴 (Test) 的情况
    # ==============================================================
    SHAP_HEATMAP_PLOT_CONFIG = {
        "plot_width_in": 2.8,  # Heatmap 主图的宽度 (英寸)
        "plot_height_in": 1.4,  # Heatmap 主图的高度 (英寸)
        "fx_plot_height_in": 0.2,  # f(x) 图的高度 (英寸)
        "plot_gap_in": 0.05,  # f(x) 图与主图之间的垂直间隙 (英寸)

        "left_margin_in": 0.6,  # 整体左边距
        "right_margin_in": 0.7,  # 整体右边距 (为 Color Bar 预留)
        "top_margin_in": 0.4,  # 整体上边距
        "bottom_margin_in": 0.5  # 整体下边距
    }
    # ==============================================================

    try:
        # 1. [关键] 先让 SHAP 绘图 (cbar=True 是默认值)
        shap.plots.heatmap(shap_values, show=False)

        # 2. [关键] 获取 SHAP 创建的 Figure 和 *所有* Axes
        fig_heatmap = plt.gcf()
        all_axes = fig_heatmap.get_axes()
        num_axes = len(all_axes)

        # 3. 从配置中读取所有尺寸
        plot_width_in = SHAP_HEATMAP_PLOT_CONFIG["plot_width_in"]
        plot_height_in = SHAP_HEATMAP_PLOT_CONFIG["plot_height_in"]  # 主图高度
        fx_plot_height_in = SHAP_HEATMAP_PLOT_CONFIG["fx_plot_height_in"]  # f(x)图高度
        plot_gap_in = SHAP_HEATMAP_PLOT_CONFIG["plot_gap_in"]

        left_margin = SHAP_HEATMAP_PLOT_CONFIG["left_margin_in"]
        right_margin = SHAP_HEATMAP_PLOT_CONFIG["right_margin_in"]
        top_margin = SHAP_HEATMAP_PLOT_CONFIG["top_margin_in"]
        bottom_margin = SHAP_HEATMAP_PLOT_CONFIG["bottom_margin_in"]

        # Color Bar 的固定相对尺寸
        cbar_gap_inches = 0.1
        cbar_width_inches = 0.15

        # --- 开始动态布局 ---

        if num_axes == 3:
            # --- [布局 A: 3 轴 (f(x) + 主图 + CBar)] ---
            # (通常用于样本量大的训练集)

            # 1. 计算总 Figure 尺寸
            figure_width_inches = plot_width_in + left_margin + right_margin
            figure_height_inches = plot_height_in + fx_plot_height_in + plot_gap_in + top_margin + bottom_margin

            # 2. 重设 Figure 尺寸
            fig_heatmap.set_size_inches(figure_width_inches, figure_height_inches)

            # 3. 计算 [主图 Heatmap] 的相对位置
            ax_heatmap_left_rel = left_margin / figure_width_inches
            ax_heatmap_bottom_rel = bottom_margin / figure_height_inches
            ax_heatmap_width_rel = plot_width_in / figure_width_inches
            ax_heatmap_height_rel = plot_height_in / figure_height_inches

            # 4. 计算 [f(x) 图] 的相对位置 (放在主图上方)
            ax_fx_left_rel = ax_heatmap_left_rel
            ax_fx_bottom_rel = (bottom_margin + plot_height_in + plot_gap_in) / figure_height_inches
            ax_fx_width_rel = ax_heatmap_width_rel
            ax_fx_height_rel = fx_plot_height_in / figure_height_inches

            # 5. 计算 [Color Bar] 的相对位置 (与主图 Heatmap 对齐)
            cbar_left_rel = (plot_width_in + left_margin + cbar_gap_inches) / figure_width_inches
            cbar_bottom_rel = ax_heatmap_bottom_rel  # 与主图底部对齐
            cbar_width_rel = cbar_width_inches / figure_width_inches
            cbar_height_rel = ax_heatmap_height_rel  # 与主图等高

            # 6. 识别并重设所有 Axes
            ax_fx = all_axes[0]
            ax_heatmap = all_axes[1]
            ax_colorbar = all_axes[2]

            ax_fx.set_position([ax_fx_left_rel, ax_fx_bottom_rel, ax_fx_width_rel, ax_fx_height_rel])
            ax_heatmap.set_position(
                [ax_heatmap_left_rel, ax_heatmap_bottom_rel, ax_heatmap_width_rel, ax_heatmap_height_rel])
            ax_colorbar.set_position([cbar_left_rel, cbar_bottom_rel, cbar_width_rel, cbar_height_rel])

            # 7. 统一重设字体
            ax_fx.set_title("")
            ax_fx.tick_params(axis='both', which='major', labelsize=8)
            ax_fx.set_xlabel("")
            ax_fx.set_ylabel(ax_fx.get_ylabel(), fontsize=8)

            ax_heatmap.set_title("")
            ax_heatmap.tick_params(axis='both', which='major', labelsize=8)
            ax_heatmap.set_xlabel("Instances", fontsize=8)
            for label in ax_heatmap.get_yticklabels(): label.set_fontsize(8)

            ax_colorbar.tick_params(labelsize=8)
            ax_colorbar.set_ylabel(ax_colorbar.get_ylabel(), fontsize=8)

        elif num_axes == 2:
            # --- [布局 B: 2 轴 (主图 + CBar)] ---
            # (通常用于样本量小的测试集)

            # 1. 计算总 Figure 尺寸 (不含 f(x) 图)
            figure_width_inches = plot_width_in + left_margin + right_margin
            figure_height_inches = plot_height_in + top_margin + bottom_margin  # [修改] 移除了 f(x) 高度

            # 2. 重设 Figure 尺寸
            fig_heatmap.set_size_inches(figure_width_inches, figure_height_inches)

            # 3. 计算 [主图 Heatmap] 的相对位置
            ax_heatmap_left_rel = left_margin / figure_width_inches
            ax_heatmap_bottom_rel = bottom_margin / figure_height_inches
            ax_heatmap_width_rel = plot_width_in / figure_width_inches
            ax_heatmap_height_rel = plot_height_in / figure_height_inches

            # 4. 计算 [Color Bar] 的相对位置
            cbar_left_rel = (plot_width_in + left_margin + cbar_gap_inches) / figure_width_inches
            cbar_bottom_rel = ax_heatmap_bottom_rel
            cbar_width_rel = cbar_width_inches / figure_width_inches
            cbar_height_rel = ax_heatmap_height_rel

            # 5. 识别并重设所有 Axes
            ax_heatmap = all_axes[0]
            ax_colorbar = all_axes[1]

            ax_heatmap.set_position(
                [ax_heatmap_left_rel, ax_heatmap_bottom_rel, ax_heatmap_width_rel, ax_heatmap_height_rel])
            ax_colorbar.set_position([cbar_left_rel, cbar_bottom_rel, cbar_width_rel, cbar_height_rel])

            # 6. 统一重设字体
            ax_heatmap.set_title(f"({dataset_name}) SHAP Heatmap", fontsize=8)  # [修改] 标题加在这里
            ax_heatmap.tick_params(axis='both', which='major', labelsize=8)
            ax_heatmap.set_xlabel("Instances", fontsize=8)
            for label in ax_heatmap.get_yticklabels(): label.set_fontsize(8)

            ax_colorbar.tick_params(labelsize=8)
            ax_colorbar.set_ylabel(ax_colorbar.get_ylabel(), fontsize=8)

        else:
            # --- [布局 C: 异常情况] ---
            print(f"  ⚠️ SHAP Heatmap 未产生预期的 2 或 3 个坐标轴 (实际: {num_axes})。跳过重设位置。")
            # 仍然尝试设置标题
            try:
                all_axes[0].set_title(f"({dataset_name}) SHAP Heatmap", fontsize=8)
            except Exception:
                pass

        # --- 动态布局结束 ---

        # 11. (布局和保存)
        _save_shap_plot("Heatmap_Plot", fig_heatmap)

    except Exception as e:
        print(f"  ⚠️ 跳过 Heatmap 绘制: {e}")
        try:
            plt.close(fig_heatmap)  # 尝试关闭 SHAP 创建的图
        except NameError:
            plt.close()  # 关闭当前活动的图
    # ==============================================================
    # 配置板块：SHAP Dependence Plot (每张图共用此配置)
    # [!] 这是您的原始代码，它是正确的
    # ==============================================================
    SHAP_DEPENDENCE_PLOT_CONFIG = {
        "plot_width_in": 1.4,  # 绘图区域宽度（英寸），约 8cm
        "plot_height_in": 1.4,  # 绘图区域高度（英寸），约 7cm
        "left_margin_in": 0.6,
        "right_margin_in": 0.2,
        "top_margin_in": 0.4,
        "bottom_margin_in": 0.5
    }
    # ==============================================================
    for feat_name in feature_names_for_plot:
        fig_dep, ax_dep = _create_axes_with_plot_area_in(
            plot_width_in=SHAP_DEPENDENCE_PLOT_CONFIG["plot_width_in"],
            plot_height_in=SHAP_DEPENDENCE_PLOT_CONFIG["plot_height_in"],
            left_margin_in=SHAP_DEPENDENCE_PLOT_CONFIG["left_margin_in"],
            right_margin_in=SHAP_DEPENDENCE_PLOT_CONFIG["right_margin_in"],
            top_margin_in=SHAP_DEPENDENCE_PLOT_CONFIG["top_margin_in"],
            bottom_margin_in=SHAP_DEPENDENCE_PLOT_CONFIG["bottom_margin_in"]
        )
        try:
            shap.dependence_plot(feat_name, shap_values.values, X_data,
                                 feature_names=feature_names_for_plot,
                                 interaction_index=None, show=False, ax=ax_dep)

            # --- [开始 字体修复] ---
            title_fontsize = 8
            label_fontsize = 8
            tick_fontsize = 8

            ax_dep.set_title(f"({dataset_name}) Dependence: {feat_name}", fontsize=title_fontsize)
            # 重设 SHAP 自动生成的标签的字体
            ax_dep.set_xlabel(ax_dep.get_xlabel(), fontsize=label_fontsize)
            ax_dep.set_ylabel(ax_dep.get_ylabel(), fontsize=label_fontsize)
            ax_dep.tick_params(axis='both', which='major', labelsize=tick_fontsize)

            # 修复 colorbar 字体 (如果存在自动交互)
            if len(fig_dep.axes) > 1:
                cbar_ax = fig_dep.axes[1]
                cbar_ax.set_ylabel(cbar_ax.get_ylabel(), fontsize=label_fontsize)
                cbar_ax.tick_params(labelsize=tick_fontsize)
            # --- [结束 字体修复] ---

            for line in ax_dep.get_lines():
                line.set_linewidth(0.5)
            _save_shap_plot(f"Dependence_Plot_{feat_name}", fig_dep)
        except Exception as e:
            print(f"  ⚠️ 跳过 {feat_name} 依赖图绘制: {e}")
            plt.close(fig_dep)

    # --- Interaction Dependence Plots ---
    print(f"  绘制 {dataset_name} 交互依赖图...")
    # ==============================================================
    # 配置板块：SHAP Interaction Dependence Plot (每张图共用此配置)
    # [!] 这是您的原始代码，它是正确的
    # ==============================================================
    SHAP_INTERACTION_PLOT_CONFIG = {
        "plot_width_in": 1.4,  # [修改] 绘图区域宽度 (英寸) - 增加尺寸并保持方形
        "plot_height_in": 1.0,  # [修改] 绘图区域高度 (英寸) - 增加尺寸并保持方形
        "left_margin_in": 0.6,
        "right_margin_in": 0.2,
        "top_margin_in": 0.4,
        "bottom_margin_in": 0.5
    }
    # ==============================================================
    interaction_pairs = [('Lactate', 'pH'), ('Glucose', 'Lactate'), ('Urea', 'Lactate'), ('Glucose', 'pH'),
                         ('Glucose', 'Urea')]
    for main_feat, inter_feat in interaction_pairs:
        if main_feat not in feature_names_for_plot or inter_feat not in feature_names_for_plot: continue
        fig_int, ax_int = _create_axes_with_plot_area_in(
            plot_width_in=SHAP_INTERACTION_PLOT_CONFIG["plot_width_in"],
            plot_height_in=SHAP_INTERACTION_PLOT_CONFIG["plot_height_in"],
            left_margin_in=SHAP_INTERACTION_PLOT_CONFIG["left_margin_in"],
            right_margin_in=SHAP_INTERACTION_PLOT_CONFIG["right_margin_in"],
            top_margin_in=SHAP_INTERACTION_PLOT_CONFIG["top_margin_in"],
            bottom_margin_in=SHAP_INTERACTION_PLOT_CONFIG["bottom_margin_in"]
        )
        try:
            # 1. 绘制 SHAP 图
            shap.dependence_plot(main_feat, shap_values.values, X_data,
                                 feature_names=feature_names_for_plot,
                                 interaction_index=inter_feat, show=False, ax=ax_int)

            # --- [开始 字体修复] ---
            title_fontsize = 8
            label_fontsize = 8
            tick_fontsize = 8

            # 2. 覆盖标题和坐标轴标签字体
            ax_int.set_title(f"({dataset_name}) Interaction: {main_feat} vs {inter_feat}", fontsize=title_fontsize)
            ax_int.set_xlabel(ax_int.get_xlabel(), fontsize=label_fontsize)  # 重设X轴
            ax_int.set_ylabel(ax_int.get_ylabel(), fontsize=label_fontsize)  # 重设Y轴
            ax_int.tick_params(axis='both', which='major', labelsize=tick_fontsize)  # 刻度

            # 3. 修复 Colorbar 的字体
            if len(fig_int.axes) > 1:  # 检查 colorbar (fig.axes[1]) 是否存在的
                cbar_ax = fig_int.axes[1]
                # 重设 Colorbar 标签字体
                cbar_ax.set_ylabel(cbar_ax.get_ylabel(), fontsize=label_fontsize)
                # 重设 Colorbar 刻度字体
                cbar_ax.tick_params(labelsize=tick_fontsize)
            # --- [结束 字体修复] ---

            # 4. (保留) 原始的线条宽度设置
            for line in ax_int.get_lines():
                line.set_linewidth(0.5)

            _save_shap_plot(f"Interaction_Plot_{main_feat}_vs_{inter_feat}", fig_int)
        except Exception as e:
            print(f"  ⚠️ 跳过 {main_feat} x {inter_feat} 交互图绘制: {e}")
            plt.close(fig_int)

    # (Decision Plot 部分保持不变)
    # ==============================================================
    # 配置板块：SHAP Decision Plot
    # [!] 已添加：这是新添加的 Decision Plot 块
    # ==============================================================
    SHAP_DECISION_PLOT_CONFIG = {
        "plot_width_in": 2.4,  # 绘图区域宽度（英寸）- 决策图通常较宽
        "plot_height_in": 1.6,  # 绘图区域高度（英寸）
        "left_margin_in": 0.6,  # Figure 左边距（英寸）
        "right_margin_in": 0.2,  # Figure 右边距（英寸）
        "top_margin_in": 0.4,  # Figure 上边距（英寸），为标题预留
        "bottom_margin_in": 0.5  # Figure 下边距（英寸），为 X 轴标签预留
    }
    # ==============================================================
    # 1. 从配置中读取尺寸
    plot_width_in = SHAP_DECISION_PLOT_CONFIG["plot_width_in"]
    plot_height_in = SHAP_DECISION_PLOT_CONFIG["plot_height_in"]
    left_margin = SHAP_DECISION_PLOT_CONFIG["left_margin_in"]
    right_margin = SHAP_DECISION_PLOT_CONFIG["right_margin_in"]
    top_margin = SHAP_DECISION_PLOT_CONFIG["top_margin_in"]
    bottom_margin = SHAP_DECISION_PLOT_CONFIG["bottom_margin_in"]

    # 2. 计算总 Figure 尺寸
    figure_width_inches = plot_width_in + left_margin + right_margin
    figure_height_inches = plot_height_in + top_margin + bottom_margin

    # 3. 计算 Axes 相对位置 ([left, bottom, width, height])
    axes_left = left_margin / figure_width_inches
    axes_bottom = bottom_margin / figure_height_inches
    axes_width = plot_width_in / figure_width_inches
    axes_height = plot_height_in / figure_height_inches

    try:
        base_value = shap_values.base_values[0]

        # 4. [关键] 先让 SHAP 绘图
        shap.decision_plot(base_value, shap_values.values, shap_values.data, feature_names=feature_names_for_plot,
                           show=False)

        # 5. [关键] 获取 SHAP 创建的 Figure 和 Axes
        fig_decision = plt.gcf()
        ax_decision = plt.gca()

        # 6. [关键] 将 Figure 尺寸重设为我们计算的
        fig_decision.set_size_inches(figure_width_inches, figure_height_inches)

        # 7. [关键] 将 Axes 位置重设为我们计算的
        ax_decision.set_position([axes_left, axes_bottom, axes_width, axes_height])

        # 8. 统一重设字体和样式
        ax_decision.set_title(f"({dataset_name}) SHAP Decision Plot (All Samples)", fontsize=8)
        ax_decision.tick_params(axis='both', which='major', labelsize=8)
        ax_decision.set_xlabel(ax_decision.get_xlabel(), fontsize=8)
        ax_decision.set_ylabel(ax_decision.get_ylabel(), fontsize=8)

        for line in ax_decision.get_lines():
            line.set_linewidth(0.5)
        for spine in ax_decision.spines.values():
            spine.set_linewidth(0.5)  # [!] 修复了拼写错误 0.E -> 0.5

        # 9. (布局和保存)
        _save_shap_plot("Decision_Plot_All_Samples", fig_decision)
    except Exception as e:
        print(f"⚠️ 跳过 Decision Plot 绘制（原因: {e}）")
        # 尝试关闭 fig_decision (如果它在 try 块中被创建了)
        try:
            plt.close(fig_decision)
        except NameError:
            plt.close()  # 关闭当前活动的图
    # [!!!] 已删除末尾的 else: plt.show() 块
    # ==============================================================


# ==============================================================
# 8.1 训练集 SHAP 分析
# ==============================================================
print("\n--- 分析开发集 (Train) ---")
shap_values_train_scaled = explainer(X_train_shap_scaled)
shap_values_train = shap.Explanation(
    values=shap_values_train_scaled.values,
    base_values=shap_values_train_scaled.base_values,
    data=X_train_shap_unscaled,
    feature_names=feature_names_shap  # 使用新的变量名
)
plot_all_shap_figures(shap_values_train, X_train_shap_unscaled, feature_names_for_plot=feature_names_shap,
                      dataset_name="Train", save_dir=SAVE_DIR)

# ==============================================================
# 8.2 测试集 SHAP 分析
# ==============================================================
print("\n--- 分析测试集 (Test) ---")
X_test_shap_scaled = scaler_to_explain.transform(X_test_shap_unscaled)
shap_values_test_scaled = explainer(X_test_shap_scaled)
shap_values_test = shap.Explanation(
    values=shap_values_test_scaled.values,
    base_values=shap_values_test_scaled.base_values,
    data=X_test_shap_unscaled,
    feature_names=feature_names_shap  # 使用新的变量名
)
plot_all_shap_figures(shap_values_test, X_test_shap_unscaled, feature_names_for_plot=feature_names_shap,
                      dataset_name="Test", save_dir=SAVE_DIR)

# ==============================================================
# 9. [替换] 总结报告 (并保存 Full-Model 结果到文件)
# ==============================================================
print("\n=== Workflow Completed ===")

# --- 打印对比结果 ---
print(f"GA-CV 探索发现: {ga_feature_names} 是最优组合 (CV AUC: {np.mean(scores):.3f} ± {np.std(scores):.3f})")
print(f"--- 最终测试集对比 (n=12) ---")
print(f"GA-Optimal ({len(ga_feature_names)}特征) 测试 AUC: {final_results['GA-Optimal']:.3f}")
print(f"Full-Model ({len(full_feature_names)}特征) 测试 AUC: {final_results['Full-Model']:.3f}")

# --- 保存 Full-Model 的详细测试指标到文件 ---

# 1. 从字典中获取 Full-Model 的 model 和 scaler
model_full_data = final_models["Full-Model"]
model_full = model_full_data["model"]
scaler_full = model_full_data["scaler"]
indices_full = models_to_test["Full-Model"][0]

# 2. 手动缩放测试数据并预测
X_test_full_scaled = scaler_full.transform(X_test[:, indices_full])
y_pred_prob_full = model_full.predict_proba(X_test_full_scaled)[:, 1]
y_pred_full = (y_pred_prob_full > 0.5).astype(int)

# 3. 获取指标值
full_auc = roc_auc_score(y_test, y_pred_prob_full)
full_accuracy = accuracy_score(y_test, y_pred_full)
full_precision = precision_score(y_test, y_pred_full)
full_recall = recall_score(y_test, y_pred_full)
full_f1 = f1_score(y_test, y_pred_full)

# 4. 写入文件
report_path = os.path.join(SAVE_DIR, "Model_Report_Full_Model.txt")
try:
    with open(report_path, "w") as f:
        f.write("=== Final Test Metrics (Full-Model, 4 Features) ===\n")
        f.write(f"AUC: {full_auc:.3f}\n")
        f.write(f"Accuracy: {full_accuracy:.3f}\n")
        f.write(f"Precision: {full_precision:.3f}\n")
        f.write(f"Recall: {full_recall:.3f}\n")
        f.write(f"F1 Score: {full_f1:.3f}\n")

        # (为了确保 CI 可用，需要从上面循环中获取 full_ci_lower 和 full_ci_upper)
        # 假设我们为 Full-Model 存储了这些值
        # 这里需要从 `final_models` 或 `final_results` 中获取
        # 假设在 `final_results` 中存储了 CI (如果上一步您修改了的话)
        # 否则，需要从步骤7的循环中捕获并保存 `ci_lower` 和 `ci_upper`
        # 暂时跳过CI写入，以免引起新的错误
        pass  # placeholder for CI
        # if 'full_model_ci_lower' in locals(): # 需要在步骤7的循环中设置这些变量
        #    f.write(f"AUC 95% CI: [{full_model_ci_lower:.3f}, {full_model_ci_upper:.3f}]\n")

    print(f"\n✅ Full-Model 的详细测试报告已保存到: {report_path}")

except Exception as e:
    print(f"\n❌ 保存报告失败: {e}")

# ==============================================================
# 10. [替换] 最终稳健性验证 (在 n=60 上)
# ==============================================================
print("\n=== 最终稳健性验证 (50x CV on n=60) ===")
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_SEED)

# 验证 GA-Optimal (2特征) 模型
model_ga = deepcopy(model_pipeline)
X_ga = X[:, ga_features_idx]
cv_scores_ga = cross_val_score(model_ga, X_ga, y, cv=cv, scoring='roc_auc')
print(f"GA-Optimal (2特征) Mean AUC: {np.mean(cv_scores_ga):.3f} ± {np.std(cv_scores_ga):.3f}")

# 验证 Full-Model (4特征) 模型
model_full = deepcopy(model_pipeline)
X_full = X[:, full_features_idx]
cv_scores_full = cross_val_score(model_full, X_full, y, cv=cv, scoring='roc_auc')
print(f"Full-Model (4特征) Mean AUC: {np.mean(cv_scores_full):.3f} ± {np.std(cv_scores_full):.3f}")