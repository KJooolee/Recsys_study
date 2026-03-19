import matplotlib.pyplot as plt
import os

def plot_training_history(train_losses, val_losses, val_metrics, val_epochs, metric_name, save_path):
    """
    에포크별 Train Loss 그래프와 Validation 그래프를 따로 그려 저장합니다.
    """
    epochs = range(1, len(train_losses) + 1)
    
    # ---- 1. Train History Plot ----
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    color_loss = '#E24A33' # 빨간색 계열
    plt.xlabel('Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Train Loss', color=color_loss, fontsize=12, fontweight='bold')
    plt.plot(epochs, train_losses, color=color_loss, marker='o', linestyle='-', linewidth=2, label='Train Loss')
    plt.tick_params(axis='y', labelcolor=color_loss)
    if len(train_losses) > 0 and max(train_losses) > 0:
        plt.ylim(0, max(train_losses) * 1.1) 

    plt.title('Training History: Train Loss', fontsize=15, fontweight='bold')
    plt.legend(loc='upper right', frameon=True, fontsize=11)
    plt.tight_layout()
    
    # 저장 경로 확보
    plot_save_path = save_path.replace('.pt', '_train_history.png')
    plot_save_path = plot_save_path.replace('checkpoints/', 'results/')
    dir_name = os.path.dirname(plot_save_path)
    if not os.path.exists(dir_name):
        parts = dir_name.split('/')
        if len(parts) >= 2:
            base_dir = f"{parts[0]}/{parts[1]}"
            os.makedirs(base_dir, exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'train'), exist_ok=True)
            filename = os.path.basename(plot_save_path)
            plot_save_path = os.path.join(base_dir, 'train', filename)

    plt.savefig(plot_save_path, dpi=300)
    plt.close()
    
    # ---- 2. Val History Plot ----
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color_vloss = '#8B0000' # 어두운 빨간색
    ax1.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Val Loss', color=color_vloss, fontsize=12, fontweight='bold')
    ax1.plot(val_epochs, val_losses, color=color_vloss, marker='o', linestyle='-', linewidth=2, label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color_vloss)
    if len(val_losses) > 0 and max(val_losses) > 0:
        ax1.set_ylim(0, max(val_losses) * 1.1) 

    ax2 = ax1.twinx()
    color_metric = '#348ABD' # 파란색 계열
    ax2.set_ylabel(metric_name, color=color_metric, fontsize=12, fontweight='bold')
    ax2.plot(val_epochs, val_metrics, color=color_metric, marker='s', linestyle='--', linewidth=2, label=metric_name)
    ax2.tick_params(axis='y', labelcolor=color_metric)
    ax2.set_ylim(0, 1.0) 

    plt.title(f'Validation History: Val Loss & {metric_name}', fontsize=15, fontweight='bold')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', frameon=True, fontsize=11)
    
    plt.tight_layout()
    
    val_plot_save_path = plot_save_path.replace('_train_history.png', '_val_history.png')
    plt.savefig(val_plot_save_path, dpi=300)
    plt.close()
    
    print(f"✅ 학습 히스토리 그래프 분리 저장 완료: \n   - {plot_save_path}\n   - {val_plot_save_path}")