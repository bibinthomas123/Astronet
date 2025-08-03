import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from scipy import stats
import matplotlib.patches as mpatches

# Set publication-quality plot parameters
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (10, 8),
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Load your data
def load_results():
    """Load your model results - replace with your actual data loading"""
    data = {
        'KIC_ID': [11442793, 8480285, 11568987, 4852528, 5972334, 10337517, 11030475, 4548011, 
                   3832474, 11669239, 10130039, 5351250, 5301750, 8282651, 6786037, 6690082, 
                   9896018, 11450414, 6021275, 4851530, 8804283, 3323887, 6508221, 9006186, 
                   12061969, 11074178, 8397947, 11968463, 10600261],
        'Period': [331.603, 29.6663, 15.96, 7.05354, 15.3588, 7.05395, 2.81816, 6.28372,
                   60.3252, 41.8855, 12.758, 7.38199, 4.2245, 2.36172, 127.904, 2.13956,
                   2.7296, 12.7984, 6.41495, 23.0818, 22.7902, 19.2222, 18.208, 5.45297,
                   14.0945, 14.3051, 3.38081, 10.0438, 17.308],
        'Duration': [14.49, 8.488, 4.498, 2.725, 4.176, 1.712, 1.786, 4.094,
                     6.556, 6.728, 4.357, 3.349, 3.168, 1.361, 14.01, 2.198,
                     2.852, 6.334, 3.53, 9.43, 5.239, 3.084, 3.96, 2.231,
                     6.24, 4.406, 3.558, 5.932, 5.365],
        'Prediction_Score': [0.747197, 0.981653, 0.975086, 0.952657, 0.921463, 0.92191, 0.868597, 0.837181,
                            0.921655, 0.979409, 0.912683, 0.971264, 0.939629, 0.93583, 0.945543, 0.829385,
                            0.836133, 0.944399, 0.968652, 0.677538, 0.985913, 0.299434, 0.966952, 0.928571,
                            0.164833, 0.463184, 0.024054, 0.912622, 0.978694]
    }
    return pd.DataFrame(data)

def create_model_architecture_diagram():
    """Create a diagram showing your model architecture"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define components
    components = [
        {'name': 'Light Curve\nPreprocessing', 'x': 1, 'y': 6, 'width': 1.5, 'height': 1, 'color': 'lightblue'},
        {'name': 'Shallue CNN\n(Conv1D Layers)', 'x': 3.5, 'y': 6, 'width': 2, 'height': 1, 'color': 'lightgreen'},
        {'name': 'BiLSTM Layer\n(Forward + Backward)', 'x': 6.5, 'y': 6, 'width': 2, 'height': 1, 'color': 'orange'},
        {'name': 'Attention\nMechanism', 'x': 9.5, 'y': 6, 'width': 1.5, 'height': 1, 'color': 'pink'},
        {'name': 'Dense Layers\n+ Dropout', 'x': 5, 'y': 3.5, 'width': 2, 'height': 1, 'color': 'yellow'},
        {'name': 'Planet\nProbability', 'x': 5, 'y': 1, 'width': 2, 'height': 1, 'color': 'lightcoral'}
    ]
    
    # Draw components
    for comp in components:
        rect = Rectangle((comp['x'], comp['y']), comp['width'], comp['height'], 
                        facecolor=comp['color'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(comp['x'] + comp['width']/2, comp['y'] + comp['height']/2, comp['name'], 
                ha='center', va='center', fontsize=10, weight='bold')
    
    # Draw arrows
    arrows = [
        ((2.5, 6.5), (3.5, 6.5)),  # Preprocessing to CNN
        ((5.5, 6.5), (6.5, 6.5)),  # CNN to BiLSTM
        ((8.5, 6.5), (9.5, 6.5)),  # BiLSTM to Attention
        ((10.25, 6), (6.5, 4.5)),  # Attention to Dense (curved)
        ((6, 3.5), (6, 2))         # Dense to Output
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start, 
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Hybrid CNN-BiLSTM-Attention Architecture for Exoplanet Detection', 
                fontsize=16, weight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def plot_period_vs_confidence(df):
    """Plot orbital period vs prediction confidence"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create period bins for color coding
    period_bins = [0, 10, 30, 100, 400]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['Short (<10d)', 'Medium (10-30d)', 'Long (30-100d)', 'Very Long (>100d)']
    
    for i in range(len(period_bins)-1):
        mask = (df['Period'] >= period_bins[i]) & (df['Period'] < period_bins[i+1])
        if mask.any():
            ax.scatter(df[mask]['Period'], df[mask]['Prediction_Score'], 
                      c=colors[i], s=100, alpha=0.7, label=labels[i], edgecolors='black')
    
    # Add trend line
    z = np.polyfit(df['Period'], df['Prediction_Score'], 1)
    p = np.poly1d(z)
    ax.plot(df['Period'], p(df['Period']), "r--", alpha=0.8, linewidth=2, label='Trend Line')
    
    # Highlight the long-period detection
    highlight_idx = df['Period'].idxmax()
    ax.annotate(f'KIC {df.loc[highlight_idx, "KIC_ID"]}\n{df.loc[highlight_idx, "Period"]:.1f}d\n{df.loc[highlight_idx, "Prediction_Score"]:.3f}', 
                xy=(df.loc[highlight_idx, 'Period'], df.loc[highlight_idx, 'Prediction_Score']),
                xytext=(250, 0.9), arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Orbital Period (days)', fontsize=14)
    ax.set_ylabel('Model Confidence Score', fontsize=14)
    ax.set_title('Exoplanet Detection Confidence vs Orbital Period\n(Hybrid CNN-BiLSTM-Attention Model)', fontsize=16, weight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)
    
    # Add correlation info
    corr, p_val = stats.pearsonr(np.log(df['Period']), df['Prediction_Score'])
    ax.text(0.02, 0.98, f'Correlation (log-period): r = {corr:.3f}, p = {p_val:.3f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_performance_comparison():
    """Compare your model with Shallue's model (simulated data - replace with actual)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Simulated comparison data - replace with your actual benchmark results
    methods = ['Shallue CNN', 'Our Hybrid Model']
    
    # Overall accuracy
    accuracies = [0.94, 0.91]  # Replace with actual values
    ax1.bar(methods, accuracies, color=['skyblue', 'orange'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Overall Accuracy', fontsize=14)
    ax1.set_title('Overall Model Performance', fontsize=14, weight='bold')
    ax1.set_ylim(0.8, 1.0)
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=12, weight='bold')
    
    # Performance by period range
    period_ranges = ['<10 days', '10-30 days', '30-100 days', '>100 days']
    shallue_perf = [0.96, 0.94, 0.85, 0.45]  # Replace with actual values
    hybrid_perf = [0.94, 0.95, 0.92, 0.78]   # Replace with actual values
    
    x = np.arange(len(period_ranges))
    width = 0.35
    
    ax2.bar(x - width/2, shallue_perf, width, label='Shallue CNN', color='skyblue', alpha=0.7, edgecolor='black')
    ax2.bar(x + width/2, hybrid_perf, width, label='Our Hybrid Model', color='orange', alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel('Orbital Period Range', fontsize=14)
    ax2.set_ylabel('Detection Accuracy', fontsize=14)
    ax2.set_title('Performance by Period Range', fontsize=14, weight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(period_ranges)
    ax2.legend()
    ax2.set_ylim(0.3, 1.0)
    
    # Add value labels
    for i, (v1, v2) in enumerate(zip(shallue_perf, hybrid_perf)):
        ax2.text(i - width/2, v1 + 0.02, f'{v1:.2f}', ha='center', fontsize=10)
        ax2.text(i + width/2, v2 + 0.02, f'{v2:.2f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrices():
    """Plot confusion matrices for both models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Simulated confusion matrices - replace with your actual test results
    # Format: [[TN, FP], [FN, TP]]
    cm_shallue = np.array([[850, 45], [25, 80]])  # Replace with actual
    cm_hybrid = np.array([[820, 75], [15, 90]])   # Replace with actual
    
    # Plot Shallue confusion matrix
    sns.heatmap(cm_shallue, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Non-Planet', 'Planet'], 
                yticklabels=['Non-Planet', 'Planet'],
                cbar_kws={'label': 'Count'})
    ax1.set_title('Shallue CNN Model\nConfusion Matrix', fontsize=14, weight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    
    # Calculate metrics for Shallue
    tn_s, fp_s, fn_s, tp_s = cm_shallue.ravel()
    precision_s = tp_s / (tp_s + fp_s)
    recall_s = tp_s / (tp_s + fn_s)
    f1_s = 2 * (precision_s * recall_s) / (precision_s + recall_s)
    accuracy_s = (tp_s + tn_s) / (tp_s + tn_s + fp_s + fn_s)
    
    ax1.text(0.02, 0.98, f'Accuracy: {accuracy_s:.3f}\nPrecision: {precision_s:.3f}\nRecall: {recall_s:.3f}\nF1-Score: {f1_s:.3f}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9),
             fontsize=10)
    
    # Plot Hybrid confusion matrix
    sns.heatmap(cm_hybrid, annot=True, fmt='d', cmap='Oranges', ax=ax2,
                xticklabels=['Non-Planet', 'Planet'], 
                yticklabels=['Non-Planet', 'Planet'],
                cbar_kws={'label': 'Count'})
    ax2.set_title('Hybrid CNN-BiLSTM-Attention\nConfusion Matrix', fontsize=14, weight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    
    # Calculate metrics for Hybrid
    tn_h, fp_h, fn_h, tp_h = cm_hybrid.ravel()
    precision_h = tp_h / (tp_h + fp_h)
    recall_h = tp_h / (tp_h + fn_h)
    f1_h = 2 * (precision_h * recall_h) / (precision_h + recall_h)
    accuracy_h = (tp_h + tn_h) / (tp_h + tn_h + fp_h + fn_h)
    
    ax2.text(0.02, 0.98, f'Accuracy: {accuracy_h:.3f}\nPrecision: {precision_h:.3f}\nRecall: {recall_h:.3f}\nF1-Score: {f1_h:.3f}', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9),
             fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_roc_pr_curves():
    """Plot ROC and Precision-Recall curves (simulated data - replace with actual)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Simulated data - replace with your actual test results
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.3, 1000)  # 30% positive class
    
    # Simulate scores for both models
    shallue_scores = np.random.beta(2, 1, 1000) * y_true + np.random.beta(1, 3, 1000) * (1 - y_true)
    hybrid_scores = np.random.beta(2.2, 1, 1000) * y_true + np.random.beta(1, 3.2, 1000) * (1 - y_true)
    
    # ROC Curves
    fpr_s, tpr_s, _ = roc_curve(y_true, shallue_scores)
    fpr_h, tpr_h, _ = roc_curve(y_true, hybrid_scores)
    auc_s = auc(fpr_s, tpr_s)
    auc_h = auc(fpr_h, tpr_h)
    
    ax1.plot(fpr_s, tpr_s, label=f'Shallue CNN (AUC = {auc_s:.3f})', color='blue', linewidth=2)
    ax1.plot(fpr_h, tpr_h, label=f'Hybrid Model (AUC = {auc_h:.3f})', color='red', linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    ax1.set_xlabel('False Positive Rate', fontsize=14)
    ax1.set_ylabel('True Positive Rate', fontsize=14)
    ax1.set_title('ROC Curves', fontsize=14, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curves
    precision_s, recall_s, _ = precision_recall_curve(y_true, shallue_scores)
    precision_h, recall_h, _ = precision_recall_curve(y_true, hybrid_scores)
    auc_pr_s = auc(recall_s, precision_s)
    auc_pr_h = auc(recall_h, precision_h)
    
    ax2.plot(recall_s, precision_s, label=f'Shallue CNN (AUC = {auc_pr_s:.3f})', color='blue', linewidth=2)
    ax2.plot(recall_h, precision_h, label=f'Hybrid Model (AUC = {auc_pr_h:.3f})', color='red', linewidth=2)
    ax2.axhline(y=0.3, color='k', linestyle='--', alpha=0.5, label='Random Classifier')
    ax2.set_xlabel('Recall', fontsize=14)
    ax2.set_ylabel('Precision', fontsize=14)
    ax2.set_title('Precision-Recall Curves', fontsize=14, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_confidence_distribution(df):
    """Plot distribution of confidence scores"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram of confidence scores
    ax1.hist(df['Prediction_Score'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(df['Prediction_Score'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {df["Prediction_Score"].mean():.3f}')
    ax1.axvline(df['Prediction_Score'].median(), color='green', linestyle='--', linewidth=2,
                label=f'Median: {df["Prediction_Score"].median():.3f}')
    ax1.set_xlabel('Prediction Score', fontsize=14)
    ax1.set_ylabel('Frequency', fontsize=14)
    ax1.set_title('Distribution of Model Confidence Scores', fontsize=14, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot by period ranges
    df_copy = df.copy()
    df_copy['Period_Range'] = pd.cut(df_copy['Period'], 
                                   bins=[0, 10, 30, 100, 400], 
                                   labels=['<10d', '10-30d', '30-100d', '>100d'])
    
    box_data = [df_copy[df_copy['Period_Range'] == pr]['Prediction_Score'].values 
                for pr in ['<10d', '10-30d', '30-100d', '>100d'] if not df_copy[df_copy['Period_Range'] == pr].empty]
    labels = [pr for pr in ['<10d', '10-30d', '30-100d', '>100d'] 
              if not df_copy[df_copy['Period_Range'] == pr].empty]
    
    bp = ax2.boxplot(box_data, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'orange', 'lightcoral'][:len(bp['boxes'])]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Orbital Period Range', fontsize=14)
    ax2.set_ylabel('Prediction Score', fontsize=14)
    ax2.set_title('Confidence Score Distribution by Period Range', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_duration_vs_confidence(df):
    """Plot transit duration vs confidence"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot with size representing period
    scatter = ax.scatter(df['Duration'], df['Prediction_Score'], 
                        s=np.sqrt(df['Period'])*20, alpha=0.6, 
                        c=df['Period'], cmap='viridis', edgecolors='black')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Orbital Period (days)', fontsize=14)
    
    # Add trend line
    z = np.polyfit(df['Duration'], df['Prediction_Score'], 1)
    p = np.poly1d(z)
    ax.plot(df['Duration'], p(df['Duration']), "r--", alpha=0.8, linewidth=2)
    
    ax.set_xlabel('Transit Duration (hours)', fontsize=14)
    ax.set_ylabel('Model Confidence Score', fontsize=14)
    ax.set_title('Transit Duration vs Detection Confidence\n(Bubble size ∝ √Period)', fontsize=16, weight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add correlation info
    corr, p_val = stats.pearsonr(df['Duration'], df['Prediction_Score'])
    ax.text(0.02, 0.98, f'Correlation: r = {corr:.3f}, p = {p_val:.3f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_summary_table(df):
    """Create a summary statistics table"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate statistics
    stats_data = {
        'Metric': ['Total Targets', 'Mean Confidence', 'Median Confidence', 'Std Confidence',
                   'High Confidence (>0.9)', 'Medium Confidence (0.5-0.9)', 'Low Confidence (<0.5)',
                   'Mean Period (days)', 'Median Period (days)', 'Long Period (>100d)',
                   'Mean Duration (hours)', 'Median Duration (hours)'],
        'Value': [
            len(df),
            f"{df['Prediction_Score'].mean():.3f}",
            f"{df['Prediction_Score'].median():.3f}",
            f"{df['Prediction_Score'].std():.3f}",
            f"{(df['Prediction_Score'] > 0.9).sum()} ({(df['Prediction_Score'] > 0.9).mean()*100:.1f}%)",
            f"{((df['Prediction_Score'] >= 0.5) & (df['Prediction_Score'] <= 0.9)).sum()} ({((df['Prediction_Score'] >= 0.5) & (df['Prediction_Score'] <= 0.9)).mean()*100:.1f}%)",
            f"{(df['Prediction_Score'] < 0.5).sum()} ({(df['Prediction_Score'] < 0.5).mean()*100:.1f}%)",
            f"{df['Period'].mean():.2f}",
            f"{df['Period'].median():.2f}",
            f"{(df['Period'] > 100).sum()} ({(df['Period'] > 100).mean()*100:.1f}%)",
            f"{df['Duration'].mean():.2f}",
            f"{df['Duration'].median():.2f}"
        ]
    }
    
    table = ax.table(cellText=[[metric, value] for metric, value in zip(stats_data['Metric'], stats_data['Value'])],
                     colLabels=['Metric', 'Value'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.6, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(stats_data['Metric']) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
            cell.set_edgecolor('black')
            cell.set_linewidth(1)
    
    ax.set_title('Model Performance Summary Statistics', fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    return fig

# Main function to generate all plots
def generate_all_plots():
    """Generate all publication-quality plots"""
    df = load_results()
    
    plots = {
        'architecture': create_model_architecture_diagram(),
        'period_confidence': plot_period_vs_confidence(df),
        'performance_comparison': plot_performance_comparison(),
        'confusion_matrices': plot_confusion_matrices(),
        'roc_pr_curves': plot_roc_pr_curves(),
        'confidence_distribution': plot_confidence_distribution(df),
        'duration_confidence': plot_duration_vs_confidence(df),
        'summary_table': create_summary_table(df)
    }
    
    return plots

# Run the script
if __name__ == "__main__":
    plots = generate_all_plots()
    
    # Save all plots
    for name, fig in plots.items():
        fig.savefig(f'{name}_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(f'{name}_plot.pdf', bbox_inches='tight', facecolor='white')
        print(f"Saved {name}_plot.png and {name}_plot.pdf")
    
    plt.show()

print("All publication-quality plots generated successfully!")
print("\nGenerated plots:")
print("1. Model Architecture Diagram")
print("2. Period vs Confidence Analysis")
print("3. Performance Comparison with Baseline")
print("4. Confusion Matrices (Side-by-side)")
print("5. ROC and Precision-Recall Curves")
print("6. Confidence Score Distributions")
print("7. Duration vs Confidence Analysis")
print("8. Summary Statistics Table")