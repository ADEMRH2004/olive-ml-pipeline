"""
Enhanced Olive ML Pipeline — IoU Metrics & Visualizations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_iou_from_cm(cm):
    """Calculate IoU from confusion matrix"""
    ious = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        ious.append(iou)
    return np.array(ious)

def calculate_mean_iou(cm):
    """Calculate mean IoU"""
    ious = calculate_iou_from_cm(cm)
    return np.mean(ious)

def load_and_enhance_metrics(metrics_path="./models/metrics.json"):
    """Load metrics and add IoU calculations"""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    enhanced_metrics = []

    for metric in metrics:
        enhanced = metric.copy()

        if 'cm' in metric and metric['cm']:
            cm = np.array(metric['cm'])
            ious = calculate_iou_from_cm(cm)
            mean_iou = calculate_mean_iou(cm)

            enhanced['ious'] = ious.tolist()
            enhanced['mean_iou'] = mean_iou

            if 'cm_labels' in metric:
                iou_dict = dict(zip(metric['cm_labels'], ious))
                enhanced['iou_by_class'] = iou_dict

        enhanced_metrics.append(enhanced)

    return enhanced_metrics

def create_simple_visualization(metrics, save_path="./plots/iou_f1_comparison.png"):
    """Create simple IoU vs F1 visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Extract data
    zones = ['SFAX', 'SAHEL', 'MIDWEST', 'NORTH', 'SOUTH']
    intensity_f1 = []
    intensity_iou = []

    for zone in zones:
        intensity_metric = next((m for m in metrics if m['zone'] == zone and m['stage'] == 'intensity'), None)
        if intensity_metric:
            intensity_f1.append(intensity_metric.get('macro_f1', 0))
            intensity_iou.append(intensity_metric.get('mean_iou', 0))
        else:
            intensity_f1.append(0)
            intensity_iou.append(0)

    # Scatter plot
    ax1.scatter(intensity_f1, intensity_iou, s=100, alpha=0.7, color='blue')
    for i, zone in enumerate(zones):
        ax1.annotate(zone, (intensity_f1[i], intensity_iou[i]), xytext=(5, 5), textcoords='offset points')
    ax1.set_xlabel('Macro F1 Score')
    ax1.set_ylabel('Mean IoU')
    ax1.set_title('F1 vs IoU Correlation')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Bar chart comparison
    x = np.arange(len(zones))
    width = 0.35

    ax2.bar(x - width/2, intensity_f1, width, label='Macro F1', alpha=0.8, color='lightgreen')
    ax2.bar(x + width/2, intensity_iou, width, label='Mean IoU', alpha=0.8, color='orange')
    ax2.set_xlabel('Zones')
    ax2.set_ylabel('Score')
    ax2.set_title('Intensity Classification: F1 vs IoU')
    ax2.set_xticks(x)
    ax2.set_xticklabels(zones)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 IoU vs F1 visualization saved → {save_path}")

def create_metrics_summary(metrics):
    """Print enhanced metrics summary"""
    print("\n" + "="*80)
    print("ENHANCED METRICS SUMMARY (with IoU)")
    print("="*80)

    for zone in ['SFAX', 'SAHEL', 'MIDWEST', 'NORTH', 'SOUTH']:
        print(f"\n🏞️  ZONE: {zone}")
        print("-" * 40)

        # Binary metrics
        binary = next((m for m in metrics if m['zone'] == zone and m['stage'] == 'binary'), None)
        if binary:
            print(f"  🔍 Binary F1: {binary.get('f1_binary', 0):.3f}")
            print(f"  📈 Binary AUC: {binary.get('auc', 0):.3f}")

        # Intensity metrics
        intensity = next((m for m in metrics if m['zone'] == zone and m['stage'] == 'intensity'), None)
        if intensity:
            print(f"  🎯 Intensity Macro F1: {intensity.get('macro_f1', 0):.3f}")
            print(f"  📐 Intensity Mean IoU: {intensity.get('mean_iou', 0):.3f}")
            if 'iou_by_class' in intensity:
                print("  Class IoU:")
                for cls, iou_val in intensity['iou_by_class'].items():
                    print(f"    {cls}: {iou_val:.3f}")
        else:
            print("  ❌ No intensity classification (insufficient data)")

def main():
    """Main execution"""
    print("🚀 Enhanced Olive ML Pipeline - IoU Metrics & Visualizations")
    print("="*70)

    # Load and enhance metrics
    print("\n📊 Loading and enhancing metrics with IoU calculations...")
    metrics = load_and_enhance_metrics()

    # Save enhanced metrics
    with open('./models/enhanced_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✅ Enhanced metrics saved → ./models/enhanced_metrics.json")

    # Print summary
    create_metrics_summary(metrics)

    # Create visualizations
    print("\n📈 Creating IoU vs F1 visualization...")
    create_simple_visualization(metrics)

    print("\n✅ Enhanced analysis complete!")
    print("   📊 Check ./plots/iou_f1_comparison.png for visualizations")
    print("   📋 Check ./models/enhanced_metrics.json for detailed IoU metrics")

if __name__ == "__main__":
    main()
