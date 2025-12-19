#!/usr/bin/env python3
"""
Plot Performance Metrics for Pose Estimation System

This script loads performance metrics from JSON files and generates:
1. Distribution plots (histograms) for each metric
2. Bar charts comparing average metrics across implementations
3. Summary statistics CSV

Metrics are collected from:
- FoundationPose: foundationpose_metrics.json
- Grounded SAM: grounded_sam_metrics.json
- YOLO+SAM: yolo_sam_metrics.json (if available)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from pathlib import Path

# Try to import seaborn for better styling (optional)
try:
    import seaborn as sns
    HAS_SEABORN = True
    # Set seaborn style for better-looking plots
    sns.set_style("whitegrid")
    sns.set_palette("husl")
except ImportError:
    HAS_SEABORN = False
    print("Note: seaborn not available, using matplotlib defaults")

# Set matplotlib parameters for better-looking plots
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# Color palette for implementations
COLORS = {
    "FoundationPose": "#2E86AB",  # Blue
    "Grounded SAM": "#A23B72",    # Purple
    "YOLO+SAM": "#F18F01"          # Orange
}

# Configuration
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
PLOTS_DIR = SCRIPT_DIR / "plots"
CSV_DIR = SCRIPT_DIR / "data"

# Create plots directory if it doesn't exist
PLOTS_DIR.mkdir(exist_ok=True)

# Metric files to load
METRIC_FILES = {
    "FoundationPose": "foundationpose_metrics.json",
    "Grounded SAM": "grounded_sam_metrics.json",
    "YOLO+SAM": "yolo_sam_metrics.json"
}

# Metrics to plot
METRICS = {
    "time_s": {
        "label": "Processing Time (seconds)",
        "unit": "s",
        "log_scale": True
    },
    "time_ms": {
        "label": "Processing Time (milliseconds)",
        "unit": "ms",
        "log_scale": True
    },
    "gpu_memory_allocated_gb": {
        "label": "GPU Memory Allocated (GB)",
        "unit": "GB",
        "log_scale": False
    },
    "gpu_memory_reserved_gb": {
        "label": "GPU Memory Reserved (GB)",
        "unit": "GB",
        "log_scale": False
    },
    "gpu_memory_allocated_pct": {
        "label": "GPU Memory Allocated (%)",
        "unit": "%",
        "log_scale": False
    },
    "gpu_utilization_pct": {
        "label": "GPU Utilization (%)",
        "unit": "%",
        "log_scale": False
    },
    "cpu_process_pct": {
        "label": "CPU Usage - Process (%)",
        "unit": "%",
        "log_scale": False
    },
    "cpu_system_pct": {
        "label": "CPU Usage - System (%)",
        "unit": "%",
        "log_scale": False
    },
    "memory_mb": {
        "label": "Memory Usage (MB)",
        "unit": "MB",
        "log_scale": False
    }
}


def load_metrics(filepath):
    """Load metrics from JSON file."""
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def extract_metric_values(data, metric_key):
    """Extract values for a specific metric from data."""
    if not data:
        return []
    
    values = []
    for entry in data:
        if metric_key in entry and entry[metric_key] is not None:
            value = entry[metric_key]
            # Filter out invalid values
            if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                values.append(value)
    
    return values


def plot_distributions(all_data):
    """Create distribution plots (histograms) for each metric."""
    print("Generating distribution plots...")
    
    # Reorder implementations: YOLO+SAM, Grounded SAM, FoundationPose
    preferred_order = ["YOLO+SAM", "Grounded SAM", "FoundationPose"]
    ordered_data = sorted(all_data.items(), 
                         key=lambda x: preferred_order.index(x[0]) if x[0] in preferred_order else 999)
    
    for metric_key, metric_info in METRICS.items():
        fig, axes = plt.subplots(1, len(ordered_data), figsize=(6 * len(ordered_data), 5))
        if len(ordered_data) == 1:
            axes = [axes]
        
        fig.suptitle(f'Distribution: {metric_info["label"]}', fontsize=14, fontweight='bold', y=1.02)
        
        for idx, (name, data) in enumerate(ordered_data):
            values = extract_metric_values(data, metric_key)
            
            if not values:
                axes[idx].text(0.5, 0.5, f'No data for {name}', 
                              ha='center', va='center', transform=axes[idx].transAxes,
                              fontsize=12, style='italic')
                axes[idx].set_title(f'{name}', fontweight='bold')
                axes[idx].set_facecolor('#f8f8f8')
                continue
            
            # Create histogram with better styling
            color = COLORS.get(name, None)
            if color is None:
                # Fallback color palette if name not in COLORS
                if HAS_SEABORN:
                    color = sns.color_palette("husl")[idx]
                else:
                    colors_list = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#D4A574']
                    color = colors_list[idx % len(colors_list)]
            axes[idx].hist(values, bins=30, edgecolor='white', linewidth=1.2, 
                         alpha=0.8, color=color, density=False)
            
            # Add vertical lines for mean and median
            mean_val = np.mean(values)
            median_val = np.median(values)
            axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                            label=f'Mean: {mean_val:.2f}')
            axes[idx].axvline(median_val, color='green', linestyle='--', linewidth=2, 
                            label=f'Median: {median_val:.2f}')
            
            axes[idx].set_title(f'{name}', fontweight='bold', pad=10)
            # Check if unit is already in label to avoid duplication
            if f'({metric_info["unit"]})' in metric_info["label"]:
                axes[idx].set_xlabel(f'{metric_info["label"]}', fontweight='bold')
            else:
                axes[idx].set_xlabel(f'{metric_info["label"]} ({metric_info["unit"]})', fontweight='bold')
            axes[idx].set_ylabel('Frequency', fontweight='bold')
            axes[idx].grid(True, alpha=0.3, linestyle='--')
            
            # Add statistics text box in upper right
            std_val = np.std(values)
            stats_text = f'N: {len(values)}\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}'
            axes[idx].text(0.98, 0.98, stats_text, 
                          transform=axes[idx].transAxes,
                          verticalalignment='top',
                          horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='white', 
                                  edgecolor='gray', alpha=0.8, pad=0.5),
                          fontsize=9, family='monospace')
            
            # Add legend in upper left to avoid overlap with stats box
            axes[idx].legend(loc='upper left', framealpha=0.9, fontsize=9, 
                           bbox_to_anchor=(0.02, 0.98), frameon=True)
            
            if metric_info["log_scale"]:
                axes[idx].set_xscale('log')
        
        plt.tight_layout()
        plot_filename = PLOTS_DIR / f"distribution_{metric_key}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {plot_filename}")


def plot_comparison_bars(all_data):
    """Create bar charts comparing average metrics across implementations."""
    print("Generating comparison bar charts...")
    
    # Calculate averages for each implementation
    comparison_data = {}
    
    for metric_key, metric_info in METRICS.items():
        comparison_data[metric_key] = {}
        for name, data in all_data.items():
            values = extract_metric_values(data, metric_key)
            if values:
                comparison_data[metric_key][name] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
    
    # Create bar charts for key metrics
    # Note: Some implementations may only have time metrics (e.g., Grounded SAM)
    # We create plots for all metrics that have data from at least one implementation
    key_metrics = ['time_s', 'time_ms', 'gpu_memory_allocated_gb', 'gpu_utilization_pct', 
                   'cpu_process_pct', 'memory_mb']
    
    for metric_key in key_metrics:
        if metric_key not in comparison_data:
            continue
        
        metric_info = METRICS[metric_key]
        implementations = list(comparison_data[metric_key].keys())
        
        # Skip if no implementations have this metric
        if not implementations:
            continue
        
        # Reorder implementations: YOLO+SAM, Grounded SAM, FoundationPose
        preferred_order = ["YOLO+SAM", "Grounded SAM", "FoundationPose"]
        implementations = sorted(implementations, 
                                key=lambda x: preferred_order.index(x) if x in preferred_order else 999)
        
        # For time metrics, include all implementations (even if they only have time data)
        # For GPU/CPU metrics, only include implementations that have those metrics
        if metric_key in ['time_s', 'time_ms']:
            # Include all implementations that have time data
            pass  # Already filtered by comparison_data
        else:
            # For GPU/CPU metrics, only include if at least 2 implementations have data
            # (to make comparison meaningful)
            if len(implementations) < 2:
                print(f"  Skipping {metric_key}: only {len(implementations)} implementation(s) have this metric")
                continue
        
        means = [comparison_data[metric_key][impl]['mean'] for impl in implementations]
        stds = [comparison_data[metric_key][impl]['std'] for impl in implementations]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(implementations))
        
        # Use colors from palette
        colors = []
        for i, impl in enumerate(implementations):
            if impl in COLORS:
                colors.append(COLORS[impl])
            elif HAS_SEABORN:
                colors.append(sns.color_palette("husl")[i])
            else:
                colors_list = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#D4A574']
                colors.append(colors_list[i % len(colors_list)])
        
        # Create bars with better styling
        bars = ax.bar(x_pos, means, yerr=stds, capsize=8,
                     alpha=0.85, edgecolor='black', linewidth=1.5,
                     color=colors, width=0.6,
                     error_kw={'elinewidth': 2})
        
        # Add value labels on bars with better formatting
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            # Position label above error bar
            label_y = height + std + (max(means) * 0.02)  # Small offset
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{mean:.2f}\n±{std:.2f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           edgecolor='gray', alpha=0.8))
        
        ax.set_xlabel('Implementation', fontweight='bold', fontsize=12)
        # Check if unit is already in label to avoid duplication
        if f'({metric_info["unit"]})' in metric_info["label"]:
            ax.set_ylabel(f'{metric_info["label"]}', fontweight='bold', fontsize=12)
        else:
            ax.set_ylabel(f'{metric_info["label"]} ({metric_info["unit"]})', 
                         fontweight='bold', fontsize=12)
        ax.set_title(f'Average {metric_info["label"]} Comparison', 
                    fontweight='bold', fontsize=13, pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(implementations, rotation=0, ha='center', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        if metric_info["log_scale"]:
            ax.set_yscale('log')
        
        plt.tight_layout()
        plot_filename = PLOTS_DIR / f"comparison_{metric_key}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {plot_filename}")


def save_summary_statistics(all_data):
    """Save summary statistics to CSV."""
    print("Generating summary statistics CSV...")
    
    summary_rows = []
    
    for name, data in all_data.items():
        for metric_key, metric_info in METRICS.items():
            values = extract_metric_values(data, metric_key)
            
            if not values:
                continue
            
            summary_rows.append({
                'Implementation': name,
                'Metric': metric_info["label"],
                'Unit': metric_info["unit"],
                'Count': len(values),
                'Mean': np.mean(values),
                'Median': np.median(values),
                'Std': np.std(values),
                'Min': np.min(values),
                'Max': np.max(values),
                'Q25': np.percentile(values, 25),
                'Q75': np.percentile(values, 75)
            })
    
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_filename = CSV_DIR / "metrics_summary.csv"
        df.to_csv(csv_filename, index=False)
        print(f"  Saved: {csv_filename}")
        print(f"\nSummary Statistics Preview:")
        print(df.to_string(index=False))
    else:
        print("  No data to save")


def main():
    """Main function to generate all plots and statistics."""
    print("=" * 60)
    print("Performance Metrics Plotting Script")
    print("=" * 60)
    
    # Load all metric files
    all_data = {}
    for name, filename in METRIC_FILES.items():
        filepath = DATA_DIR / filename
        data = load_metrics(filepath)
        if data:
            all_data[name] = data
            print(f"Loaded {len(data)} entries from {name}")
        else:
            print(f"Warning: Could not load {name} from {filepath}")
    
    if not all_data:
        print("Error: No metric data found. Please run the pose estimation nodes first.")
        return
    
    print(f"\nFound {len(all_data)} implementation(s) with data")
    print("-" * 60)
    
    # Generate plots
    plot_distributions(all_data)
    print()
    plot_comparison_bars(all_data)
    print()
    save_summary_statistics(all_data)
    
    print("\n" + "=" * 60)
    print("Plotting complete!")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"Summary CSV saved to: {CSV_DIR / 'metrics_summary.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
