import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

# Set style for high-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# =============================================================================
# CONFIGURATION SECTION - EDIT THESE TO CUSTOMIZE YOUR ANALYSIS
# =============================================================================

log_dir = "."
TARGET_STEPS = list(range(31, 32))  # Analyze steps 94-143


# Events to analyze
EVENTS = [
    ("async_rollout_request_complete", "Total Request", "#C73E1D"),
    # Add more events here if needed:
    # ("engine_async_generate", "Model Inference", "#2E86AB"),
    # ("tool_execution", "Tool Execution", "#A23B72"),
    # ("barrier_wait_duration", "Barrier Wait", "#F18F01"),
]

# Steps to analyze for detailed CDF (set to None to analyze all steps)

# Output directory for plots
OUTPUT_DIR = "zpics"

# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def analyze_event_durations(log_dir, target_steps=None):
    """
    Analyze event durations for specified steps.
    
    Args:
        log_dir: Directory containing log files
        target_steps: List of steps to analyze, or None for all steps
    """
    if target_steps is None:
        print("📊 Analyzing event durations for all steps...")
        durations_dict = _collect_event_durations_by_step(log_dir, None)
        _plot_enhanced_cdf_by_step(durations_dict, None)
    else:
        print(f"📊 Analyzing event durations for {len(target_steps)} specific steps...")
        for target_step in target_steps:
            print(f"\n  Analyzing step {target_step}...")
            durations_dict = _collect_event_durations_by_step(log_dir, target_step)
            
            # Print statistics for this step
            for event, label, color in EVENTS:
                count = len(durations_dict[event])
                if count > 0:
                    data = np.array(durations_dict[event])
                    print(f"    {label}: {count:,} samples (mean: {np.mean(data):.2f}s, "
                          f"p95: {np.percentile(data, 95):.2f}s, p99: {np.percentile(data, 99):.2f}s)")
                else:
                    print(f"    {label}: No data found")
                    break
            
            # Generate CDF plot for this step
            _plot_enhanced_cdf_by_step(durations_dict, target_step)

def main():
    """Main analysis pipeline with CDF visualization."""
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("🚀 Starting CDF Performance Analysis...")
    print("=" * 50)
    
    # Event duration analysis
    analyze_event_durations(log_dir, TARGET_STEPS)
    
    print("\n✅ Analysis complete! Check the generated PNG files for detailed visualizations.")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _collect_event_durations_by_step(log_dir, target_step=None):
    """Collect duration data for events in a specific step or all steps."""
    if target_step is not None:
        durations = {event: [] for event, _, _ in EVENTS}
        step_dir = os.path.join(log_dir, f"step_{target_step}")
        if os.path.exists(step_dir):
            for file in os.listdir(step_dir):
                if file.endswith('.jsonl'):
                    file_path = os.path.join(step_dir, file)
                    with open(file_path, 'r') as f:
                        for line in f:
                            try:
                                entry = json.loads(line)
                                event = entry.get("event")
                                if event in durations:
                                    durations[event].append(entry.get("duration_sec", 0))
                            except Exception:
                                continue
        return durations
    else:
        durations = {event: [] for event, _, _ in EVENTS}
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if file.endswith('.jsonl'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        for line in f:
                            try:
                                entry = json.loads(line)
                                event = entry.get("event")
                                if event in durations:
                                    durations[event].append(entry.get("duration_sec", 0))
                            except Exception:
                                continue
        return durations

def _plot_enhanced_cdf_by_step(durations_dict, step_num=None):
    """Plot enhanced CDF with comprehensive percentile analysis for a specific step."""
    percentiles = [50, 80, 90, 95, 99, 99.9]
    percentile_colors = ['#FFD23F', '#EE6A50', '#CD853F', '#9370DB', '#FF1493', '#DC143C']
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot CDFs for each event
    for idx, (event, label, color) in enumerate(EVENTS):
        data = np.array(durations_dict[event])
        if len(data) == 0:
            continue
            
        data = np.sort(data)
        cdf = np.arange(1, len(data) + 1) / len(data)
        
        # Main CDF line
        ax.plot(data, cdf, label=label, color=color, linewidth=3, alpha=0.8)
        
        # Add percentile lines only for the primary event (first one)
        if idx == 0:
            perc_values = np.percentile(data, percentiles)
            for perc, val, p_color in zip(percentiles, perc_values, percentile_colors):
                ax.axvline(val, linestyle='--', color=p_color, linewidth=2.5, alpha=0.8,
                          label=f"P{perc}: {val:.2f}s")
                
                # Add value annotation
                ax.annotate(f'{val:.1f}s', 
                           xy=(val, 0.5), 
                           xytext=(5, 0), 
                           textcoords='offset points',
                           rotation=90, 
                           verticalalignment='center',
                           fontweight='bold',
                           fontsize=10,
                           color=p_color)
    
    # Enhanced styling
    ax.set_xlabel("Duration (seconds)", fontsize=16, fontweight='bold')
    ax.set_ylabel("Cumulative Distribution Function (CDF)", fontsize=16, fontweight='bold')
    
    # Set title based on whether specific step is analyzed
    if step_num is not None:
        title = f"Performance Distribution Analysis - Step {step_num}\nCDF of Different Event Types"
        save_path = os.path.join(OUTPUT_DIR, f"enhanced_cdf_step_{step_num}.png")
    else:
        title = "Performance Distribution Analysis - All Steps\nCDF of Different Event Types"
        save_path = os.path.join(OUTPUT_DIR, "enhanced_cdf_all_steps.png")
    
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    
    # Grid and layout
    ax.grid(True, linestyle=':', alpha=0.6, linewidth=1)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    
    # Enhanced legend
    legend1 = ax.legend(loc="lower right", fontsize=12, frameon=True, 
                       fancybox=True, framealpha=0.95, shadow=True,
                       title="Event Types & Percentiles", title_fontsize=13)
    legend1.get_title().set_fontweight('bold')
    
    # Add summary statistics box
    primary_event = EVENTS[0][0]
    if primary_event in durations_dict and len(durations_dict[primary_event]) > 0:
        data = np.array(durations_dict[primary_event])
        step_text = f"Step {step_num}" if step_num is not None else "All Steps"
        stats_text = f"""Performance Summary ({EVENTS[0][1]}) - {step_text}:
        Samples: {len(data):,}
        Mean: {np.mean(data):.2f}s
        Median: {np.median(data):.2f}s
        Std: {np.std(data):.2f}s
        Min: {np.min(data):.2f}s
        Max: {np.max(data):.2f}s"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=11, fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to: {save_path}")

if __name__ == "__main__":
    main()