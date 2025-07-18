import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import seaborn as sns

# Set style for high-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

log_dir = "."
OUTPUT_DIR = "zpics"

def main():
    """Main analysis pipeline with enhanced visualizations."""
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("🚀 Starting Enhanced Performance Analysis...")
    print("=" * 50)
    
    # Step maximum duration analysis
    analyze_step_max_durations(log_dir)
    
    print("\n✅ Analysis complete! Check the generated PNG files for detailed visualizations.")

def analyze_step_max_durations(log_dir):
    """Analyze maximum step durations across all workers."""
    print("⏱️  Analyzing step maximum durations...")
    steps, max_durations = _collect_step_max_total(log_dir)
    
    # Get step start times for interval calculation
    interval_steps, interval_times = _collect_step_start_times(log_dir)
    intervals = []
    interval_step_nums = []
    
    if len(interval_times) >= 2:
        # Calculate intervals: step(i+1) start time - step(i) start time
        # The interval for step i represents the time from step i start to step i+1 start
        intervals = [(interval_times[i+1] - interval_times[i]).total_seconds() for i in range(len(interval_times)-1)]
        interval_step_nums = interval_steps[:-1]  # These are the steps for which we have intervals (step i)
    
    if steps and max_durations:
        print(f"  Steps analyzed: {len(steps)}")
        print(f"  Duration range: {min(max_durations):.2f}s - {max(max_durations):.2f}s")
        _plot_enhanced_step_max_with_intervals(steps, max_durations, interval_step_nums, intervals)
    else:
        print("  No step duration data found")

def _collect_step_start_times(log_dir):
    """Collect the earliest actual start time for each step across all workers.
    
    The start time is calculated as the first timestamp minus its duration_sec.
    For example, if the first record is:
    {"timestamp": "2025-07-13T11:04:25.706221", "event": "preprocessing_duration", "duration_sec": 13.30946397781372}
    Then the actual start time is 2025-07-13T11:04:25.706221 - 13.30946397781372 seconds
    """
    step_start_times = {}
    
    for root, dirs, files in os.walk(log_dir):
        if not os.path.basename(root).startswith("step_"):
            continue
        
        step_str = os.path.basename(root).replace("step_", "")
        try:
            step = int(step_str)
        except Exception:
            continue
        
        min_start_time = None
        
        for file in files:
            if file.endswith('.jsonl'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r') as f:
                        # Read the first line to get the first timestamp and duration
                        first_line = f.readline().strip()
                        if first_line:
                            entry = json.loads(first_line)
                            timestamp_str = entry.get("timestamp")
                            duration_sec = entry.get("duration_sec", 0)
                            
                            if timestamp_str:
                                # Parse timestamp
                                timestamp = datetime.fromisoformat(timestamp_str)
                                
                                # Calculate actual start time by subtracting duration
                                from datetime import timedelta
                                actual_start_time = timestamp - timedelta(seconds=duration_sec)
                                
                                # Keep track of the earliest start time across all workers
                                if min_start_time is None or actual_start_time < min_start_time:
                                    min_start_time = actual_start_time
                                    
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue
        
        if min_start_time:
            step_start_times[step] = min_start_time
    
    # Sort by step number and return
    steps = sorted(step_start_times.keys())
    times = [step_start_times[step] for step in steps]
    
    # Debug output
    print(f"\n🔍 Step Start Times Analysis:")
    print(f"  Found {len(steps)} steps with start times")
    if len(steps) > 0:
        print(f"  First step: {steps[0]} at {times[0]}")
        print(f"  Last step: {steps[-1]} at {times[-1]}")
        if len(times) >= 2:
            total_duration = (times[-1] - times[0]).total_seconds()
            print(f"  Total duration: {total_duration:.2f} seconds")
    
    return steps, times

def _collect_step_max_total(log_dir):
    """Collect maximum total_step_duration for each step across workers."""
    step_worker_duration = defaultdict(dict)
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith('.jsonl'):
                file_path = os.path.join(root, file)
                try:
                    step = int(os.path.basename(os.path.dirname(file_path)).replace("step_", ""))
                    worker = file.replace(".jsonl", "")
                except Exception:
                    continue
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if entry.get("event") == "total_step_duration":
                                step_worker_duration[step][worker] = entry.get("duration_sec", 0)
                        except Exception:
                            continue
    
    steps = sorted(step_worker_duration.keys())
    max_durations = [max(step_worker_duration[step].values()) for step in steps]
    return steps, max_durations

def _plot_enhanced_step_max_with_intervals(steps, max_durations, interval_steps, intervals):
    """Plot step maximum durations with step intervals overlay using same y-axis for comparison."""
    # Create figure with more space for stats
    fig = plt.figure(figsize=(21, 10))
    
    # Create main plot area (leave space on the right for stats)
    ax = plt.subplot2grid((1, 4), (0, 0), colspan=3)
    
    # Use single y-axis for both metrics (both are in seconds)
    color1 = '#C73E1D'
    color2 = '#2E86AB'
    ax.set_xlabel("Step", fontsize=16, fontweight='bold')
    ax.set_ylabel("Time (seconds)", fontsize=16, fontweight='bold')
    
    # Print data to console
    print("\n" + "="*60)
    print("📊 DETAILED DATA ANALYSIS")
    print("="*60)
    print(f"Step Internal Duration (Red line):")
    print(f"Steps: {steps}")
    print(f"Durations: {[f'{d:.2f}' for d in max_durations]}")
    
    if intervals and interval_steps:
        print(f"\nStep-to-Next-Step Start Time Intervals (Blue line):")
        print(f"Steps: {interval_steps}")
        print(f"Intervals (step i to step i+1): {[f'{i:.2f}' for i in intervals]}")
    
    print("\n" + "="*60)
    
    # Plot 1: Step internal durations
    line1 = ax.plot(steps, max_durations, marker='o', linewidth=3, 
                     markersize=10, color=color1, markerfacecolor='white', 
                     markeredgewidth=3, markeredgecolor=color1, alpha=0.9, 
                     label='Step Internal Duration (Max across workers)')
    
    # Add fill under curve for durations
    ax.fill_between(steps, max_durations, alpha=0.15, color=color1)
    
    # Plot 2: Step intervals (same y-axis)
    if intervals and interval_steps:
        line2 = ax.plot(interval_steps, intervals, marker='s', linewidth=3, 
                        markersize=10, color=color2, markerfacecolor='white', 
                        markeredgewidth=3, markeredgecolor=color2, alpha=0.9,
                        label='Step i to Step i+1 Start Time Interval')
        
        # Add fill for intervals
        ax.fill_between(interval_steps, intervals, alpha=0.15, color=color2)
    
    # Add value annotations for every 5th point to make it readable
    for i, (step, duration) in enumerate(zip(steps, max_durations)):
        if i % 5 == 0:  # Show every 5th point
            ax.annotate(f'{duration:.1f}', 
                       xy=(step, duration), 
                       xytext=(0, 10), 
                       textcoords='offset points',
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold',
                       color=color1, alpha=0.8)
    
    if intervals and interval_steps:
        for i, (step, interval) in enumerate(zip(interval_steps, intervals)):
            if i % 5 == 0:  # Show every 5th point
                ax.annotate(f'{interval:.1f}', 
                           xy=(step, interval), 
                           xytext=(0, -15), 
                           textcoords='offset points',
                           ha='center', va='top',
                           fontsize=9, fontweight='bold',
                           color=color2, alpha=0.8)
    
    # Highlight outliers for step durations
    mean_duration = np.mean(max_durations)
    std_duration = np.std(max_durations)
    
    for step, duration in zip(steps, max_durations):
        if abs(duration - mean_duration) > 2 * std_duration:
            ax.annotate(f'{duration:.1f}s\n(outlier)', 
                        xy=(step, duration), 
                        xytext=(0, 20), 
                        textcoords='offset points',
                        ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='red'))
    
    # Highlight outliers for intervals
    if intervals and interval_steps:
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        for step, interval in zip(interval_steps, intervals):
            if abs(interval - mean_interval) > 2 * std_interval:
                ax.annotate(f'{interval:.1f}s\n(outlier)', 
                            xy=(step, interval), 
                            xytext=(10, -20), 
                            textcoords='offset points',
                            ha='center',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                            arrowprops=dict(arrowstyle='->', color='yellow'))
    
    ax.grid(True, alpha=0.3)
    
    # Add vertical reference lines at key intervals
    reference_steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 
                       210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 
                       410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 
                       610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 
                       810, 820, 830, 840, 850]
    for ref_step in reference_steps:
        if ref_step <= max(max(steps), max(interval_steps) if interval_steps else 0):
            ax.axvline(x=ref_step, color='gray', linestyle='--', alpha=0.6, linewidth=2)
            # Add label at the top
            ax.text(ref_step, ax.get_ylim()[1] * 0.95, f'Step {ref_step}', 
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Enhanced styling with updated title
    ax.set_title("Step Performance Comparison: Internal Duration vs Step-to-Next Intervals\n" + 
                 "Red: Step i total duration (slowest worker) | Blue: Time from Step i start to Step i+1 start\n" +
                 "Both metrics on same scale for direct comparison", 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Create dedicated space for statistics on the right
    stats_ax = plt.subplot2grid((1, 4), (0, 3))
    stats_ax.axis('off')  # Hide axes
    
    # Prepare comprehensive statistics with comparison insights
    stats_text = f"""Step Internal Duration (Red):
Mean: {mean_duration:.2f}s
Std: {std_duration:.2f}s
Min: {min(max_durations):.2f}s
Max: {max(max_durations):.2f}s"""
    
    if intervals:
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        stats_text += f"""\n\nStep-to-Next Interval (Blue):
Mean: {mean_interval:.2f}s
Std: {std_interval:.2f}s
Min: {min(intervals):.2f}s
Max: {max(intervals):.2f}s"""
        
        # Add comparison insights
        if mean_interval > mean_duration:
            gap = mean_interval - mean_duration
            stats_text += f"""\n\n🔍 Analysis:
Gap: {gap:.2f}s (Interval > Duration)
Ratio: {mean_interval/mean_duration:.2f}x
Indicates: Coordination overhead"""
        else:
            gap = mean_duration - mean_interval
            stats_text += f"""\n\n🔍 Analysis:
Gap: {gap:.2f}s (Duration > Interval)
Ratio: {mean_duration/mean_interval:.2f}x
Indicates: Parallel processing"""
    
    # Place statistics in dedicated area
    stats_ax.text(0.05, 0.95, stats_text, transform=stats_ax.transAxes, 
                  verticalalignment='top', fontsize=11, fontfamily='monospace',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9))
    
    # Legend for main plot with updated label
    ax.legend(loc='upper right', fontsize=12, frameon=True, 
              fancybox=True, framealpha=0.95, shadow=True)
    
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "step_performance_comparison_step_to_next.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Plot saved to: {save_path}")


if __name__ == "__main__":
    main()