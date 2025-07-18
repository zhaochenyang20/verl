import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

log_dir = "."

def main():
    """Main function to run the analysis"""
    # Example usage

    for i in range(0,8):
        log_file_path = log_dir + f"/step_32/worker_{i}.jsonl"
        if not Path(log_file_path).exists():
            print(f"Log file not found: {log_file_path}")
            print("Please update the log_file_path variable with the correct path to your log file")
            return
        print(f"Parsing log file: {log_file_path}")
        timing_data = parse_log_file(log_file_path)
        output_path = "zpics/"+log_file_path.replace("/", "_").replace(".jsonl", "_timing_analysis.png") # Optional: specify output path
        create_timing_chart(timing_data, output_path, filename=log_file_path)

def parse_log_file(log_file_path):
    """Parse JSONL log file and extract timing data"""
    timing_data = {}
    
    with open(log_file_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                event = entry.get('event')
                duration = entry.get('duration_sec')
                
                if event and duration is not None:
                    timing_data[event] = duration
            except json.JSONDecodeError:
                continue
    
    return timing_data

def create_timing_chart(timing_data, output_path=None, figsize=(15, 8), filename=None):
    """Create a bar chart showing timing for each step"""
    
    # Define the order of key steps we want to show
    key_steps = [
        'preprocessing_duration',
        'async_generate_duration', 
        'sorting_duration',
        'barrier_wait_duration',
        'broadcast_duration',
        'data_extraction_duration',
        'padding_duration',
        'concatenation_duration',
        'batch_construction_duration',
        'cache_flush_duration',
        'final_construction_duration',
        'total_step_duration'
    ]
    
    # Filter and order the data
    filtered_data = {}
    for step in key_steps:
        if step in timing_data:
            filtered_data[step] = timing_data[step]
    
    # Add any other timing events not in the predefined list
    for event, duration in timing_data.items():
        if event not in filtered_data and 'duration' in event:
            filtered_data[event] = duration
    
    if not filtered_data:
        print("No timing data found in the log file")
        return
    
    # Prepare data for plotting
    events = list(filtered_data.keys())
    durations = list(filtered_data.values())
    
    # Create the plot
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(events)), durations, color='steelblue', alpha=0.7)
    
    # Customize the plot with filename in title
    title = 'SGLang Rollout Step Timing Analysis'
    if filename:
        title += f' - {filename}'
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Processing Steps', fontsize=12)
    plt.ylabel('Duration (seconds)', fontsize=12)
    
    # Set x-axis labels
    plt.xticks(range(len(events)), events, rotation=45, ha='right')
    
    # Add value labels on top of each bar
    for i, (bar, duration) in enumerate(zip(bars, durations)):
        height = bar.get_height()
        if height > 1:
            label = f'{height:.2f}s'
        elif height > 0.01:
            label = f'{height:.3f}s'
        else:
            label = f'{height:.4f}s'
            
        plt.text(bar.get_x() + bar.get_width()/2., height + max(durations) * 0.01,
                label, ha='center', va='bottom', fontsize=9, rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {output_path}")
    else:
        plt.show()
    
    # Print summary statistics
    print(f"\nTiming Summary:")
    print(f"Total duration: {filtered_data.get('total_step_duration', 'N/A'):.3f}s")
    longest_step = max(filtered_data.items(), key=lambda x: x[1])
    shortest_step = min(filtered_data.items(), key=lambda x: x[1])
    print(f"Longest step: {longest_step[0]} ({longest_step[1]:.3f}s)")
    print(f"Shortest step: {shortest_step[0]} ({shortest_step[1]:.6f}s)")




if __name__ == "__main__":
    main()
    