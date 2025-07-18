import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import seaborn as sns

log_dir = "."


def main():
    """Main function to run the analysis"""
    analyzer = RequestPhaseAnalyzer()
    
    # Update with your actual log file path
    log_file = log_dir + "/step_32/worker_0.jsonl"
    
    print(f"Parsing log file: {log_file}")
    analyzer.parse_log_file(log_file)

    print("\nGenerating request duration CDF...")
    cdf_fig = analyzer.plot_request_duration_cdf(save_path="zpics/request_duration_cdf.png")
    
    print(f"Found {len(analyzer.request_events)} requests with detailed events")
    
    analyzer.analyze_and_plot_top_requests(top_n=60, save_plots=True)


    

class RequestPhaseAnalyzer:
    def __init__(self):
        self.request_events = defaultdict(list)
        
    def parse_log_file(self, log_file_path):
        """Parse log file and organize events by request_id"""
        with open(log_file_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    
                    # Only process events with request_id
                    if 'extra' in entry and 'request_id' in entry.get('extra', {}):
                        request_id = entry['extra']['request_id']
                        
                        timestamp = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                        
                        event_data = {
                            'timestamp': timestamp,
                            'event': entry['event'],
                            'duration': entry.get('duration_sec', 0),
                            'extra': entry.get('extra', {}),
                            'workid': entry.get('workid'),
                            'step': entry.get('step')
                        }
                        
                        self.request_events[request_id].append(event_data)
                        
                except (json.JSONDecodeError, ValueError):
                    continue
    
    def find_slowest_requests(self, top_n=10):
        """Find the slowest requests based on async_rollout_request_complete duration"""
        request_durations = []
        
        for request_id, events in self.request_events.items():
            complete_event = next((e for e in events if e['event'] == 'async_rollout_request_complete'), None)
            if complete_event:
                duration = complete_event['duration']
                extra = complete_event['extra']
                request_durations.append({
                    'request_id': request_id,
                    'duration': duration,
                    'turns': extra.get('turns', 0),
                    'response_length': extra.get('response_length', 0),
                    'finish_reason': extra.get('finish_reason', 'unknown'),
                    'batch_data_id': extra.get('batch_data_id', 'unknown')
                })
        
        sorted_durations = sorted(request_durations, key=lambda x: x['duration'], reverse=True)
        
        # 硬编码的特定范围
        result = []
        result.extend(sorted_durations[50:60])  # 50-60
        result.extend(sorted_durations[90:95])  # 90-95  
        result.extend(sorted_durations[120:125])  # 120-125
        result.extend(sorted_durations[-2:])  # 150-155


        
        return result
    
    def analyze_request_phases(self, request_id):
        """Analyze phases for a specific request based on the _async_rollout_a_request function"""
        if request_id not in self.request_events:
            return None
            
        events = sorted(self.request_events[request_id], key=lambda x: x['timestamp'])
        
        # Define the phases based on the function structure
        phase_data = {
            'request_start': 0,
            'main_loop': 0,
            'pending_state_handling': 0,
            'tool_calling_state': 0,
            'tool_execution': 0,
            'tool_response_processing': 0,
            'tool_parsing': 0,
            'running_state': 0,
            'engine_call': 0,
            'interacting_state': 0,
            'interaction_response': 0,
            'reward_calculation': 0,
            'finalization': 0,
            'async_rollout_request_complete': 0
        }
        
        # Map events to phases and accumulate durations
        for event in events:
            event_name = event['event']
            duration = event['duration']
            
            if event_name in phase_data:
                phase_data[event_name] += duration
        
        # Get summary info from complete event
        complete_event = next((e for e in events if e['event'] == 'async_rollout_request_complete'), None)
        summary = {}
        if complete_event:
            extra = complete_event['extra']
            summary = {
                'total_duration': complete_event['duration'],
                'turns': extra.get('turns', 0),
                'response_length': extra.get('response_length', 0),
                'actual_response_tokens': extra.get('actual_response_tokens', 0),
                'total_sequence_length': extra.get('total_sequence_length', 0),
                'finish_reason': extra.get('finish_reason', 'unknown'),
                'batch_data_id': extra.get('batch_data_id', 'unknown')
            }
        
        return {
            'request_id': request_id,
            'phases': phase_data,
            'summary': summary,
            'events': events
        }
    
    def plot_request_phase_bar_chart(self, analysis, figsize=(14, 8)):
        """Create a bar chart showing phase durations for a specific request"""
        if not analysis:
            return None
            
        phases = analysis['phases']
        summary = analysis['summary']
        
        # Filter out phases with zero duration
        non_zero_phases = {k: v for k, v in phases.items() if v > 0.001}
        
        if not non_zero_phases:
            print(f"No meaningful phase data for request {analysis['request_id'][:8]}...")
            return None
        
        # Prepare data for plotting
        phase_names = list(non_zero_phases.keys())
        durations = list(non_zero_phases.values())
        
        # Create color map based on phase types
        color_map = {
            'request_start': '#FF6B6B',          # Red - Start
            'main_loop': '#4ECDC4',              # Teal - Main process
            'pending_state_handling': '#45B7D1',  # Blue - Setup
            'tool_calling_state': '#96CEB4',     # Green - Tool related
            'tool_execution': '#FFEAA7',         # Yellow - Tool execution
            'tool_response_processing': '#DDA0DD', # Purple - Tool response
            'tool_parsing': '#98D8C8',           # Light green - Tool parsing
            'running_state': '#F7DC6F',          # Light yellow - Running
            'engine_call': '#FF7675',            # Red - Engine call (most important)
            'interacting_state': '#A29BFE',      # Purple - Interaction
            'interaction_response': '#FD79A8',   # Pink - Interaction response
            'reward_calculation': '#FDCB6E',     # Orange - Reward
            'finalization': '#6C5CE7',           # Dark purple - Final
            'async_rollout_request_complete': '#2D3436'  # Dark - Complete
        }
        
        colors = [color_map.get(phase, '#95A5A6') for phase in phase_names]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(range(len(phase_names)), durations, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Customize the plot
        request_short = analysis['request_id'][:8]
        total_dur = summary.get('total_duration', 0)
        turns = summary.get('turns', 0)
        resp_len = summary.get('response_length', 0)
        finish_reason = summary.get('finish_reason', 'unknown').split('.')[-1] if '.' in summary.get('finish_reason', '') else summary.get('finish_reason', 'unknown')
        
        title = f'Request {request_short}... Phase Durations\n'
        title += f'Total: {total_dur:.2f}s | Turns: {turns} | Tokens: {resp_len} | Reason: {finish_reason}'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xlabel('Processing Phases', fontsize=12)
        ax.set_ylabel('Duration (seconds)', fontsize=12)
        
        # Set x-axis labels with rotation
        ax.set_xticks(range(len(phase_names)))
        ax.set_xticklabels([name.replace('_', '\n') for name in phase_names], 
                          rotation=45, ha='right', fontsize=10)
        
        # Add value labels on top of each bar
        for i, (bar, duration) in enumerate(zip(bars, durations)):
            height = bar.get_height()
            
            # Format duration label based on magnitude
            if height >= 1:
                label = f'{height:.2f}s'
            elif height >= 0.01:
                label = f'{height:.3f}s'
            else:
                label = f'{height:.4f}s'
            
            # Calculate percentage of total
            if total_dur > 0:
                percentage = (height / total_dur) * 100
                label += f'\n({percentage:.1f}%)'
            
            ax.text(bar.get_x() + bar.get_width()/2., height + max(durations) * 0.01,
                   label, ha='center', va='bottom', fontsize=9, weight='bold')
        
        # Add grid for better readability
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def generate_phase_summary_table(self, analysis):
        """Generate a summary table of phase durations"""
        if not analysis:
            return None
            
        phases = analysis['phases']
        summary = analysis['summary']
        total_duration = summary.get('total_duration', 0)
        
        # Create summary data
        summary_data = []
        for phase, duration in phases.items():
            if duration > 0.001:  # Only include meaningful durations
                percentage = (duration / total_duration * 100) if total_duration > 0 else 0
                summary_data.append({
                    'Phase': phase,
                    'Duration (s)': f'{duration:.4f}',
                    'Percentage': f'{percentage:.1f}%'
                })
        
        # Sort by duration
        summary_data.sort(key=lambda x: float(x['Duration (s)'].rstrip('s')), reverse=True)
        
        return pd.DataFrame(summary_data)
    
    def analyze_and_plot_top_requests(self, top_n=20, save_plots=True):
        """Analyze and plot phase breakdown for top N slowest requests"""
        slowest_requests = self.find_slowest_requests(top_n)
        
        if not slowest_requests:
            print("No request data found")
            return
        
        print(f"Analyzing top {len(slowest_requests)} slowest requests:")
        print("=" * 80)
        
        for i, req_info in enumerate(slowest_requests):
            print(f"\nRequest {i+1}: {req_info['request_id'][:8]}... ({req_info['duration']:.2f}s)")
            print(f"  - Turns: {req_info['turns']}")
            print(f"  - Response Length: {req_info['response_length']} tokens") 
            print(f"  - Finish Reason: {req_info['finish_reason']}")
            print(f"  - Batch Data ID: {req_info['batch_data_id']}")
            
            # Analyze phases
            analysis = self.analyze_request_phases(req_info['request_id'])
            
            if analysis:
                # Create plot
                fig = self.plot_request_phase_bar_chart(analysis)
                
                if fig and save_plots:
                    filename = f"zpics/request_{i+1}_{req_info['request_id'][:8]}_phases.png"
                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                    print(f"  - Plot saved as: {filename}")
                
                # Generate summary table
                summary_table = self.generate_phase_summary_table(analysis)
                if summary_table is not None:
                    print(f"  - Phase breakdown:")
                    for _, row in summary_table.head(5).iterrows():  # Show top 5 phases
                        print(f"    {row['Phase']}: {row['Duration (s)']} ({row['Percentage']})")
                
                if fig:
                    plt.show()
            else:
                print(f"  - No detailed phase data available")
            
            print("-" * 60)

    def plot_request_duration_cdf(self, figsize=(12, 8), save_path=None):
        """Create a CDF plot of request durations"""
        durations = []
        
        # Collect all request durations
        for request_id, events in self.request_events.items():
            complete_event = next((e for e in events if e['event'] == 'async_rollout_request_complete'), None)
            if complete_event:
                durations.append(complete_event['duration'])
        
        if not durations:
            print("No request duration data found")
            return None
        
        # Sort durations for CDF
        sorted_durations = np.sort(durations)
        n = len(sorted_durations)
        
        # Calculate CDF values (percentiles)
        cdf_values = np.arange(1, n + 1) / n
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(sorted_durations, cdf_values, linewidth=2, color='steelblue', marker='o', markersize=3, alpha=0.7)
        
        # Customize the plot
        ax.set_xlabel('Request Duration (seconds)', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.set_title('CDF of Request Processing Durations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add percentile lines
        percentiles = [50, 90, 95, 99]
        colors = ['red', 'orange', 'purple', 'darkred']
        
        for p, color in zip(percentiles, colors):
            percentile_value = np.percentile(sorted_durations, p)
            ax.axvline(percentile_value, color=color, linestyle='--', alpha=0.7, 
                    label=f'P{p}: {percentile_value:.2f}s')
            ax.text(percentile_value, 0.1 + p/200, f'P{p}\n{percentile_value:.1f}s', 
                rotation=90, ha='right', va='bottom', fontsize=9)
        
        # Add statistics text box
        stats_text = f"""Statistics:
        Total Requests: {n}
        Min: {np.min(durations):.2f}s
        Max: {np.max(durations):.2f}s
        Mean: {np.mean(durations):.2f}s
        Median: {np.median(durations):.2f}s
        Std: {np.std(durations):.2f}s"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
        
        # Set y-axis to show percentages
        ax.set_ylim(0, 1)
        ax.set_ylabel('Cumulative Probability')
        
        # Add secondary y-axis with percentage labels
        ax2 = ax.twinx()
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('Percentile (%)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"CDF plot saved to {save_path}")
        
        return fig



if __name__ == "__main__":
    main()