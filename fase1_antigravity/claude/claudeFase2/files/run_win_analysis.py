import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# Add current dir to path
sys.path.insert(0, os.getcwd())

from ibovespa_bcp_reversal_detector import analyze_ibovespa_reversal

def run_analysis():
    # Construct path to WIN_5min.csv
    # Current dir: .../claude/claudeFase2/files
    # Target: .../roboMamhedgeM1.0/WIN_5min.csv
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
    # No, that's too many levels.
    # getcwd() ends in 'files'
    # Parent 1: claudeFase2
    # Parent 2: claude
    # Parent 3: roboMamhedgeM1.0
    project_root = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
    csv_path = os.path.join(project_root, 'WIN_5min.csv')
    
    print(f"Project root: {project_root}")
    print(f"Looking for CSV at: {csv_path}")
    
    if not os.path.exists(csv_path):
        print("Error: CSV file not found!")
        return

    print("Running BCP analysis...")
    print("Parameters: p0=0.01 (structural), max_block=300")
    
    try:
        results = analyze_ibovespa_reversal(
            csv_path, 
            date_column='Datetime', 
            price_column='close', 
            p0=0.01, 
            max_block=150
        )
        
        if results:
            output_file = 'win_bcp_analysis_today.png'
            results['figure'].savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Graph saved to {output_file}")
            
            # Save probability to file
            last_prob = results['detector'].posterior_probability[-1]
            with open('prob.txt', 'w') as f:
                f.write(f"{last_prob:.6f}")
            print(f"Probability saved to prob.txt: {last_prob:.6f}")
            
            print("Analysis successful.")
            
    except Exception as e:
        print(f"Detailed error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_analysis()
