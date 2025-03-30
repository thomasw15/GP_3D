#!/usr/bin/env python3
"""
C17 Gate Sizes vs Volume Constraint
===================================

Plots how individual gate sizes in the c17 circuit change as 
volume constraint increases. Analyzes tradeoffs between volume
and gate sizing decisions.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Add script directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import the necessary functions
from ISCAS_gate_sizing import (
    extract_circuit_topology,
    load_asap7_parameters,
    size_gates_with_gp,
    validate_circuit_topology,
    validate_tech_parameters
)

def run_c17_gate_sizes_experiment():
    """
    Run experiment analyzing gate size changes with varying volume constraints.
    
    Returns:
        DataFrame with experiment results
    """
    print("\n===== C17 GATE SIZES VS VOLUME CONSTRAINT EXPERIMENT =====")
    
    # Set up paths
    circuit_file = os.path.join(script_dir, "ISCAS85", "c17.v")
    
    # Find parameters file
    params_paths = [
        os.path.join(script_dir, "asap7_extracted_params.json")
    ]
    
    params_file = None
    for path in params_paths:
        if os.path.exists(path):
            params_file = path
            break
    
    if params_file is None:
        print("Error: Could not find ASAP7 parameters file.")
        sys.exit(1)
    
    print(f"Using circuit file: {circuit_file}")
    print(f"Using parameters file: {params_file}")
    
    # Extract circuit topology
    print("\nExtracting circuit topology...")
    circuit_topology = extract_circuit_topology(circuit_file)
    
    # Validate circuit topology
    if not validate_circuit_topology(circuit_topology):
        print("Circuit topology validation failed.")
        sys.exit(1)
    
    # Load technology parameters
    print("\nLoading technology parameters...")
    tech_parameters = load_asap7_parameters(params_file, circuit_topology)
    
    # Validate technology parameters
    if not validate_tech_parameters(tech_parameters, circuit_topology):
        print("Technology parameter validation failed.")
        sys.exit(1)
    
    # First, find the minimum volume (optimize for volume only)
    print("\nFinding minimum volume...")
    min_vol_options = {
        'volume_weight': 1.0,
        'min_size': 1.0,
        'max_size': 10.0,
        'verbose': False
    }
    
    min_vol_results = size_gates_with_gp(circuit_topology, tech_parameters, min_vol_options)
    min_volume = min_vol_results['volume']
    print(f"Minimum achievable volume: {min_volume:.4f} μm³")
    
    # Next, find delay with no volume constraint
    print("\nFinding delay with no volume constraint...")
    unconstrained_options = {
        'volume_weight': 0.0,
        'min_size': 1.0,
        'max_size': 10.0,
        'verbose': False
    }
    
    unconstrained_results = size_gates_with_gp(circuit_topology, tech_parameters, unconstrained_options)
    max_volume = unconstrained_results['volume']
    print(f"Maximum volume (with no constraint): {max_volume:.4f} μm³")
    
    # Find all gate names
    gate_names = sorted(list(tech_parameters['gate_params'].keys()))
    print(f"Found {len(gate_names)} gates: {', '.join(gate_names)}")
    
    # Generate range of volume constraints
    # Use a reasonable number of points between min_volume and max_volume
    num_points = 15
    volume_constraints = np.linspace(min_volume, max_volume * 1.1, num_points)
    
    # Store gate sizes for each volume constraint
    gate_sizes_by_constraint = []
    actual_volumes = []
    
    print(f"\nRunning optimization with {num_points} different volume constraints...")
    
    # Run optimization for each volume constraint
    for i, volume_constraint in enumerate(volume_constraints):
        print(f"  Progress: {i+1}/{num_points} - Volume constraint: {volume_constraint:.4f} μm³")
        
        options = {
            'volume_weight': 0.0,  # Optimize delay only
            'min_size': 1.0,
            'max_size': 10.0,
            'volume_constraint': volume_constraint,
            'verbose': False
        }
        
        results = size_gates_with_gp(circuit_topology, tech_parameters, options)
        
        if results['success']:
            gate_sizes_by_constraint.append(results['gate_sizes'])
            actual_volumes.append(results['volume'])
        else:
            print(f"  Optimization failed for volume constraint {volume_constraint:.4f}")
            # Use previous results or fallback values
            if gate_sizes_by_constraint:
                gate_sizes_by_constraint.append(gate_sizes_by_constraint[-1])
            else:
                gate_sizes_by_constraint.append({gate: 1.0 for gate in gate_names})
            
            if actual_volumes:
                actual_volumes.append(actual_volumes[-1])
            else:
                actual_volumes.append(volume_constraint)
    
    # Create output directory
    output_dir = os.path.join(script_dir, "results", "c17_gate_sizes")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data for CSV
    data = {
        'VolumeConstraint': volume_constraints,
        'ActualVolume': actual_volumes
    }
    
    # Add gate sizes
    for gate in gate_names:
        data[gate] = [sizes.get(gate, 1.0) for sizes in gate_sizes_by_constraint]
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    csv_file = os.path.join(output_dir, f"c17_gate_sizes_{timestamp}.csv")
    df.to_csv(csv_file, index=False)
    print(f"\nData saved to {csv_file}")
    
    # Create visualization with styling like plot_from_csv.py
    plt.figure(figsize=(20, 10))
    
    # Set larger default font sizes
    plt.rcParams.update({'font.size': 20})
    
    # Plot gate sizes with dotted lines and markers
    for gate in gate_names:
        gate_sizes = [sizes.get(gate, 1.0) for sizes in gate_sizes_by_constraint]
        plt.plot(volume_constraints, gate_sizes, '--o', linewidth=2, markersize=8, label=gate)
    
    # Add labels with larger font size (all lowercase)
    plt.xlabel('volume constraint', fontsize=24)
    plt.ylabel('gate size', fontsize=24)
    
    # Add legend with larger font
    plt.legend(fontsize=20, loc='best')
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Set font size for tick labels
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_pdf_file = os.path.join(output_dir, f"c17_gate_sizes_{timestamp}.pdf")
    plot_png_file = os.path.join(output_dir, f"c17_gate_sizes_{timestamp}.png")
    
    plt.savefig(plot_pdf_file)
    plt.savefig(plot_png_file)
    
    print(f"Plot saved to {plot_pdf_file}")
    
    return df

if __name__ == "__main__":
    run_c17_gate_sizes_experiment()