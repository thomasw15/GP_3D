#!/usr/bin/env python3
"""
Gate Sizing Path Delay Analysis
================================

Performs gate sizing optimization on ISCAS circuits and analyzes path delays
before and after optimization with a fixed volume constraint.

Generates visualization comparing path delays pre/post optimization.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import argparse
from datetime import datetime
import subprocess

# Add script directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import functions from ISCAS_gate_sizing.py
from ISCAS_gate_sizing import (
    extract_circuit_topology,
    load_asap7_parameters,
    size_gates_with_gp,
    validate_circuit_topology,
    validate_tech_parameters
)

def get_path_delays(circuit_topology, tech_parameters, gate_sizes):
    """
    Calculate delay for each path using the given gate sizes.
    
    Args:
        circuit_topology: Dictionary containing circuit topology
        tech_parameters: Dictionary containing technology parameters
        gate_sizes: Dictionary mapping gate names to their sizes
        
    Returns:
        List of (path, delay, gates) tuples
    """
    # Get paths and gate mappings
    paths = circuit_topology['all_paths']
    gate_drivers = circuit_topology['gate_drivers']
    
    # Get gate parameters
    gate_params = tech_parameters['gate_params']
    
    path_delays = []
    
    # Process each path
    for path_idx, path in enumerate(paths):
        path_delay = 0
        path_gates = []
        
        # Extract gates in this path and calculate delay
        for i in range(len(path) - 1):
            # Current node
            node = path[i]
            next_node = path[i + 1]
            
            # Skip if not a gate output
            if node not in gate_drivers:
                continue
                
            # Get gate information
            gate_name = gate_drivers[node]
            path_gates.append(gate_name)
            
            # Skip if gate has no parameters
            if gate_name not in gate_params:
                continue
                
            # Get gate size and parameters
            gate_size = gate_sizes.get(gate_name, 1.0)  # Default to 1.0 if not sized
            gate_R = gate_params[gate_name]['R']
            gate_C_int = gate_params[gate_name]['C_int']
            
            # Calculate intrinsic delay
            intrinsic_delay = gate_R / gate_size
            
            # Load capacitance - internal capacitance
            load_cap = gate_C_int * gate_size
            
            # Add load from next node if it's a gate input
            if next_node in circuit_topology['gate_drivers']:
                next_gate = circuit_topology['gate_drivers'][next_node]
                if next_gate in gate_params:
                    load_cap += gate_params[next_gate]['C_in'] * gate_sizes.get(next_gate, 1.0)
            
            # Gate delay
            gate_delay = intrinsic_delay * load_cap
            path_delay += gate_delay
        
        if path_gates:  # Only include paths with gates
            path_delays.append((path, path_delay, path_gates))
    
    return path_delays

def run_optimization(circuit_file, params_file, volume_constraint_factor=1.5):
    """
    Perform gate sizing optimization with a fixed volume constraint.
    
    Args:
        circuit_file: Path to the circuit Verilog file
        params_file: Path to the technology parameters file
        volume_constraint_factor: Factor to multiply minimum volume for constraint
        
    Returns:
        tuple: (circuit_topology, tech_parameters, pre_opt_delays, post_opt_delays, results)
    """
    print(f"Running optimization on {circuit_file}")
    
    # Extract circuit topology
    circuit_topology = extract_circuit_topology(circuit_file)
    
    # Validate circuit topology
    if not validate_circuit_topology(circuit_topology):
        print("Circuit topology validation failed.")
        sys.exit(1)
    
    # Load technology parameters
    tech_parameters = load_asap7_parameters(params_file, circuit_topology)
    
    # Validate technology parameters
    if not validate_tech_parameters(tech_parameters, circuit_topology):
        print("Technology parameter validation failed.")
        sys.exit(1)
    
    # Calculate pre-optimization delays (all gates at min size)
    pre_opt_sizes = {gate: 1.0 for gate in tech_parameters['gate_params']}
    pre_opt_path_delays = get_path_delays(circuit_topology, tech_parameters, pre_opt_sizes)
    
    # Sort by delay for better visualization
    pre_opt_path_delays.sort(key=lambda x: x[1], reverse=True)
    
    # First find the minimum volume (optimize for volume only)
    min_vol_options = {
        'volume_weight': 1.0,
        'min_size': 1.0,
        'max_size': 10.0,
        'verbose': True
    }
    
    min_vol_results = size_gates_with_gp(circuit_topology, tech_parameters, min_vol_options)
    min_volume = min_vol_results['volume']
    
    # Then run optimization with a volume constraint
    volume_constraint = min_volume * volume_constraint_factor
    
    optimization_options = {
        'volume_weight': 0.0,  # Optimize delay only
        'min_size': 1.0,
        'max_size': 10.0,
        'volume_constraint': volume_constraint,
        'verbose': True
    }
    
    # Run optimization
    results = size_gates_with_gp(circuit_topology, tech_parameters, optimization_options)
    
    if not results['success']:
        print("Optimization failed.")
        sys.exit(1)
    
    # Calculate post-optimization delays
    post_opt_path_delays = get_path_delays(circuit_topology, tech_parameters, results['gate_sizes'])
    
    # Sort in the same order as pre-optimization delays for consistency
    paths_order = {tuple(path[0]): i for i, path in enumerate(pre_opt_path_delays)}
    post_opt_path_delays.sort(key=lambda x: paths_order.get(tuple(x[0]), 999))
    
    print(f"Optimization results:")
    print(f"  Minimum volume: {min_volume:.4f} μm³")
    print(f"  Volume constraint: {volume_constraint:.4f} μm³")
    print(f"  Optimized delay: {results['delay']:.4f} ps")
    print(f"  Optimized volume: {results['volume']:.4f} μm³")
    
    return circuit_topology, tech_parameters, pre_opt_path_delays, post_opt_path_delays, results

def create_visualization(pre_opt_delays, post_opt_delays, circuit_name):
    """
    Create bar chart visualization comparing pre/post optimization path delays.
    
    Args:
        pre_opt_delays: List of (path, delay, gates) tuples before optimization
        post_opt_delays: List of (path, delay, gates) tuples after optimization
        circuit_name: Name of the circuit for the output file
        
    Returns:
        tuple: (csv_file_path, plot_file_path)
    """
    # Save all data to CSV
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare all data for CSV export
    path_indices = range(1, len(pre_opt_delays) + 1)  # Start from 1
    pre_delays = [d[1] for d in pre_opt_delays]
    post_delays = [d[1] for d in post_opt_delays]
    
    # Create DataFrame for all paths
    all_df = pd.DataFrame({
        'PathIndex': path_indices,
        'PreDelay': pre_delays,
        'PostDelay': post_delays
    })
    
    # Save all data to CSV
    all_csv_file = os.path.join(output_dir, f'{circuit_name}_all_path_delays.csv')
    all_df.to_csv(all_csv_file, index=False)
    print(f"All path delay data saved to {all_csv_file}")
    
    # Get indices of the 20 paths with highest delays before optimization
    top_pre_indices = np.argsort(pre_delays)[-20:][::-1]  # Get indices of top 20 highest delays, reversed
    
    # Get indices of the 20 paths with highest delays after optimization
    top_post_indices = np.argsort(post_delays)[-20:][::-1]  # Get indices of top 20 highest delays, reversed
    
    # Combine and sort for better presentation
    all_top_indices = sorted(list(set(top_pre_indices) | set(top_post_indices)))[:20]  # Ensure we have at most 20
    
    # Extract top paths data
    top_indices = [i+1 for i in all_top_indices]  # Path indices start from 1
    top_pre_delays = [pre_delays[i] for i in all_top_indices]
    top_post_delays = [post_delays[i] for i in all_top_indices]
    
    # Create DataFrame for top paths
    top_df = pd.DataFrame({
        'PathIndex': top_indices,
        'PreDelay': top_pre_delays,
        'PostDelay': top_post_delays
    })
    
    # Save top paths data to separate CSV
    top_csv_file = os.path.join(output_dir, f'{circuit_name}_top_path_delays.csv')
    top_df.to_csv(top_csv_file, index=False)
    print(f"Top path delay data saved to {top_csv_file}")
    
    # Create the plot
    plt.figure(figsize=(20, 10))
    
    # Set much larger default font sizes
    plt.rcParams.update({'font.size': 20})
    
    # Plot bars - placed on same vertical line
    bar_positions = np.arange(len(top_df))
    
    # Plot the bars with semi-transparency to see both values
    plt.bar(bar_positions, top_df['PreDelay'], 
            color='red', alpha=0.6, label='before optimization')
    plt.bar(bar_positions, top_df['PostDelay'], 
            color='blue', alpha=0.6, label='after optimization')
    
    # Add labels with larger font size (all lowercase)
    plt.xlabel('critical paths', fontsize=24)
    plt.ylabel('delay', fontsize=24)
    
    # Add legend with larger font
    plt.legend(fontsize=24)
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Use path indices as x-tick labels
    plt.xticks(bar_positions, top_df['PathIndex'], fontsize=20)
    plt.yticks(fontsize=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(output_dir, f'{circuit_name}_top_path_delays.pdf')
    plt.savefig(plot_file)
    print(f"Path delay visualization saved to {plot_file}")
    
    # Also save as PNG for easy viewing
    plt.savefig(os.path.join(output_dir, f'{circuit_name}_top_path_delays.png'))
    
    return top_csv_file, plot_file

def print_path_details(pre_opt_delays, post_opt_delays, gate_sizes):
    """
    Print detailed information about path delays and gate sizes.
    
    Args:
        pre_opt_delays: List of (path, delay, gates) tuples before optimization
        post_opt_delays: List of (path, delay, gates) tuples after optimization
        gate_sizes: Dictionary mapping gate names to their optimized sizes
    """
    print("\nPath delay details:")
    print("-" * 80)
    print(f"{'Path':^10} | {'Before (ps)':^15} | {'After (ps)':^15} | {'Improvement':^15} | {'Gates':^20}")
    print("-" * 80)
    
    for i, ((path1, pre_delay, pre_gates), (path2, post_delay, post_gates)) in enumerate(zip(pre_opt_delays, post_opt_delays)):
        # Calculate improvement percentage
        improvement = (pre_delay - post_delay) / pre_delay * 100 if pre_delay > 0 else 0
        
        # Format gates with sizes
        gate_str = ", ".join([f"{g}({gate_sizes.get(g, 1.0):.2f}x)" for g in post_gates[:3]])
        if len(post_gates) > 3:
            gate_str += f", ... ({len(post_gates)} total)"
        
        print(f"{i+1:^10} | {pre_delay:^15.4f} | {post_delay:^15.4f} | {improvement:^15.2f}% | {gate_str}")
    
    print("-" * 80)

def main():
    """
    Parse command-line arguments and run path delay analysis.
    """
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Gate Sizing Path Delay Analysis")
    parser.add_argument("--volume-factor", type=float, default=1.5, 
                      help="Factor to multiply minimum volume for constraint (default: 1.5)")
    parser.add_argument("--circuit", type=str, default="c499",
                      help="Circuit to analyze (default: c499)")
    args = parser.parse_args()
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    circuit_file = os.path.join(script_dir, "ISCAS85", f"{args.circuit}.v")
    
    # Find parameters file
    params_paths = [
        os.path.join(script_dir, "asap7_extracted_params.json")
        # Removed alternative paths that no longer exist
    ]
    
    params_file = None
    for path in params_paths:
        if os.path.exists(path):
            params_file = path
            break
    
    if params_file is None:
        print("Error: Could not find ASAP7 parameters file.")
        sys.exit(1)
    
    # Run optimization
    circuit_topology, tech_parameters, pre_opt_delays, post_opt_delays, results = run_optimization(
        circuit_file, params_file, args.volume_factor
    )
    
    # Print detailed information about each path
    print_path_details(pre_opt_delays, post_opt_delays, results['gate_sizes'])
    
    # Create visualization
    circuit_name = os.path.splitext(os.path.basename(circuit_file))[0]
    csv_file, plot_file = create_visualization(pre_opt_delays, post_opt_delays, circuit_name)
    
    print(f"\nAnalysis complete!")
    print(f"Visualization saved to {plot_file}")
    print(f"Data saved to {csv_file}")

if __name__ == "__main__":
    main() 