#!/usr/bin/env python3
"""
ISCAS Gate Sizing using Geometric Programming
==============================================

This script performs gate sizing optimization on ISCAS benchmark circuits
using geometric programming. It uses the ASAP7 technology parameters to
model gate delay and area.

Author: Claude
Date: March 27, 2024
"""

import os
import sys
import json
import networkx as nx
import numpy as np
import time
import cvxpy as cp
from collections import defaultdict, deque
import argparse
import traceback
import psutil

# Make sure verilog_parser.py is in the path
script_dir = os.path.dirname(os.path.abspath(__file__))
if 'verilog_parser' not in sys.modules:
    sys.path.append(script_dir)

# Import the VerilogParser
from verilog_parser import VerilogParser


def extract_circuit_topology(verilog_file):
    """
    Extract the circuit topology from a Verilog file for optimization.
    
    This function parses the given Verilog file and extracts all the information
    needed for gate sizing optimization: gates, signals, connectivity, etc.
    
    Args:
        verilog_file (str): Path to the Verilog file
        
    Returns:
        dict: A dictionary containing circuit information with the following keys:
            - module_name: Name of the circuit module
            - inputs: List of primary input signals
            - outputs: List of primary output signals
            - wires: List of internal wires
            - gates: List of (gate_type, gate_name, output, inputs) tuples
            - circuit_graph: NetworkX directed graph representation of the circuit
            - gate_drivers: Dictionary mapping signals to the gates that drive them
            - gate_fanouts: Dictionary mapping gates to their fanout signals
            - gate_fanins: Dictionary mapping gates to their fanin signals
            - gate_types: Dictionary mapping gate names to their types
            - signal_load: Dictionary mapping signals to their load gates
            - all_paths: List of all paths from primary inputs to primary outputs
    """
    print(f"Extracting circuit topology from {verilog_file}...")
    
    # Create and parse the Verilog file
    parser = VerilogParser(verilog_file)
    parser.parse()
    
    # Initialize the topology dictionary
    topology = {
        'module_name': parser.module_name,
        'inputs': parser.inputs,
        'outputs': parser.outputs,
        'wires': parser.wires,
        'gates': parser.gates,
        'circuit_graph': parser.circuit_graph,
        'gate_drivers': {},      # Which gate drives each signal
        'gate_fanouts': {},      # Fanout signals for each gate
        'gate_fanins': {},       # Fanin signals for each gate 
        'gate_types': {},        # Type of each gate
        'signal_load': defaultdict(list)  # Gates that load each signal
    }
    
    # Extract gate types
    for gate_type, gate_name, output, inputs in parser.gates:
        topology['gate_types'][gate_name] = gate_type
    
    # For each gate, determine which signals it drives and which signals drive it
    for gate_type, gate_name, output, inputs in parser.gates:
        # Skip buffers (they're not real gates for optimization)
        if gate_type == 'buf':
            continue
            
        # This gate drives the output signal
        topology['gate_drivers'][output] = gate_name
        
        # The output signal is a fanout of this gate
        if gate_name not in topology['gate_fanouts']:
            topology['gate_fanouts'][gate_name] = []
        topology['gate_fanouts'][gate_name].append(output)
        
        # The input signals are fanins of this gate
        if gate_name not in topology['gate_fanins']:
            topology['gate_fanins'][gate_name] = []
        topology['gate_fanins'][gate_name].extend(inputs)
        
        # This gate loads each of its input signals
        for input_signal in inputs:
            topology['signal_load'][input_signal].append(gate_name)
    
    # Create gate-to-gate connectivity maps
    topology['gate_to_gate_fanout'] = defaultdict(list)
    topology['gate_to_gate_fanin'] = defaultdict(list)
    
    for gate_type, gate_name, output, inputs in parser.gates:
        if gate_type == 'buf':
            continue
            
        # Find gates that are driven by this gate's output
        for load_gate in topology['signal_load'].get(output, []):
            topology['gate_to_gate_fanout'][gate_name].append(load_gate)
        
        # Find gates that drive this gate's inputs
        for input_signal in inputs:
            if input_signal in topology['gate_drivers']:
                driver_gate = topology['gate_drivers'][input_signal]
                topology['gate_to_gate_fanin'][gate_name].append(driver_gate)
    
    # Find all paths from primary inputs to primary outputs
    # This is essential for the gate sizing optimization as we need to consider all paths
    all_paths = find_all_paths(parser.circuit_graph, parser.inputs, parser.outputs)
    topology['all_paths'] = all_paths
    
    # Print a summary of the extracted topology
    print(f"Extracted topology for module: {topology['module_name']}")
    print(f"  Primary inputs: {len(topology['inputs'])}")
    print(f"  Primary outputs: {len(topology['outputs'])}")
    print(f"  Internal wires: {len(topology['wires'])}")
    print(f"  Gates: {len([g for g in topology['gates'] if g[0] != 'buf'])}")
    print(f"  Buffers: {len([g for g in topology['gates'] if g[0] == 'buf'])}")
    
    # Print gate type distribution
    gate_type_counts = defaultdict(int)
    for gate_type, _, _, _ in topology['gates']:
        if gate_type != 'buf':
            gate_type_counts[gate_type] += 1
    
    print("  Gate type distribution:")
    for gate_type, count in sorted(gate_type_counts.items()):
        print(f"    {gate_type}: {count}")
    
    # Print the number of paths
    print(f"\nFound {len(topology['all_paths'])} paths from primary inputs to primary outputs")
    
    # Print a few sample paths for verification if available
    if topology['all_paths']:
        n_samples = min(3, len(topology['all_paths']))
        print(f"Sample paths through the circuit:")
        for i, path in enumerate(topology['all_paths'][:n_samples]):
            print(f"  Path {i+1}: {' -> '.join(path)}")
    
    return topology


def find_all_paths(graph, inputs, outputs, max_paths=None):
    """
    Find all paths from primary inputs to primary outputs in the circuit graph.
    
    This function finds ALL paths without any limits, which may be
    computationally intensive for large circuits.
    
    Args:
        graph: NetworkX directed graph of the circuit
        inputs: List of primary input signals
        outputs: List of primary output signals
        max_paths: No longer used, kept for backward compatibility
        
    Returns:
        list: List of paths, where each path is a list of node names
    """
    import time
    start_time = time.time()
    all_paths = []
    paths_count = 0
    
    print(f"Finding all paths from primary inputs to primary outputs (no limit)...")
    
    # For each input-output pair, find all paths
    input_count = len(inputs)
    output_count = len(outputs)
    total_pairs = input_count * output_count
    pair_count = 0
    
    print(f"Searching paths between {input_count} inputs and {output_count} outputs ({total_pairs} pairs)")
    
    # Monitor memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    try:
        for input_node in inputs:
            for output_node in outputs:
                pair_count += 1
                pair_start_time = time.time()
                print(f"Processing input-output pair {pair_count}/{total_pairs}: {input_node} -> {output_node}...")
                
                # Use a modified DFS to find all paths - no limits here
                paths = find_paths_dfs(graph, input_node, output_node)
                
                pair_end_time = time.time()
                print(f"  Found {len(paths)} paths in {pair_end_time - pair_start_time:.2f} seconds")
                
                all_paths.extend(paths)
                paths_count += len(paths)
                
                # Check memory usage
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                # Print progress periodically
                if paths_count > 0 and (paths_count % 100000 == 0 or memory_increase > 1000):  # Check every 100k paths or if memory usage increased by >1GB
                    elapsed = time.time() - start_time
                    print(f"  Current total: {paths_count} paths found in {elapsed:.2f} seconds")
                    print(f"  Current memory usage: {current_memory:.2f} MB (increase: {memory_increase:.2f} MB)")
                    
                    # Warning if memory usage is getting high
                    if memory_increase > 10000:  # >10GB increase
                        print("WARNING: Memory usage is very high. Consider implementing a subset of critical paths if needed.")
    except MemoryError:
        print("ERROR: Out of memory while finding paths.")
        print(f"Found {paths_count} paths before running out of memory.")
        print("Will continue with the paths found so far, but results may not be optimal.")
        # Continue with the paths we've found so far
    except Exception as e:
        print(f"Error during path finding: {str(e)}")
        traceback.print_exc()
        if paths_count == 0:
            raise  # Re-raise if we haven't found any paths
        print("Will continue with the paths found so far.")
    
    print(f"Found {paths_count} paths total")
        
    end_time = time.time()
    print(f"Total path finding time: {end_time - start_time:.2f} seconds")
    
    # Final memory usage report
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Final memory usage: {final_memory:.2f} MB (increase: {final_memory - initial_memory:.2f} MB)")
    
    return all_paths


def find_paths_dfs(graph, start, end, max_paths=None):
    """
    Use depth-first search to find all paths from start to end in a graph.
    
    Args:
        graph: NetworkX directed graph
        start: Start node
        end: End node
        max_paths: No longer used, kept for backward compatibility
        
    Returns:
        list: List of paths from start to end
    """
    paths = []
    
    # Use a stack for DFS
    stack = [(start, [start])]
    
    while stack:
        (node, path) = stack.pop()
        
        # Get nodes that aren't already in the path to avoid cycles
        next_nodes = [n for n in graph.successors(node) if n not in path]
        
        for next_node in next_nodes:
            if next_node == end:
                # Found a path to the end
                paths.append(path + [next_node])
            else:
                # Continue DFS
                stack.append((next_node, path + [next_node]))
    
    return paths


def load_asap7_parameters(params_file, circuit_topology):
    """
    Load ASAP7 technology parameters from a JSON file for gate sizing.
    
    This function loads only the parameters needed for the gate sizing model described
    in Section 4.3 of the referenced paper, mapping them to gates based on their types.
    
    Args:
        params_file (str): Path to the JSON file containing ASAP7 parameters
        circuit_topology (dict): Circuit topology dictionary from extract_circuit_topology()
        
    Returns:
        dict: Dictionary containing the parameters for the transistor sizing GP:
            - gate_params: Dictionary mapping gate names to their parameters:
                - vol: Volume/area at unit scaling (μm²)
                - R: Resistance at unit scaling (kΩ)
                - C_in: Input capacitance at unit scaling (fF)
                - C_int: Internal capacitance at unit scaling (fF)
                - I_leak: Leakage current at unit scaling (nA)
            - PO_caps: Dictionary of input capacitances for primary outputs
            - freqs: Dictionary of activity frequencies for each gate
            - vdd: Supply voltage (V)
    """
    print(f"Loading ASAP7 parameters from {params_file}...")
    
    # Load the parameters from the JSON file
    try:
        with open(params_file, 'r') as f:
            asap7_data = json.load(f)
    except Exception as e:
        # If parameters cannot be loaded, raise an error instead of using defaults
        raise RuntimeError(f"Error loading parameters file: {e}. Please ensure the file exists and has the correct format.")
    
    # Extract supply voltage
    vdd = asap7_data.get('vdd', 0.7)  # Supply voltage (V)
    
    # Extract parameter maps from the JSON data - map to our expected parameters
    # For existing asap7_extracted_params.json format:
    # - gate_area maps to our vol/area
    # - gate_delay[x][1] (second element) maps to our R (output resistance)
    # - input_cap maps to our C_in
    # - We'll derive C_int as a fraction of C_in
    # - We'll derive I_leak from leakage data (if present)
    
    gate_areas = asap7_data.get('gate_area', {})
    gate_delays = asap7_data.get('gate_delay', {})
    input_caps = asap7_data.get('input_cap', {})
    leakage_data = asap7_data.get('leakage', {})
    
    # Verify that we have the required data
    if not gate_areas or not gate_delays or not input_caps:
        raise ValueError("The parameter file is missing critical data (gate_area, gate_delay, or input_cap). Please check the file.")
    
    # Print debug information about parameter data
    print("\nParameter data diagnostics:")
    print(f"- Number of gate types with area data: {len(gate_areas)}")
    print(f"- Number of gate types with delay data: {len(gate_delays)}")
    print(f"- Number of gate types with input cap data: {len(input_caps)}")
    
    # Check for extremely large or small values in the parameters
    max_area = max(gate_areas.values()) if gate_areas else 0
    min_area = min(gate_areas.values()) if gate_areas else 0
    print(f"- Gate area range: [{min_area}, {max_area}]")
    
    max_delay = max([d[1] if isinstance(d, list) and len(d) > 1 else 0 for d in gate_delays.values()]) if gate_delays else 0
    min_delay = min([d[1] if isinstance(d, list) and len(d) > 1 else float('inf') for d in gate_delays.values() if isinstance(d, list) and len(d) > 1]) if gate_delays else 0
    print(f"- Gate delay (resistance) range: [{min_delay}, {max_delay}]")
    
    max_cap = max(input_caps.values()) if input_caps else 0
    min_cap = min(input_caps.values()) if input_caps else 0
    print(f"- Input capacitance range: [{min_cap}, {max_cap}]")
    
    # Ensure all gate areas are non-zero for GP compatibility
    for gate_type in gate_areas:
        if gate_areas[gate_type] <= 0:
            # Use a small epsilon value for zero area gates (e.g., 'inv' with 0 area)
            gate_areas[gate_type] = 0.001  # Small default area (0.001 μm²)
    
    # Map the parameters to our format
    gate_resistances = {}
    for gate_type, delay_data in gate_delays.items():
        if isinstance(delay_data, list) and len(delay_data) > 1:
            gate_resistances[gate_type] = delay_data[1]  # Second element is output resistance
    
    # Internal capacitance (estimated as percentage of input capacitance)
    internal_caps = {}
    for gate_type, cap in input_caps.items():
        internal_caps[gate_type] = cap * 0.4  # Internal cap is roughly 40% of input cap
    
    # Leakage current (in nA)
    leakage_currents = {}
    for gate_type, leakage in leakage_data.items():
        # Convert leakage power to current: I = P/V
        leakage_currents[gate_type] = leakage / vdd if leakage else 0.01
    
    # Set default activity frequencies based on gate type
    activity_freqs = {}
    for gate_type in gate_areas.keys():
        if 'inv' in gate_type or 'buf' in gate_type:
            activity_freqs[gate_type] = 200.0  # MHz
        else:
            activity_freqs[gate_type] = 100.0  # MHz
    
    # Initialize parameter dictionaries
    gate_params = {}    # Parameters for gates in combinational blocks
    PO_caps = {}        # Input capacitances for primary outputs
    freqs = {}          # Activity frequencies for each gate
    
    # Identify primary inputs (PI), primary outputs (PO), and combinational blocks (CB)
    primary_inputs = set(circuit_topology['inputs'])
    primary_outputs = set(circuit_topology['outputs'])
    
    # Find gate drivers and signal loads
    gate_drivers = circuit_topology['gate_drivers']
    signal_load = circuit_topology['signal_load']
    
    # Set primary output capacitances (default 1.0 fF)
    for signal in primary_outputs:
        PO_caps[signal] = 1.0  # Default primary output capacitance
    
    # Extract parameters for each gate in the circuit
    for gate_type, gate_name, output, inputs in circuit_topology['gates']:
        # Skip buffers
        if gate_type == 'buf':
            continue
            
        # Normalize gate type for parameter lookup
        norm_gate_type = normalize_gate_type(gate_type)
        
        # Set activity frequency for this gate
        freqs[gate_name] = activity_freqs.get(norm_gate_type, 100.0)
        
        # Get parameters for this gate type (or raise error if not found)
        if norm_gate_type not in gate_areas:
            print(f"Warning: No parameters found for gate type '{norm_gate_type}' ({gate_type}). This gate will be skipped.")
            continue
            
        vol = gate_areas.get(norm_gate_type)
        r = gate_resistances.get(norm_gate_type)
        c_in = input_caps.get(norm_gate_type)
        
        if vol is None or r is None or c_in is None:
            print(f"Warning: Incomplete parameters for gate type '{norm_gate_type}'. This gate will be skipped.")
            continue
        
        # Ensure volume is non-zero (required for DGP compatibility)
        if vol <= 0:
            vol = 0.001  # Small default area (0.001 μm²)
            
        c_int = internal_caps.get(norm_gate_type)
        i_leak = leakage_currents.get(norm_gate_type, 0.01)  # Default to 0.01 nA if not available
        
        # Store parameters for this gate
        gate_params[gate_name] = {
            'vol': vol,           # Area/volume (μm²)
            'R': r,               # Resistance (kΩ)
            'C_in': c_in,         # Input capacitance (fF)
            'C_int': c_int,       # Internal capacitance (fF)
            'I_leak': i_leak      # Leakage current (nA)
        }
    
    # Print summary of loaded parameters
    print(f"Loaded parameters for {len(gate_params)} gates")
    print(f"Supply voltage: {vdd} V")
    print(f"Identified {len(primary_inputs)} primary inputs and {len(primary_outputs)} primary outputs")
    
    # Print some sample parameters for verification
    if gate_params:
        print("\nSample gate parameters:")
        sample_gates = list(gate_params.keys())[:3]
        for gate_name in sample_gates:
            gate_type = circuit_topology['gate_types'][gate_name]
            params = gate_params[gate_name]
            print(f"  {gate_name} ({gate_type}):")
            print(f"    Volume/Area: {params['vol']:.4f} μm²")
            print(f"    Resistance: {params['R']:.2f} kΩ")
            print(f"    Input capacitance: {params['C_in']:.2f} fF")
            print(f"    Internal capacitance: {params['C_int']:.2f} fF")
            print(f"    Leakage current: {params['I_leak']:.4f} nA")
            print(f"    Activity frequency: {freqs[gate_name]:.1f} MHz")
    
    return {
        'gate_params': gate_params,
        'PO_caps': PO_caps,
        'freqs': freqs,
        'vdd': vdd
    }


def normalize_gate_type(gate_type):
    """
    Normalize gate type names for consistent parameter lookup.
    
    This function standardizes gate type names across different formats to ensure
    we can find the corresponding parameters in the ASAP7 parameter file. 
    
    For example:
    - "NAND2" or "nand" would be normalized to "nand2"
    - "OR3" would be normalized to "or3"
    - "INV" or "not" would be normalized to "inv"
    
    This is necessary because the Verilog parser may extract gate types in various
    formats, while the parameter file uses specific standardized names.
    
    Args:
        gate_type (str): Original gate type name
        
    Returns:
        str: Normalized gate type name that matches the keys in the parameter file
    """
    gate_type = gate_type.lower()
    
    # Remove any trailing digits and convert to lowercase
    base_type = ''.join([c for c in gate_type if not c.isdigit()])
    
    # Extract the number of inputs (if any)
    input_count = ''.join([c for c in gate_type if c.isdigit()])
    
    # Handle special cases
    if base_type == 'nand':
        return f"nand{input_count}" if input_count else "nand2"
    elif base_type == 'nor':
        return f"nor{input_count}" if input_count else "nor2"
    elif base_type == 'and':
        return f"and{input_count}" if input_count else "and2"
    elif base_type == 'or':
        return f"or{input_count}" if input_count else "or2"
    elif base_type == 'xor':
        return "xor"
    elif base_type == 'xnor':
        return "xnor"
    elif base_type in ['inv', 'not']:
        return "inv"
    elif base_type == 'buf':
        return "buf"
    elif 'aoi' in base_type:
        return "aoi"
    elif 'oai' in base_type:
        return "oai"
    else:
        return base_type


def size_gates_with_gp(circuit_topology, tech_parameters, options=None):
    """
    Size gates in a circuit using geometric programming to minimize a weighted combination
    of circuit delay and volume.
    
    Args:
        circuit_topology (dict): Circuit topology including gates, inputs, outputs, connections
                               and paths from inputs to outputs
        tech_parameters (dict): Technology parameters for each gate type including 
                              resistance, capacitance, and volume
        options (dict): Optional parameters for optimization
                      - volume_weight: Weight for volume in objective (default: 0.2)
                      - min_size: Minimum gate size (default: 1.0)
                      - power_constraint: Maximum power allowed (default: None)
                      - volume_constraint: Maximum volume allowed (default: None)
                      - verbose: Whether to print intermediate results (default: False)
    
    Returns:
        dict: Optimization results including:
             - gate_sizes: Dictionary of gate sizes
             - delay: Critical path delay
             - volume: Total circuit volume
             - power: Total circuit power
             - runtime: Optimization runtime
             - success: Whether optimization was successful
    """
    import time
    start_time = time.time()
    
    # Setup options with defaults
    options = options or {}
    volume_weight = options.get('volume_weight', 0.2)  # Changed from 0.5 to 0.2 to emphasize delay more
    min_size = options.get('min_size', 1)  
    max_size = options.get('max_size', 10.0)
    power_constraint = options.get('power_constraint', None)
    volume_constraint = options.get('volume_constraint', None)
    verbose = options.get('verbose', False)
    
    if verbose:
        print("\n=== STARTING GATE SIZING OPTIMIZATION ===")
        print(f"Parameters: volume_weight={volume_weight}, min_size={min_size}, max_size={max_size}")
        if power_constraint:
            print(f"Power constraint: {power_constraint}")
        if volume_constraint:
            print(f"Volume constraint: {volume_constraint}")
    
    # Extract key elements from the topology
    gates = circuit_topology['gates']
    inputs = circuit_topology['inputs']
    outputs = circuit_topology['outputs']
    paths = circuit_topology['all_paths']
    
    if verbose:
        print(f"Circuit: {len(gates)} gates, {len(inputs)} inputs, {len(outputs)} outputs, {len(paths)} paths")
    
    # Create mappings for gate lookup
    gate_name_to_idx = {}  # Map gate name to its index
    gate_idx_to_name = {}  # Map gate index to its name
    gate_name_to_type = {} # Map gate name to its type
    
    # Handle gates stored as tuples: (type, name, output, [inputs])
    idx_counter = 0  # Use a separate index counter to ensure sequential indices
    for i, gate in enumerate(gates):
        if isinstance(gate, tuple) and len(gate) >= 3:
            gate_type, gate_name, gate_output = gate[0], gate[1], gate[2]
            
            # Skip buffers (they're not real gates for optimization)
            if gate_type == 'buf':
                continue
                
            gate_name_to_idx[gate_name] = idx_counter
            gate_idx_to_name[idx_counter] = gate_name
            gate_name_to_type[gate_name] = gate_type
            idx_counter += 1
    
    if verbose:
        print(f"Processed {len(gate_name_to_idx)} gates for optimization (excluding buffers)")
    
    # Create mapping from output signals to gates
    output_signal_to_gate = {}
    for gate in gates:
        if isinstance(gate, tuple) and len(gate) >= 3:
            gate_type, gate_name, gate_output = gate[0], gate[1], gate[2]
            if gate_type != 'buf':  # Skip buffers
                output_signal_to_gate[gate_output] = gate_name
    
    # Extract gate parameters from tech_parameters
    gate_params = tech_parameters.get('gate_params', {})
    vdd = tech_parameters.get('vdd', 0.7)  # Default to 0.7V
    
    if verbose:
        print(f"Using Vdd = {vdd}V")
        print(f"Found parameters for {len(gate_params)} gates")
    
    # Initialize parameter arrays
    gate_volumes = np.zeros(len(gate_name_to_idx))
    gate_resistances = np.zeros(len(gate_name_to_idx))
    gate_input_caps = np.zeros(len(gate_name_to_idx))
    gate_internal_caps = np.zeros(len(gate_name_to_idx))
    gate_leakages = np.zeros(len(gate_name_to_idx))
    
    # Map parameters to gates
    param_mapped_count = 0
    missing_params_count = 0
    
    for gate_name, idx in gate_name_to_idx.items():
        if gate_name in gate_params:
            params = gate_params[gate_name]
            gate_volumes[idx] = params.get('vol', 1.0)
            gate_resistances[idx] = params.get('R', 1.0)
            gate_input_caps[idx] = params.get('C_in', 0.1)
            gate_internal_caps[idx] = params.get('C_int', 0.1)
            gate_leakages[idx] = params.get('I_leak', 0.001)
            param_mapped_count += 1
        else:
            # Use default values if parameters not available
            gate_volumes[idx] = 0.1
            gate_resistances[idx] = 1.0
            gate_input_caps[idx] = 0.1
            gate_internal_caps[idx] = 0.1
            gate_leakages[idx] = 0.001
            missing_params_count += 1
    
    if verbose:
        print(f"Successfully mapped parameters for {param_mapped_count} gates")
        if missing_params_count > 0:
            print(f"Warning: Using default parameters for {missing_params_count} gates")
    
    # Check for zero volumes - not allowed in GP
    zero_vols = np.sum(gate_volumes <= 0)
    if zero_vols > 0:
        if verbose:
            print(f"Warning: Found {zero_vols} gates with zero or negative volume. Setting to small value.")
        min_pos_vol = np.min(gate_volumes[gate_volumes > 0]) if np.any(gate_volumes > 0) else 0.001
        gate_volumes[gate_volumes <= 0] = min_pos_vol * 0.1
    
    # Normalize parameters to reduce numerical issues
    vol_mean = np.mean(gate_volumes)
    res_mean = np.mean(gate_resistances) 
    cap_mean = np.mean(gate_input_caps)
    int_cap_mean = np.mean(gate_internal_caps)
    leak_mean = np.mean(gate_leakages)
    
    # Store original values for final calculations
    orig_gate_volumes = gate_volumes.copy()
    orig_gate_resistances = gate_resistances.copy()
    orig_gate_input_caps = gate_input_caps.copy()
    orig_gate_internal_caps = gate_internal_caps.copy()
    orig_gate_leakages = gate_leakages.copy()
    
    # Normalize values (only if mean is non-zero)
    if vol_mean > 0:
        gate_volumes = gate_volumes / vol_mean
    if res_mean > 0:
        gate_resistances = gate_resistances / res_mean
    if cap_mean > 0:
        gate_input_caps = gate_input_caps / cap_mean
    if int_cap_mean > 0:
        gate_internal_caps = gate_internal_caps / int_cap_mean
    if leak_mean > 0:
        gate_leakages = gate_leakages / leak_mean
    
    if verbose:
        print("\n=== PARAMETER NORMALIZATION ===")
        print(f"Volume norm factor: {vol_mean:.2e}")
        print(f"Resistance norm factor: {res_mean:.2e}")
        print(f"Input capacitance norm factor: {cap_mean:.2e}")
        
        # Print parameter ranges
        print(f"Gate volume range: [{np.min(gate_volumes):.2e}, {np.max(gate_volumes):.2e}] (ratio: {np.max(gate_volumes)/np.min(gate_volumes):.2e})")
        print(f"Gate resistance range: [{np.min(gate_resistances):.2e}, {np.max(gate_resistances):.2e}] (ratio: {np.max(gate_resistances)/np.min(gate_resistances):.2e})")
        print(f"Input capacitance range: [{np.min(gate_input_caps):.2e}, {np.max(gate_input_caps):.2e}] (ratio: {np.max(gate_input_caps)/np.min(gate_input_caps):.2e})")
    
    # Get fanout relationships
    gate_to_gate_fanout = circuit_topology.get('gate_to_gate_fanout', {})
    
    if verbose:
        print("\n=== ANALYZING PATHS ===")
    
    # Process paths to identify gate sequences
    valid_paths = []
    gate_sequences = []
    path_to_primary_output = []  # Track if path ends at primary output
    
    for path_idx, path in enumerate(paths):
        # Extract gates in this path
        gate_idx_sequence = []
        last_gate_drives_output = False
        
        for i, node in enumerate(path):
            # Check if this node is a gate output signal
            if node in output_signal_to_gate:
                gate_name = output_signal_to_gate[node]
                if gate_name in gate_name_to_idx:
                    gate_idx_sequence.append(gate_name_to_idx[gate_name])
                    
                    # If this is the last node and it's in primary outputs, mark it
                    if i == len(path) - 1 and node in outputs:
                        last_gate_drives_output = True
            
            # If last node in path is a primary output but not a gate output
            elif i == len(path) - 1 and node in outputs:
                last_gate_drives_output = True
        
        # Only consider paths with gates
        if gate_idx_sequence:
            valid_paths.append(path)
            gate_sequences.append(gate_idx_sequence)
            path_to_primary_output.append(last_gate_drives_output)
    
    if verbose:
        print(f"Found {len(valid_paths)} valid paths with gates out of {len(paths)} total paths")
        print(f"Paths leading to primary outputs: {sum(path_to_primary_output)}")
        
        # Show sample paths
        if valid_paths:
            print("\nSample paths:")
            for i in range(min(3, len(valid_paths))):
                print(f"Path {i+1}: {' -> '.join(valid_paths[i])}")
                gate_names = [gate_idx_to_name[idx] for idx in gate_sequences[i]]
                print(f"  Gates: {' -> '.join(gate_names)}")
                print(f"  Ends at primary output: {path_to_primary_output[i]}")
    
    # If no valid paths found, optimization cannot proceed
    if not valid_paths:
        print("Error: No valid paths with gates found. Cannot perform optimization.")
        return {
            'gate_sizes': {gate_idx_to_name[i]: 1.0 for i in range(len(gate_name_to_idx))},
            'status': 'FAILED',
            'error': 'No valid paths found',
            'success': False,
            'delay': 0.0,
            'volume': np.sum(orig_gate_volumes),
            'power': 0.0,
            'runtime': time.time() - start_time
        }
    
    # Setup optimization problem
    if verbose:
        print("\n=== SETTING UP OPTIMIZATION PROBLEM ===")
    
    gate_size = cp.Variable(len(gate_name_to_idx), pos=True, name="gate_size")
    
    # Constraints: minimum and maximum sizes
    # Need to use the min_size passed in from options, not hard-code 1.0
    constraints = [gate_size >= min_size, gate_size <= max_size]
    
    # Calculate total circuit volume
    total_volume = cp.sum(cp.multiply(gate_volumes, gate_size))
    
    # Print shape analysis for debugging
    if verbose:
        print("\n=== SHAPE ANALYSIS FOR NUMERICAL STABILITY ===")
        print(f"Gate volume tensor shape: {gate_volumes.shape}")
        print(f"Gate size variable shape: {gate_size.shape}")
        print(f"Gate resistance tensor shape: {gate_resistances.shape}")
        print(f"Gate input cap tensor shape: {gate_input_caps.shape}")
        
        # Check condition numbers to identify potential numerical issues
        try:
            vol_condition = np.max(gate_volumes) / np.min(gate_volumes) if np.min(gate_volumes) > 0 else float('inf')
            res_condition = np.max(gate_resistances) / np.min(gate_resistances) if np.min(gate_resistances) > 0 else float('inf')
            cap_condition = np.max(gate_input_caps) / np.min(gate_input_caps) if np.min(gate_input_caps) > 0 else float('inf')
            
            print(f"Volume condition number: {vol_condition:.1e}")
            print(f"Resistance condition number: {res_condition:.1e}")
            print(f"Input capacitance condition number: {cap_condition:.1e}")
            
            if vol_condition > 1e6 or res_condition > 1e6 or cap_condition > 1e6:
                print("WARNING: High condition numbers detected - may cause numerical instability")
                print("Consider additional parameter normalization or pre-conditioning")
        except Exception as e:
            print(f"Error calculating condition numbers: {e}")
    
    # Calculate leakage power if a constraint is specified
    if power_constraint is not None:
        total_leakage = vdd * cp.sum(cp.multiply(gate_leakages, gate_size)) * leak_mean
        constraints.append(total_leakage <= power_constraint)
    
    # Add volume constraint if specified
    if volume_constraint is not None:
        # Denormalize the volume constraint - since we normalized the gate volumes
        normalized_volume_constraint = volume_constraint / vol_mean
        constraints.append(total_volume <= normalized_volume_constraint)
        if verbose:
            print(f"Added volume constraint: {volume_constraint} (normalized: {normalized_volume_constraint:.4f})")
    
    # Process each path to calculate delay
    path_delays = []
    
    path_start_time = time.time()
    total_path_count = len(gate_sequences)
    print(f"Setting up delays for {total_path_count} paths...")
    
    # For very large path counts, adjust progress reporting frequency
    report_frequency = max(1, min(50, total_path_count // 100))  # Report at most 100 times, at least once
    
    for path_idx, gate_idx_sequence in enumerate(gate_sequences):
        if verbose and path_idx % report_frequency == 0:
            elapsed = time.time() - path_start_time
            if path_idx > 0:
                avg_time_per_path = elapsed / path_idx
                est_remaining = avg_time_per_path * (total_path_count - path_idx)
                print(f"Processing path {path_idx+1}/{total_path_count}... ({elapsed:.2f}s elapsed, ~{est_remaining:.2f}s remaining)")
                print(f"  Progress: {path_idx/total_path_count*100:.1f}% complete")
            else:
                print(f"Processing path {path_idx+1}/{total_path_count}...")
        
        # Calculate delay for this path
        path_delay = 0
        
        for i, gate_idx in enumerate(gate_idx_sequence):
            # Intrinsic delay (R/size)
            intrinsic_delay = gate_resistances[gate_idx] / gate_size[gate_idx]
            
            # Load capacitance calculation
            load_cap = gate_internal_caps[gate_idx] * gate_size[gate_idx]
            
            # Add load from next gate in path if it exists
            if i < len(gate_idx_sequence) - 1:
                next_gate_idx = gate_idx_sequence[i + 1]
                load_cap += gate_input_caps[next_gate_idx] * gate_size[next_gate_idx]
            elif path_to_primary_output[path_idx]:
                # This gate drives a primary output
                # Use a fixed load capacitance for primary outputs
                # Ideally, this would come from tech_parameters['PO_caps']
                load_cap += 0.5 * cap_mean  # Default PO capacitance
            else:
                # Add load from fanout gates not in this path
                gate_name = gate_idx_to_name[gate_idx]
                has_fanout = False
                if gate_name in gate_to_gate_fanout:
                    for fanout_gate in gate_to_gate_fanout[gate_name]:
                        if fanout_gate in gate_name_to_idx:
                            fanout_idx = gate_name_to_idx[fanout_gate]
                            # Only add if this is not the next gate in path (already accounted for)
                            if i == len(gate_idx_sequence) - 1 or fanout_idx != gate_idx_sequence[i + 1]:
                                load_cap += gate_input_caps[fanout_idx] * gate_size[fanout_idx]
                                has_fanout = True
                
                # If no fanout was added, add a small default load
                if not has_fanout:
                    load_cap += 0.01 * cap_mean  # Small default capacitance
            
            # Add gate delay (R/size * C)
            gate_delay = intrinsic_delay * load_cap
            path_delay += gate_delay
        
        path_delays.append(path_delay)
    
    path_end_time = time.time()
    print(f"Path delay setup completed in {path_end_time - path_start_time:.2f} seconds")
    print(f"Average time per path: {(path_end_time - path_start_time)/len(gate_sequences):.4f} seconds")
    
    # Check for potential numerical issues in the problem
    if verbose:
        # Calculate statistics about the coefficients in the problem
        try:
            # CVXPY expressions don't have values until after solve(), 
            # so we can't access .value before solving
            print("\n=== PROBLEM COEFFICIENT ANALYSIS ===")
            print(f"Total optimization variables: {len(gate_name_to_idx)}")
            print(f"Total delay constraints: {len(path_delays)}")
                
            # Calculate constraint sparsity 
            total_params = len(gate_name_to_idx)
            avg_gates_per_path = sum(len(seq) for seq in gate_sequences) / len(gate_sequences) if gate_sequences else 0
            sparsity = avg_gates_per_path / total_params if total_params > 0 else 0
            print(f"Average gates per path: {avg_gates_per_path:.2f}")
            print(f"Constraint density: {sparsity:.4f} ({sparsity*100:.2f}%)")
        except Exception as e:
            print(f"Warning: Error during problem coefficient analysis: {e}")
            traceback.print_exc()
    
    # Circuit delay is the maximum path delay
    if len(path_delays) == 1:
        circuit_delay = path_delays[0]
    else:
        circuit_delay = cp.maximum(*path_delays)
    
    # Objective function: weighted sum of delay and volume
    # Handle the special case where volume_weight is 0.0
    if volume_weight == 0.0:
        objective = circuit_delay
    else:
        objective = volume_weight * total_volume + (1 - volume_weight) * circuit_delay
    
    # Create and solve the problem
    if verbose:
        if volume_weight == 0.0:
            print(f"\nSolving GP with objective: delay only (volume_weight = 0.0)")
        else:
            print(f"\nSolving GP with objective: {volume_weight:.2f} * volume + {1-volume_weight:.2f} * delay")
        print(f"Problem has {len(gate_name_to_idx)} variables and {len(path_delays)} delay constraints")
    
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    # Try multiple solvers with progressive fallback - start with SCS which works best
    solvers = ['SCS', 'ECOS', 'MOSEK']  # Changed order to try SCS first
    final_status = None
    solver_statuses = {}
    
    for solver in solvers:
        try:
            if verbose:
                print(f"\nAttempting to solve with {solver}...")
                print(f"Problem characteristics: {len(gate_name_to_idx)} variables, {len(path_delays)} delay constraints")
                solver_start_time = time.time()
            
            # Different solvers support different options
            if solver == 'MOSEK':
                # MOSEK supports full GP mode
                solver_verbose = True if verbose else False
                problem.solve(solver=solver, gp=True, verbose=solver_verbose)
            else:
                # Other solvers may not support all options
                solver_verbose = True if verbose else False
                problem.solve(solver=solver, gp=True, verbose=solver_verbose)
            
            if verbose:
                solver_end_time = time.time()
                print(f"{solver} completed in {solver_end_time - solver_start_time:.2f} seconds")
                print(f"{solver} status: {problem.status}")
                if hasattr(problem, 'value'):
                    print(f"{solver} objective value: {problem.value}")
                    
                # More detailed diagnostics
                if problem.status in ['OPTIMAL', 'OPTIMAL_INACCURATE']:
                    gate_size_vals = gate_size.value
                    min_size_val = np.min(gate_size_vals)
                    max_size_val = np.max(gate_size_vals)
                    avg_size_val = np.mean(gate_size_vals)
                    std_size_val = np.std(gate_size_vals)
                    
                    print(f"\nSolution statistics:")
                    print(f"  Min gate size: {min_size_val:.4f}")
                    print(f"  Max gate size: {max_size_val:.4f}")
                    print(f"  Avg gate size: {avg_size_val:.4f}")
                    print(f"  Std dev of sizes: {std_size_val:.4f}")
                    print(f"  Size range: {max_size_val/min_size_val:.2f}x")
                    
                    sizes_at_min = np.sum(np.isclose(gate_size_vals, min_size))
                    sizes_at_max = np.sum(np.isclose(gate_size_vals, max_size))
                    
                    print(f"  Gates at min size: {sizes_at_min} ({sizes_at_min/len(gate_size_vals)*100:.1f}%)")
                    print(f"  Gates at max size: {sizes_at_max} ({sizes_at_max/len(gate_size_vals)*100:.1f}%)")
            
            solver_statuses[solver] = problem.status
            
            # Check for successful status - note that ECOS/SCS may report "optimal" (lowercase)
            if problem.status in ['OPTIMAL', 'OPTIMAL_INACCURATE', 'optimal']:
                final_status = problem.status
                if verbose:
                    print(f"{solver} found solution with status: {problem.status}")
                break
            else:
                if verbose:
                    print(f"{solver} failed with status: {problem.status}")
                    print(f"Trying next solver...")
        except Exception as e:
            if verbose:
                print(f"Error with {solver}: {str(e)}")
                print(f"Detailed error information:")
                traceback.print_exc()
            solver_statuses[solver] = 'ERROR'
            print(f"Moving to next solver after error...")
    
    # Process optimization results
    if final_status in ['OPTIMAL', 'OPTIMAL_INACCURATE', 'optimal']:
        # Success case
        gate_sizes = gate_size.value
        
        # Calculate achieved metrics with original (denormalized) parameters
        achieved_volume = 0
        achieved_leakage = 0
        
        for i in range(len(gate_name_to_idx)):
            achieved_volume += orig_gate_volumes[i] * gate_sizes[i]
            achieved_leakage += orig_gate_leakages[i] * gate_sizes[i]
        
        achieved_power = vdd * achieved_leakage
        
        # Convert to dictionary mapping gate names to sizes
        gate_size_dict = {gate_idx_to_name[i]: float(gate_sizes[i]) for i in range(len(gate_name_to_idx))}
        
        # Calculate max path delay with original parameters
        max_delay = 0
        critical_path_idx = None
        
        for path_idx, gate_idx_sequence in enumerate(gate_sequences):
            path_delay = 0
            
            for i, gate_idx in enumerate(gate_idx_sequence):
                # Intrinsic delay calculation with original parameters
                intrinsic_delay = orig_gate_resistances[gate_idx] / gate_sizes[gate_idx]
                
                # Load capacitance calculation
                load_cap = orig_gate_internal_caps[gate_idx] * gate_sizes[gate_idx]
                
                # Add load from next gate in path if it exists
                if i < len(gate_idx_sequence) - 1:
                    next_gate_idx = gate_idx_sequence[i + 1]
                    load_cap += orig_gate_input_caps[next_gate_idx] * gate_sizes[next_gate_idx]
                elif path_to_primary_output[path_idx]:
                    # Default load for primary outputs
                    load_cap += 0.5
                else:
                    # Add load from fanout gates not in this path
                    gate_name = gate_idx_to_name[gate_idx]
                    has_fanout = False
                    if gate_name in gate_to_gate_fanout:
                        for fanout_gate in gate_to_gate_fanout[gate_name]:
                            if fanout_gate in gate_name_to_idx:
                                fanout_idx = gate_name_to_idx[fanout_gate]
                                if i == len(gate_idx_sequence) - 1 or fanout_idx != gate_idx_sequence[i + 1]:
                                    load_cap += orig_gate_input_caps[fanout_idx] * gate_sizes[fanout_idx]
                                    has_fanout = True
                    
                    # If no fanout was added, add a small default load
                    if not has_fanout:
                        load_cap += 0.01  # Small default capacitance
                
                # Gate delay
                gate_delay = intrinsic_delay * load_cap
                path_delay += gate_delay
            
            if path_delay > max_delay:
                max_delay = path_delay
                critical_path_idx = path_idx
        
        # Print optimization results
        if verbose:
            print("\n=== OPTIMIZATION RESULTS ===")
            print(f"Status: {final_status}")
            print(f"Optimal delay: {max_delay:.3f} ps")
            print(f"Optimal volume: {achieved_volume:.3f} μm³")
            print(f"Power consumption: {achieved_power:.3f} μW")
            
            # Stats about gate sizes
            min_size_val = np.min(gate_sizes)
            max_size_val = np.max(gate_sizes)
            avg_size = np.mean(gate_sizes)
            print(f"Gate size statistics: min={min_size_val:.2f}x, max={max_size_val:.2f}x, avg={avg_size:.2f}x")
            
            # Print critical path information
            if critical_path_idx is not None:
                crit_path = valid_paths[critical_path_idx]
                print(f"\nCritical path: {' -> '.join(crit_path)}")
                crit_path_gates = [gate_idx_to_name[idx] for idx in gate_sequences[critical_path_idx]]
                print(f"Critical path gates: {' -> '.join(crit_path_gates)}")
        
        # Return successful result
        return {
            'gate_sizes': gate_size_dict,
            'delay': max_delay,
            'volume': achieved_volume,
            'power': achieved_power,
            'status': final_status,
            'objective': float(problem.value),
            'runtime': time.time() - start_time,
            'success': True
        }
    else:
        # Failure case - provide diagnostics
        if verbose:
            print("\n=== OPTIMIZATION FAILED ===")
            print(f"Problem status: {final_status}")
            print(f"Solver statuses: {solver_statuses}")
            
            # Provide specific diagnosis based on status
            if any(status == 'INFEASIBLE' for status in solver_statuses.values()):
                print("Diagnosis: Problem is infeasible - constraints cannot be satisfied simultaneously.")
            elif any(status == 'UNBOUNDED' for status in solver_statuses.values()):
                print("Diagnosis: Problem is unbounded - objective can be made arbitrarily small.")
            else:
                print("Diagnosis: Solver failed to converge. Problem may be numerically difficult.")
                print(f"  - Ratio of max/min gate volume: {np.max(gate_volumes)/np.min(gate_volumes):.2e}")
                print(f"  - Ratio of max/min gate resistance: {np.max(gate_resistances)/np.min(gate_resistances):.2e}")
                print(f"  - Ratio of max/min input capacitance: {np.max(gate_input_caps)/np.min(gate_input_caps):.2e}")
        
        # Return default gate sizes (all 1.0)
        default_gate_sizes = {gate_idx_to_name[i]: 1.0 for i in range(len(gate_name_to_idx))}
        
        return {
            'gate_sizes': default_gate_sizes,
            'delay': float('inf'),
            'volume': np.sum(orig_gate_volumes),
            'power': 0.0,
            'status': 'FAILED',
            'error': 'Optimization failed to converge',
            'runtime': time.time() - start_time,
            'success': False
        }


def analyze_delay_volume_tradeoff(circuit_topology, tech_parameters, options=None):
    """
    Analyze the delay-volume tradeoff by performing multiple gate sizing optimizations
    with different volume weights.
    
    This function sweeps the volume weight parameter from 0 (minimize delay only)
    to 1 (minimize volume only) and returns the Pareto optimal solutions.
    
    Args:
        circuit_topology (dict): Circuit topology from extract_circuit_topology()
        tech_parameters (dict): Technology parameters from load_asap7_parameters()
        options (dict): Analysis options with the following keys:
            - num_points: Number of weight values to sample (default: 11)
            - min_size: Minimum gate size multiplier (default: 1.0)
            - max_size: Maximum gate size multiplier (default: 10.0)
            - power_constraint: Maximum power consumption (default: None)
            - verbose: Whether to print verbose output (default: True)
            
    Returns:
        dict: Analysis results with the following keys:
            - weights: List of volume weights used
            - delays: List of optimal delays for each weight
            - volumes: List of optimal volumes for each weight
            - powers: List of corresponding power consumption values
            - gate_sizes: Dictionary mapping each weight to the optimal gate sizes
            - pareto_optimal: List of indices corresponding to Pareto optimal solutions
    """
    if options is None:
        options = {}
    
    num_points = options.get('num_points', 11)
    min_size = options.get('min_size', 1.0)
    max_size = options.get('max_size', 10.0)
    power_constraint = options.get('power_constraint', None)
    verbose = options.get('verbose', True)
    
    # Generate volume weights to test
    weights = np.linspace(0.0, 1.0, num_points)
    
    if verbose:
        print(f"\nPerforming delay-volume tradeoff analysis with {num_points} points...")
        print(f"Volume weights: {weights}")
    
    # Results storage
    delays = []
    volumes = []
    powers = []
    all_gate_sizes = {}
    
    # Perform optimization for each volume weight
    for i, weight in enumerate(weights):
        if verbose:
            print(f"\nOptimization {i+1}/{num_points}: volume_weight = {weight:.2f}")
        
        # Set optimization options
        opt_options = {
            'volume_weight': weight,
            'min_size': min_size,
            'max_size': max_size,
            'power_constraint': power_constraint,
            'verbose': verbose
        }
        
        # Perform gate sizing optimization
        result = size_gates_with_gp(circuit_topology, tech_parameters, opt_options)
        
        if result['success']:
            delays.append(result['delay'])
            volumes.append(result['volume'])
            powers.append(result['power'])
            all_gate_sizes[weight] = result['gate_sizes']
        else:
            if verbose:
                print(f"Optimization for weight {weight:.2f} failed. Using default values.")
            
            # Use default values for failed optimizations
            delays.append(float('inf'))
            volumes.append(sum(tech_parameters['gate_params'][gate]['vol'] for gate in tech_parameters['gate_params']))
            powers.append(0.0)
            all_gate_sizes[weight] = {gate: 1.0 for gate in tech_parameters['gate_params']}
    
    # Identify Pareto optimal solutions
    # A solution is Pareto optimal if no other solution is better in both delay and volume
    pareto_optimal = []
    for i in range(len(delays)):
        is_pareto = True
        for j in range(len(delays)):
            if i != j and delays[j] <= delays[i] and volumes[j] <= volumes[i]:
                # Solution j dominates solution i
                is_pareto = False
                break
        if is_pareto:
            pareto_optimal.append(i)
    
    if verbose:
        print("\nDelay-Volume Tradeoff Analysis Results:")
        print("----------------------------------------")
        print(f"{'Weight':^10} | {'Delay (ps)':^15} | {'Volume (μm³)':^15} | {'Power (μW)':^15} | {'Pareto Optimal':^15}")
        print("-" * 75)
        
        for i, weight in enumerate(weights):
            is_pareto = i in pareto_optimal
            print(f"{weight:^10.2f} | {delays[i]:^15.2f} | {volumes[i]:^15.6f} | {powers[i]:^15.6f} | {is_pareto:^15}")
    
    return {
        'weights': list(weights),
        'delays': delays,
        'volumes': volumes,
        'powers': powers,
        'gate_sizes': all_gate_sizes,
        'pareto_optimal': pareto_optimal
    }


def plot_delay_volume_tradeoff(analysis_results, output_file=None):
    """
    Plot the delay-volume tradeoff from the analysis results.
    
    This function creates two plots:
    1. Delay vs. Volume, highlighting Pareto optimal solutions
    2. Delay and Volume vs. Volume Weight
    
    Args:
        analysis_results (dict): Results from analyze_delay_volume_tradeoff()
        output_file (str): Path to save the plot (if None, plot is displayed)
        
    Returns:
        None
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("Matplotlib is required for plotting. Please install it with:")
        print("pip install matplotlib")
        return
    
    weights = analysis_results['weights']
    delays = analysis_results['delays']
    volumes = analysis_results['volumes']
    pareto_optimal = analysis_results['pareto_optimal']
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 1, height_ratios=[1, 1])
    
    # Plot 1: Delay vs. Volume
    ax1 = fig.add_subplot(gs[0])
    
    # Plot all solutions
    ax1.scatter(volumes, delays, color='blue', label='All Solutions')
    
    # Highlight Pareto optimal solutions
    pareto_volumes = [volumes[i] for i in pareto_optimal]
    pareto_delays = [delays[i] for i in pareto_optimal]
    ax1.scatter(pareto_volumes, pareto_delays, color='red', label='Pareto Optimal')
    
    # Connect Pareto optimal points with a line
    pareto_points = sorted(zip(pareto_volumes, pareto_delays), key=lambda x: x[0])
    if pareto_points:
        pareto_x, pareto_y = zip(*pareto_points)
        ax1.plot(pareto_x, pareto_y, 'r--', alpha=0.7)
    
    ax1.set_xlabel('Volume (μm³)')
    ax1.set_ylabel('Delay (ps)')
    ax1.set_title('Delay vs. Volume Tradeoff')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for volume weights
    for i, w in enumerate(weights):
        ax1.annotate(f"{w:.1f}", (volumes[i], delays[i]), 
                     textcoords="offset points", xytext=(0,5), ha='center')
    
    # Plot 2: Delay and Volume vs. Volume Weight
    ax2 = fig.add_subplot(gs[1])
    
    # Primary y-axis: Delay
    color1 = 'blue'
    ax2.set_xlabel('Volume Weight')
    ax2.set_ylabel('Delay (ps)', color=color1)
    line1 = ax2.plot(weights, delays, 'o-', color=color1, label='Delay')
    ax2.tick_params(axis='y', labelcolor=color1)
    
    # Secondary y-axis: Volume
    color2 = 'red'
    ax3 = ax2.twinx()
    ax3.set_ylabel('Volume (μm³)', color=color2)
    line2 = ax3.plot(weights, volumes, 'o-', color=color2, label='Volume')
    ax3.tick_params(axis='y', labelcolor=color2)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right')
    
    ax2.set_title('Delay and Volume vs. Volume Weight')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Highlight Pareto optimal points
    for i in pareto_optimal:
        ax2.plot(weights[i], delays[i], 'o', markersize=10, 
                 markerfacecolor='none', markeredgecolor='green', markeredgewidth=2)
        ax3.plot(weights[i], volumes[i], 'o', markersize=10, 
                 markerfacecolor='none', markeredgecolor='green', markeredgewidth=2)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def validate_circuit_topology(circuit_topology):
    """
    Validate that the circuit topology was parsed correctly.
    
    Args:
        circuit_topology: Dictionary containing circuit topology information
        
    Returns:
        bool: True if the topology is valid, False otherwise
    """
    print("\n=== VALIDATING CIRCUIT TOPOLOGY ===")
    
    # Check that essential keys are present
    required_keys = ['module_name', 'inputs', 'outputs', 'gates', 'circuit_graph', 
                     'gate_drivers', 'gate_types', 'signal_load']
    
    for key in required_keys:
        if key not in circuit_topology:
            print(f"ERROR: Missing required key in circuit topology: {key}")
            return False
    
    # Check that there are gates in the circuit
    if not circuit_topology['gates']:
        print("ERROR: No gates found in the circuit")
        return False
    
    # Check that there are inputs and outputs
    if not circuit_topology['inputs']:
        print("ERROR: No primary inputs found in the circuit")
        return False
    
    if not circuit_topology['outputs']:
        print("ERROR: No primary outputs found in the circuit")
        return False
    
    # Check if gate types are consistent
    gate_types = circuit_topology['gate_types']
    for gate_type, gate_name, _, _ in circuit_topology['gates']:
        if gate_type == 'buf':
            continue  # Skip buffers
        if gate_name not in gate_types:
            print(f"ERROR: Gate {gate_name} has no type information")
            return False
        if gate_types[gate_name] != gate_type:
            print(f"WARNING: Gate {gate_name} has inconsistent type: {gate_types[gate_name]} vs {gate_type}")
    
    # Check connectivity
    for gate_type, gate_name, output, inputs in circuit_topology['gates']:
        if gate_type == 'buf':
            continue  # Skip buffers
        
        # Check output signal
        if not output:
            print(f"ERROR: Gate {gate_name} has no output signal")
            return False
        
        # Check input signals
        if not inputs:
            print(f"ERROR: Gate {gate_name} has no input signals")
            return False
    
    # Check that there are paths in the circuit
    if 'all_paths' not in circuit_topology or not circuit_topology['all_paths']:
        print("ERROR: No paths found in the circuit")
        return False
    
    # Detailed statistics
    gate_count = len([g for g in circuit_topology['gates'] if g[0] != 'buf'])
    gate_type_distribution = {}
    for gate_type, _, _, _ in circuit_topology['gates']:
        if gate_type != 'buf':
            gate_type_distribution[gate_type] = gate_type_distribution.get(gate_type, 0) + 1
    
    print(f"Circuit validation successful:")
    print(f"- Module name: {circuit_topology['module_name']}")
    print(f"- Primary inputs: {len(circuit_topology['inputs'])}")
    print(f"- Primary outputs: {len(circuit_topology['outputs'])}")
    print(f"- Total gates: {gate_count}")
    print(f"- Gate type distribution: {gate_type_distribution}")
    print(f"- Total paths: {len(circuit_topology['all_paths'])}")
    
    # Print a few sample paths for verification
    print("\nSample paths:")
    for i, path in enumerate(circuit_topology['all_paths'][:3]):
        print(f"- Path {i+1}: {' -> '.join(path)}")
    
    return True


def validate_tech_parameters(tech_parameters, circuit_topology):
    """
    Validate that the technology parameters were loaded correctly.
    
    Args:
        tech_parameters: Dictionary containing technology parameters
        circuit_topology: Dictionary containing circuit topology
        
    Returns:
        bool: True if the parameters are valid, False otherwise
    """
    print("\n=== VALIDATING TECHNOLOGY PARAMETERS ===")
    
    # Check that essential keys are present
    required_keys = ['gate_params', 'PO_caps', 'freqs', 'vdd']
    
    for key in required_keys:
        if key not in tech_parameters:
            print(f"ERROR: Missing required key in tech parameters: {key}")
            return False
    
    # Check that parameters exist for all gates in the circuit
    gate_params = tech_parameters['gate_params']
    missing_gates = []
    
    for gate_type, gate_name, _, _ in circuit_topology['gates']:
        if gate_type == 'buf':
            continue  # Skip buffers
        
        if gate_name not in gate_params:
            missing_gates.append((gate_name, gate_type))
    
    if missing_gates:
        print(f"ERROR: Missing parameters for {len(missing_gates)} gates:")
        for gate_name, gate_type in missing_gates[:10]:  # Show only first 10 to avoid flooding
            print(f"- {gate_name} ({gate_type})")
        if len(missing_gates) > 10:
            print(f"... and {len(missing_gates) - 10} more")
        return False
    
    # Check required parameters for each gate
    required_gate_params = ['vol', 'R', 'C_in', 'C_int', 'I_leak']
    param_issues = []
    
    for gate_name, params in gate_params.items():
        for param in required_gate_params:
            if param not in params:
                param_issues.append((gate_name, param))
    
    if param_issues:
        print(f"ERROR: Missing gate parameters:")
        for gate_name, param in param_issues[:10]:
            print(f"- {gate_name} is missing {param}")
        if len(param_issues) > 10:
            print(f"... and {len(param_issues) - 10} more issues")
        return False
    
    # Check for reasonable parameter values
    value_issues = []
    
    for gate_name, params in gate_params.items():
        # Volume should be positive
        if params['vol'] <= 0:
            value_issues.append((gate_name, 'vol', params['vol']))
        
        # Resistance should be positive
        if params['R'] <= 0:
            value_issues.append((gate_name, 'R', params['R']))
        
        # Input capacitance should be positive
        if params['C_in'] <= 0:
            value_issues.append((gate_name, 'C_in', params['C_in']))
        
        # Internal capacitance should be positive
        if params['C_int'] <= 0:
            value_issues.append((gate_name, 'C_int', params['C_int']))
    
    if value_issues:
        print(f"ERROR: Invalid parameter values:")
        for gate_name, param, value in value_issues[:10]:
            print(f"- {gate_name}.{param} = {value} (should be positive)")
        if len(value_issues) > 10:
            print(f"... and {len(value_issues) - 10} more issues")
        return False
    
    # Check supply voltage
    if tech_parameters['vdd'] <= 0:
        print(f"ERROR: Invalid supply voltage: {tech_parameters['vdd']} (should be positive)")
        return False
    
    # Show statistics about the parameters
    print("Parameter validation successful:")
    print(f"- Total gates with parameters: {len(gate_params)}")
    print(f"- Supply voltage: {tech_parameters['vdd']} V")
    
    # Print parameter ranges to check for reasonable values
    vol_values = [params['vol'] for params in gate_params.values()]
    r_values = [params['R'] for params in gate_params.values()]
    c_in_values = [params['C_in'] for params in gate_params.values()]
    c_int_values = [params['C_int'] for params in gate_params.values()]
    i_leak_values = [params['I_leak'] for params in gate_params.values()]
    
    print("\nParameter ranges:")
    print(f"- Volume (μm²): min={min(vol_values):.6f}, max={max(vol_values):.6f}, avg={sum(vol_values)/len(vol_values):.6f}")
    print(f"- Resistance (kΩ): min={min(r_values):.2f}, max={max(r_values):.2f}, avg={sum(r_values)/len(r_values):.2f}")
    print(f"- Input capacitance (fF): min={min(c_in_values):.2f}, max={max(c_in_values):.2f}, avg={sum(c_in_values)/len(c_in_values):.2f}")
    print(f"- Internal capacitance (fF): min={min(c_int_values):.2f}, max={max(c_int_values):.2f}, avg={sum(c_int_values)/len(c_int_values):.2f}")
    print(f"- Leakage current (nA): min={min(i_leak_values):.4f}, max={max(i_leak_values):.4f}, avg={sum(i_leak_values)/len(i_leak_values):.4f}")
    
    # Print sample parameters for a few gates
    print("\nSample gate parameters:")
    for i, (gate_name, params) in enumerate(gate_params.items()):
        if i >= 5:  # Only show the first 5 gates
            break
        
        gate_type = circuit_topology['gate_types'].get(gate_name, "unknown")
        print(f"- {gate_name} ({gate_type}):")
        print(f"  - Volume: {params['vol']:.6f} μm²")
        print(f"  - Resistance: {params['R']:.2f} kΩ")
        print(f"  - Input capacitance: {params['C_in']:.2f} fF")
        print(f"  - Internal capacitance: {params['C_int']:.2f} fF")
        print(f"  - Leakage current: {params['I_leak']:.4f} nA")
        print(f"  - Activity frequency: {tech_parameters['freqs'].get(gate_name, 0):.1f} MHz")
    
    return True


if __name__ == "__main__":
    # Set up paths and command-line arguments
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths for ISCAS benchmarks and parameters
    iscas_dir = os.path.join(script_dir, "ISCAS85")
    params_paths = [
        os.path.join(script_dir, "asap7_extracted_params.json")
        # Removed alternative paths in clean/asap7_params that no longer exist
    ]
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="ISCAS Gate Sizing using Geometric Programming")
    parser.add_argument("circuit", help="Circuit name (e.g., c432) or path to Verilog file")
    parser.add_argument("--params", help="Path to ASAP7 parameters file (optional)")
    parser.add_argument("--analyze-tradeoff", action="store_true", help="Perform delay-volume tradeoff analysis")
    parser.add_argument("--volume-weight", type=float, default=0.5, help="Weight for volume in objective (default: 0.5)")
    parser.add_argument("--min-size", type=float, default=1.0, help="Minimum gate size multiplier (default: 1.0)")
    parser.add_argument("--max-size", type=float, default=10.0, help="Maximum gate size multiplier (default: 10.0)")
    parser.add_argument("--volume-constraint", type=float, help="Maximum allowed volume (optional)")
    parser.add_argument("--vary-volume-constraint", action="store_true", help="Run multiple optimizations with varying volume constraints")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    # Determine Verilog file path
    verilog_file = args.circuit
    if not os.path.exists(verilog_file):
        # If just a circuit name is provided (e.g., c432), look in ISCAS85 directory
        if not verilog_file.endswith(".v"):
            verilog_file = f"{verilog_file}.v"
        verilog_file = os.path.join(iscas_dir, verilog_file)
        
        if not os.path.exists(verilog_file):
            print(f"Error: Verilog file not found: {verilog_file}")
            print(f"Please provide a valid circuit name or path.")
            sys.exit(1)
    
    # Determine parameters file path
    params_file = args.params
    if params_file is None:
        # Try to find the parameters file in common locations
        for path in params_paths:
            if os.path.exists(path):
                params_file = path
                break
        
        if params_file is None:
            print("Warning: Could not automatically find ASAP7 parameters file.")
            print("Using the first path in the search list.")
            params_file = params_paths[0]
    
    print(f"\nRunning gate sizing optimization for circuit: {os.path.basename(verilog_file)}")
    print(f"Using parameters file: {params_file}")
    
    # Check if the parameters file exists
    if not os.path.exists(params_file):
        print(f"Error: Parameters file not found: {params_file}")
        print("Please provide a valid parameters file path with --params.")
        sys.exit(1)
    
    # Extract topology and load parameters
    circuit_topology = extract_circuit_topology(verilog_file)
    
    # Validate circuit topology
    if not validate_circuit_topology(circuit_topology):
        print("Circuit topology validation failed. Please check the Verilog file.")
        sys.exit(1)
    
    tech_parameters = load_asap7_parameters(params_file, circuit_topology)
    
    # Validate technology parameters
    if not validate_tech_parameters(tech_parameters, circuit_topology):
        print("Technology parameter validation failed. Please check the parameter file.")
        sys.exit(1)
    
    # Define optimization options
    optimization_options = {
        'volume_weight': args.volume_weight,
        'min_size': args.min_size,
        'max_size': args.max_size,
        'verbose': args.verbose
    }
    
    # Add volume constraint if specified
    if args.volume_constraint:
        optimization_options['volume_constraint'] = args.volume_constraint
    
    # If vary-volume-constraint option is specified, run multiple optimizations 
    # with different volume constraints and plot the results
    if args.vary_volume_constraint:
        if args.volume_weight != 0.0:
            print("Warning: For volume constraint analysis, volume_weight is being set to 0.0")
            optimization_options['volume_weight'] = 0.0
            
        print("\nPerforming volume constraint analysis...")
        
        # Determine a reasonable range of volume constraints
        # First get the volume with no constraints
        baseline_options = optimization_options.copy()
        if 'volume_constraint' in baseline_options:
            del baseline_options['volume_constraint']
        baseline_options['volume_weight'] = 1.0  # Minimize volume only
        
        print("Running baseline optimization to determine minimum volume...")
        baseline_results = size_gates_with_gp(circuit_topology, tech_parameters, baseline_options)
        min_volume = baseline_results['volume']
        
        baseline_options['volume_weight'] = 0.0  # Minimize delay only
        print("Running baseline optimization to determine volume with no constraints...")
        unconstrained_results = size_gates_with_gp(circuit_topology, tech_parameters, baseline_options)
        max_volume = unconstrained_results['volume']
        
        print(f"Volume range: [{min_volume:.4f}, {max_volume:.4f}]")
        
        # Generate a range of volume constraints
        num_points = 10
        volume_constraints = np.linspace(min_volume, max_volume * 1.2, num_points)
        
        delays = []
        volumes = []
        
        for vol_constraint in volume_constraints:
            print(f"\nRunning optimization with volume constraint: {vol_constraint:.4f}")
            options = optimization_options.copy()
            options['volume_constraint'] = vol_constraint
            options['volume_weight'] = 0.0  # Optimize delay only
            
            result = size_gates_with_gp(circuit_topology, tech_parameters, options)
            
            if result['success']:
                delays.append(result['delay'])
                volumes.append(result['volume'])
                print(f"  Delay: {result['delay']:.4f}, Volume: {result['volume']:.4f}")
            else:
                print(f"  Optimization failed with volume constraint {vol_constraint:.4f}")
        
        # Plot the results if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(volumes, delays, 'o-', color='blue')
            plt.xlabel('Volume Constraint')
            plt.ylabel('Delay (ps)')
            plt.title(f'Delay vs. Volume Constraint for {circuit_topology["module_name"]}')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save the plot
            circuit_name = os.path.splitext(os.path.basename(verilog_file))[0]
            plot_file = f"{circuit_name}_volume_vs_delay.pdf"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_file}")
            
            # Also show it if in an interactive environment
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting. Results summary:")
            print(f"{'Volume Constraint':^15} | {'Delay (ps)':^15}")
            print("-" * 35)
            for v, d in zip(volumes, delays):
                print(f"{v:^15.4f} | {d:^15.4f}")
    else:
        # Perform gate sizing optimization
        results = size_gates_with_gp(circuit_topology, tech_parameters, optimization_options)
    
    # Print optimization results
    if results['success']:
        print("\nOptimization Results:")
        print(f"  Circuit: {circuit_topology['module_name']}")
        print(f"  Optimal delay: {results['delay']:.2f} ps")
        print(f"  Optimal volume: {results['volume']:.6f} μm³")
        print(f"  Total power: {results['power']:.6f} μW")
        print(f"  Runtime: {results['runtime']:.2f} seconds")
        print(f"  Number of gates: {len(results['gate_sizes'])}")
        
        # Print sizes for the first few gates
        print("\nSample gate sizes:")
        for i, (gate, size) in enumerate(sorted(results['gate_sizes'].items())):
            if i >= 10:  # Only show the first 10 gates
                break
            gate_type = circuit_topology['gate_types'][gate]
            print(f"  {gate} ({gate_type}): {size:.2f}x")
    else:
        print("\nOptimization failed. Using default gate sizes (1.0x).")
        
    # Perform delay-volume tradeoff analysis if requested
    if args.analyze_tradeoff:
        print("\nPerforming delay-volume tradeoff analysis...")
        analysis_results = analyze_delay_volume_tradeoff(circuit_topology, tech_parameters, {
            'num_points': 5,  # Use 5 points for quick analysis
            'min_size': args.min_size,
            'max_size': args.max_size,
            'verbose': True
        })
        
        # Plot the results if matplotlib is available
        try:
            circuit_name = os.path.splitext(os.path.basename(verilog_file))[0]
            plot_file = f"{circuit_name}_tradeoff.pdf"
            plot_delay_volume_tradeoff(analysis_results, plot_file)
            print(f"Tradeoff plot saved to {plot_file}")
        except Exception as e:
            print(f"Warning: Could not create tradeoff plot: {e}") 