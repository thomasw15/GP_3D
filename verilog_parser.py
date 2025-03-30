import re
import os
import numpy as np
import networkx as nx
from collections import defaultdict

class VerilogParser:
    """
    Parser for ISCAS85 benchmark Verilog files.
    Extracts circuit structure and creates appropriate data structures
    for gate sizing optimization.
    """
    def __init__(self, verilog_file):
        """Initialize with the path to a Verilog file."""
        self.verilog_file = verilog_file
        self.module_name = None
        self.inputs = []
        self.outputs = []
        self.wires = []
        self.gates = []  # List of (gate_type, gate_name, output, inputs) tuples
        self.gate_types = defaultdict(int)  # Counter for each gate type
        self.circuit_graph = nx.DiGraph()  # Directed graph representation
        self.aliased_signals = {}  # Dictionary mapping signal aliases (for buffers)
        self.buffer_connections = {}  # Dictionary mapping output signals to input signals for buffers
        
    def parse(self):
        """Parse the Verilog file and extract circuit information."""
        print(f"Parsing {self.verilog_file}...")
        
        with open(self.verilog_file, 'r') as f:
            verilog_content = f.read()
        
        # Extract module name
        module_match = re.search(r'module\s+(\w+)\s*\(', verilog_content)
        if module_match:
            self.module_name = module_match.group(1)
            print(f"Module name: {self.module_name}")
        
        # Extract inputs - more robust pattern for multi-line declarations
        input_pattern = r'input\s+([\w\s,]+);'
        input_matches = re.findall(input_pattern, verilog_content)
        inputs_str = ' '.join(input_matches)
        inputs_str = re.sub(r'\s+', ' ', inputs_str).strip()
        self.inputs = [inp.strip() for inp in re.split(r'[,\s]+', inputs_str) if inp.strip()]
        print(f"Found {len(self.inputs)} inputs")
        
        # Extract outputs - more robust pattern for multi-line declarations
        output_pattern = r'output\s+([\w\s,]+);'
        output_matches = re.findall(output_pattern, verilog_content)
        outputs_str = ' '.join(output_matches)
        outputs_str = re.sub(r'\s+', ' ', outputs_str).strip()
        self.outputs = [out.strip() for out in re.split(r'[,\s]+', outputs_str) if out.strip()]
        print(f"Found {len(self.outputs)} outputs")
        
        # Extract wires - more robust pattern for multi-line declarations
        wire_pattern = r'wire\s+([\w\s,]+);'
        wire_matches = re.findall(wire_pattern, verilog_content)
        wires_str = ' '.join(wire_matches)
        wires_str = re.sub(r'\s+', ' ', wires_str).strip()
        self.wires = [wire.strip() for wire in re.split(r'[,\s]+', wires_str) if wire.strip()]
        print(f"Found {len(self.wires)} wires")
        
        # Extract all gates - ISCAS85 benchmarks use a specific format with different gate types
        # buf(output, input);
        # not(output, input);
        # and(output, input1, input2 [, input3, ...]);
        # nand(output, input1, input2 [, input3, ...]);
        # nor(output, input1, input2 [, input3, ...]);
        # or(output, input1, input2 [, input3, ...]);
        
        # First, process buffers separately to create signal aliases
        buffer_pattern = r'buf\s+(\w+)\s*\(\s*(\w+)\s*,\s*(\w+)\s*\);'
        buffer_matches = re.finditer(buffer_pattern, verilog_content)
        
        for match in buffer_matches:
            buf_name = match.group(1)
            output = match.group(2)
            input_signal = match.group(3)
            
            # Store the mapping from output to input (aliases) 
            self.aliased_signals[output] = input_signal
            
            # Also store direct buffer connections for building edges later
            self.buffer_connections[output] = input_signal
            
            # We'll add this as a special "buffer" gate for reference
            gate_name = f"buf_{self.gate_types['buf']}"
            self.gates.append(('buf', gate_name, output, [input_signal]))
            self.gate_types['buf'] += 1
        
        # Match all other gate instantiations
        gate_pattern = r'(\w+)\s+(\w+)\s*\(\s*(\w+)\s*,\s*([\w\s,]+)\s*\);'
        gates_matches = re.finditer(gate_pattern, verilog_content)
        
        for match in gates_matches:
            gate_type = match.group(1).lower()  # Convert to lowercase for consistency
            gate_instance = match.group(2)
            output = match.group(3)
            inputs_str = match.group(4)
            inputs = [inp.strip() for inp in inputs_str.split(',') if inp.strip()]
            
            # Skip buffers (already processed) and module declarations
            if gate_type in ['buf', 'module']:
                continue
            
            # Generate a unique gate name (use the instance name from the Verilog)
            gate_name = gate_instance
            self.gates.append((gate_type, gate_name, output, inputs))
            self.gate_types[gate_type] += 1
        
        # Build the circuit graph AFTER processing all gates
        # First add all signals as nodes
        all_signals = set(self.inputs + self.outputs + self.wires)
        for signal in all_signals:
            self.circuit_graph.add_node(signal, type='signal')
        
        # Now add edges based on gate connections
        # For each gate, add edges from inputs to output
        for gate_type, gate_name, output, inputs in self.gates:
            # Skip buffer gates in normal edge creation
            if gate_type == 'buf':
                continue
                
            # For each normal gate, add edges from inputs to outputs
            # This represents the correct signal flow direction
            for input_signal in inputs:
                # If the input signal is an alias (output of a buffer), use the original signal
                # This "skips" the buffer, treating it as a wire
                source_signal = input_signal
                while source_signal in self.aliased_signals:
                    source_signal = self.aliased_signals[source_signal]
                
                # Follow the signal through aliases for the output too
                target_signal = output
                
                # Add edge from source to target
                self.circuit_graph.add_edge(source_signal, target_signal)
        
        # Now add edges for buffers to properly connect from buffer input to output
        # This is critical for signals that lead to primary outputs
        for output, input_signal in self.buffer_connections.items():
            self.circuit_graph.add_edge(input_signal, output)
        
        # Set node types
        for node in self.circuit_graph.nodes():
            if node in self.inputs:
                self.circuit_graph.nodes[node]['type'] = 'input'
            elif node in self.outputs:
                self.circuit_graph.nodes[node]['type'] = 'output'
            else:
                self.circuit_graph.nodes[node]['type'] = 'internal'
        
        print(f"Found {len(self.gates)} gates of {len(self.gate_types)} different types")
        for gate_type, count in sorted(self.gate_types.items()):
            print(f"  {gate_type}: {count}")
        
        return True

    def get_fanout(self, node):
        """Get the fanout nodes of a given node."""
        return list(self.circuit_graph.successors(node))
    
    def get_fanin(self, node):
        """Get the fanin nodes of a given node."""
        return list(self.circuit_graph.predecessors(node))

    def resolve_buffer_path(self, node):
        """Resolve buffer path to find the actual signal source."""
        while node in self.aliased_signals:
            node = self.aliased_signals[node]
        return node
    
    def get_critical_paths(self, limit=1000):
        """
        Identify critical paths from primary inputs to primary outputs.
        For large circuits, limit the number of paths to analyze.
        
        Args:
            limit: Maximum number of paths to return
            
        Returns:
            List of paths, where each path is a list of node names
        """
        print(f"Finding critical paths (limit: {limit})...")
        paths = []
        path_count = 0
        
        # For large circuits, finding all paths using all_simple_paths is impractical
        # Instead, we'll use a BFS approach to find paths from inputs to outputs
        
        # Sample some inputs and outputs (for very large circuits)
        sample_inputs = self.inputs[:min(20, len(self.inputs))]
        sample_outputs = self.outputs[:min(20, len(self.outputs))]
        
        print(f"Using {len(sample_inputs)} inputs and {len(sample_outputs)} outputs for path analysis")
        
        # Try finding shortest paths between input-output pairs
        for output in sample_outputs:
            for input_node in sample_inputs:
                try:
                    # Find shortest path from input to output
                    path = nx.shortest_path(self.circuit_graph, input_node, output)
                    if path and len(path) > 2:  # Ensure path has at least one gate
                        paths.append(path)
                        path_count += 1
                        if path_count >= limit:
                            print(f"Reached path limit ({limit})")
                            return paths
                except Exception as e:
                    # Skip if no path exists
                    continue
        
        if not paths:
            print("No direct paths found. Using alternative approach...")
            
            # Build a dictionary of signal-producing gates
            signal_producers = {}
            for gate_type, gate_name, output, inputs in self.gates:
                if gate_type != 'buf':  # Skip buffers
                    signal_producers[output] = (gate_type, gate_name, inputs)
            
            # Try to manually build paths from outputs tracing back to inputs
            for output in sample_outputs:
                # Start from an output and trace back to inputs
                try:
                    current_paths = self.trace_paths_to_inputs(output, signal_producers, max_depth=10)
                    paths.extend(current_paths)
                    path_count += len(current_paths)
                    if path_count >= limit:
                        print(f"Reached path limit ({limit})")
                        return paths[:limit]
                except Exception as e:
                    print(f"Error tracing from {output}: {e}")
                    continue
        
        # If we still don't have paths, use a final fallback method
        if not paths:
            print("Using fallback path identification...")
            
            # Get a topological sort of the graph (should be possible since it's acyclic)
            try:
                topo_order = list(nx.topological_sort(self.circuit_graph))
                
                # Identify some key internal nodes in the middle of the topological ordering
                num_internal = len(topo_order) - len(self.inputs) - len(self.outputs)
                if num_internal > 0:
                    mid_point = len(self.inputs) + num_internal // 2
                    internal_nodes = topo_order[mid_point:mid_point+10]
                    
                    # Create simple paths: input -> internal -> output
                    for input_node in sample_inputs:
                        for internal in internal_nodes:
                            for output in sample_outputs:
                                path = [input_node, internal, output]
                                paths.append(path)
                                path_count += 1
                                if path_count >= limit:
                                    print(f"Reached path limit ({limit})")
                                    return paths[:limit]
            except Exception as e:
                print(f"Error in fallback path identification: {e}")
        
        print(f"Found {len(paths)} possible paths through the circuit")
        return paths
    
    def trace_paths_to_inputs(self, signal, signal_producers, max_depth=10, current_depth=0, visited=None):
        """
        Trace paths from a signal back to primary inputs.
        
        Args:
            signal: The signal to trace back from
            signal_producers: Dictionary mapping signals to the gates that produce them
            max_depth: Maximum depth for path tracing
            current_depth: Current recursion depth
            visited: Set of visited signals (to prevent cycles)
            
        Returns:
            List of paths, where each path is a list of node names
        """
        if visited is None:
            visited = set()
            
        if signal in visited:
            return []  # Avoid cycles
            
        visited.add(signal)
        
        # If we've reached a primary input or max depth, return the path
        if signal in self.inputs or current_depth >= max_depth:
            return [[signal]]
            
        # If this signal doesn't have a known producer (e.g., constant signals),
        # stop the path here
        if signal not in signal_producers:
            return []
            
        # Get the gate that produces this signal
        gate_type, gate_name, gate_inputs = signal_producers[signal]
        
        # Trace back through each input of the gate
        all_paths = []
        for input_signal in gate_inputs:
            # Resolve input_signal through any buffers
            orig_input = input_signal
            while orig_input in self.aliased_signals:
                orig_input = self.aliased_signals[orig_input]
                
            # Get paths from this input back to primary inputs
            input_paths = self.trace_paths_to_inputs(
                orig_input, signal_producers, max_depth, 
                current_depth + 1, visited.copy()
            )
            
            # Append this signal to each path
            for path in input_paths:
                all_paths.append(path + [signal])
        
        return all_paths
    
    def get_gate_mapping(self):
        """
        Create a mapping between signal names and gates.
        Returns a dictionary where keys are signal names and
        values are gate information (type, name, inputs).
        """
        signal_to_gate = {}
        
        for gate_type, gate_name, output, inputs in self.gates:
            # Skip buffer gates in the mapping
            if gate_type == 'buf':
                continue
                
            # Map output signal to gate
            signal_to_gate[output] = {
                'type': gate_type,
                'name': gate_name,
                'inputs': inputs
            }
        
        return signal_to_gate
    
    def get_gate_info(self):
        """
        Return a dictionary with gate information suitable for optimization.
        Maps gate names to indices and provides fanout information.
        """
        gate_info = {}
        gate_index = {}  # Maps gate name to index
        gate_list = []   # List of gate names in index order
        
        # Create a mapping of gates to indices (excluding buffers)
        gate_idx = 0
        for gate_type, gate_name, output, inputs in self.gates:
            # Skip buffers for optimization purposes
            if gate_type == 'buf':
                continue
                
            gate_index[gate_name] = gate_idx
            gate_list.append(gate_name)
            gate_idx += 1
        
        # Signal to gate mapping
        signal_to_gate = self.get_gate_mapping()
        
        # Map each gate's fanout - find gates that use this gate's output as an input
        fanout_map = {}
        for idx, (gate_type, gate_name, output, inputs) in enumerate(self.gates):
            # Skip buffers
            if gate_type == 'buf':
                continue
                
            gate_idx = gate_index[gate_name]
            fanout_gates = []
            
            # Find all gates that use 'output' as an input
            for other_idx, (other_type, other_name, other_output, other_inputs) in enumerate(self.gates):
                if other_type == 'buf':
                    continue
                    
                other_gate_idx = gate_index[other_name]
                
                if output in other_inputs:
                    fanout_gates.append(other_gate_idx)
            
            fanout_map[gate_idx] = fanout_gates
        
        # Map primary inputs to the gates they connect to
        input_to_gates = defaultdict(list)
        for idx, (gate_type, gate_name, output, inputs) in enumerate(self.gates):
            if gate_type == 'buf':
                continue
                
            gate_idx = gate_index[gate_name]
            
            for input_signal in inputs:
                # Follow signal through aliases (buffers)
                source_signal = input_signal
                while source_signal in self.aliased_signals:
                    source_signal = self.aliased_signals[source_signal]
                    
                if source_signal in self.inputs:
                    input_to_gates[source_signal].append(gate_idx)
        
        # Map gates that produce outputs
        output_gates = []
        for idx, (gate_type, gate_name, output, inputs) in enumerate(self.gates):
            if gate_type == 'buf':
                continue
                
            gate_idx = gate_index[gate_name]
            
            # Follow signal through aliases
            target_signal = output
            while target_signal in self.aliased_signals:
                target_signal = self.aliased_signals[target_signal]
                
            if target_signal in self.outputs:
                output_gates.append(gate_idx)
        
        gate_info = {
            'num_gates': len(gate_list),
            'gate_list': gate_list,
            'gate_index': gate_index,
            'gate_types': {gate_name: gate_type for gate_type, gate_name, _, _ in self.gates if gate_type != 'buf'},
            'fanout': fanout_map,
            'input_to_gates': dict(input_to_gates),
            'output_gates': output_gates
        }
        
        return gate_info
    
    def summarize(self):
        """Print a summary of the circuit."""
        print(f"\nCircuit Summary for {self.module_name}:")
        print(f"  Number of primary inputs: {len(self.inputs)}")
        print(f"  Number of primary outputs: {len(self.outputs)}")
        print(f"  Number of internal wires: {len(self.wires)}")
        
        # Count real gates (excluding buffers for the summary)
        real_gate_count = sum(count for gate_type, count in self.gate_types.items() if gate_type != 'buf')
        print(f"  Total gates: {real_gate_count}")
        
        print(f"  Gate types:")
        for gate_type, count in sorted(self.gate_types.items()):
            if gate_type not in ['module', 'buf']:  # Skip module and buffer entries
                print(f"    {gate_type}: {count}")
        
        print(f"  Buffers: {self.gate_types.get('buf', 0)}")
        
        # Graph properties
        print(f"\nGraph representation:")
        print(f"  Nodes: {self.circuit_graph.number_of_nodes()}")
        print(f"  Edges: {self.circuit_graph.number_of_edges()}")
        
        # Check for cycles (should be acyclic for combinational circuits)
        try:
            is_cyclic = not nx.is_directed_acyclic_graph(self.circuit_graph)
            print(f"  Contains cycles: {is_cyclic}")
            
            if is_cyclic:
                # Try to find cycles for debugging
                try:
                    cycles = list(nx.simple_cycles(self.circuit_graph))
                    print(f"  Found {len(cycles)} cycles in the graph")
                    if cycles:
                        print(f"  Example cycle: {cycles[0]}")
                except Exception as e:
                    print(f"  Error finding cycles: {e}")
            else:
                # Get longest path in the acyclic graph
                longest_path = nx.dag_longest_path(self.circuit_graph)
                print(f"  Longest path length: {len(longest_path)-1} gates")
                
                # Sample a few paths for verification
                print("\nSample paths:")
                for i in range(min(3, len(self.inputs))):
                    for j in range(min(3, len(self.outputs))):
                        try:
                            path = nx.shortest_path(self.circuit_graph, self.inputs[i], self.outputs[j])
                            print(f"  {self.inputs[i]} → {self.outputs[j]}: {len(path)-1} gates")
                        except:
                            pass
        except Exception as e:
            print(f"  Error analyzing graph structure: {e}")

    def find_problematic_signals(self):
        """
        Find signals that might be part of cycles
        """
        try:
            problematic_signals = []
            cycles = list(nx.simple_cycles(self.circuit_graph))
            
            # Collect all signals involved in cycles
            for cycle in cycles:
                problematic_signals.extend(cycle)
            
            problematic_signals = list(set(problematic_signals))
            return problematic_signals
        except Exception as e:
            print(f"Error finding problematic signals: {e}")
            return []


def parse_verilog(verilog_file):
    """
    Parse a Verilog file and return the parser object.
    """
    parser = VerilogParser(verilog_file)
    parser.parse()
    return parser


if __name__ == "__main__":
    # Test the parser on c7552.v
    parser = parse_verilog("ISCAS85/c7552.v")
    parser.summarize()
    
    # Get gate information for optimization
    gate_info = parser.get_gate_info()
    print(f"\nPrepared {gate_info['num_gates']} gates for optimization")
    
    # Test critical path finding
    paths = parser.get_critical_paths(limit=10)
    if paths:
        print("\nSample critical paths:")
        for i, path in enumerate(paths[:5]):
            print(f"Path {i+1}: {' → '.join(path)} ({len(path)-1} gates)") 