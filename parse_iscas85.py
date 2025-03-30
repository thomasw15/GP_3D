import re
import networkx as nx

def parse_iscas85_verilog(filename):
    """
    Parse an ISCAS85 Verilog benchmark file and extract the circuit topology.
    
    Returns:
        G: networkx.DiGraph representing the circuit
        primary_inputs: list of primary input nodes
        primary_outputs: list of primary output nodes
    """
    with open(filename, 'r') as f:
        content = f.read()
    
    # Create directed graph to represent the circuit
    G = nx.DiGraph()
    
    # Extract module name
    module_match = re.search(r'module\s+(\w+)', content)
    if not module_match:
        raise ValueError(f"Could not find module name in {filename}")
    
    module_name = module_match.group(1)
    print(f"Module name: {module_name}")
    
    # Extract primary inputs
    input_match = re.search(r'input\s+(.*?);', content, re.DOTALL)
    if not input_match:
        raise ValueError(f"Could not find inputs in {filename}")
    
    input_str = input_match.group(1).replace('\n', ' ').replace('\r', '')
    primary_inputs = [x.strip() for x in input_str.split(',')]
    primary_inputs = [x for x in primary_inputs if x]  # Remove empty strings
    
    # Add primary inputs to graph
    for node in primary_inputs:
        G.add_node(node, type='input')
    
    # Extract primary outputs
    output_match = re.search(r'output\s+(.*?);', content, re.DOTALL)
    if not output_match:
        raise ValueError(f"Could not find outputs in {filename}")
    
    output_str = output_match.group(1).replace('\n', ' ').replace('\r', '')
    primary_outputs = [x.strip() for x in output_str.split(',')]
    primary_outputs = [x for x in primary_outputs if x]  # Remove empty strings
    
    # Add primary outputs to graph
    for node in primary_outputs:
        G.add_node(node, type='output')
    
    # Find all wire declarations
    wire_match = re.search(r'wire\s+(.*?);', content, re.DOTALL)
    wires = []
    if wire_match:
        wire_str = wire_match.group(1).replace('\n', ' ').replace('\r', '')
        wires = [x.strip() for x in wire_str.split(',')]
        wires = [x for x in wires if x]  # Remove empty strings
    
    # Add wires to graph
    for wire in wires:
        G.add_node(wire, type='wire')
    
    # Extract NAND and other gate instances for ISCAS85
    # Format is typically: nand NAND2_1 (output, input1, input2);
    gate_pattern = r'(\w+)\s+(\w+)\s*\(\s*([^,\s]+)\s*,\s*([^,\s]+)\s*(?:,\s*([^,\s]+)\s*)?(?:,\s*([^,\s]+)\s*)?(?:,\s*([^,\s]+)\s*)?\);'
    gate_matches = re.finditer(gate_pattern, content)
    
    for match in gate_matches:
        gate_type = match.group(1).lower()  # e.g., 'nand'
        gate_name = match.group(2)  # e.g., 'NAND2_1'
        
        # First argument is the output
        output_node = match.group(3).strip()
        
        # Remaining arguments are inputs
        inputs = []
        for i in range(4, 8):  # Up to 4 inputs (typical for ISCAS85)
            if match.group(i):
                inputs.append(match.group(i).strip())
        
        # Add gate to graph
        G.add_node(gate_name, type=gate_type)
        
        # Add edges (inputs -> gate -> output)
        for input_node in inputs:
            G.add_edge(input_node, gate_name)  # Input to gate
        
        G.add_edge(gate_name, output_node)  # Gate to output
    
    print(f"Extracted circuit with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G, primary_inputs, primary_outputs


def get_paths(G, primary_inputs, primary_outputs):
    """
    Find all paths from primary inputs to primary outputs.
    
    Args:
        G: Circuit graph
        primary_inputs: List of input nodes
        primary_outputs: List of output nodes
        
    Returns:
        List of paths (each path is a list of nodes)
    """
    all_paths = []
    
    for source in primary_inputs:
        for target in primary_outputs:
            try:
                paths = list(nx.all_simple_paths(G, source, target))
                all_paths.extend(paths)
            except nx.NetworkXNoPath:
                # No path between this input-output pair
                continue
    
    return all_paths


if __name__ == "__main__":
    # Test the parser with a small benchmark
    import sys
    
    if len(sys.argv) > 1:
        benchmark = sys.argv[1]
    else:
        benchmark = "ISCAS85/c17.v"
    
    G, inputs, outputs = parse_iscas85_verilog(benchmark)
    print(f"Primary inputs: {inputs}")
    print(f"Primary outputs: {outputs}")
    
    # Print some basic statistics
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")
    
    # Count node types
    node_types = {}
    for node, attrs in G.nodes(data=True):
        node_type = attrs.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print("Node types:")
    for node_type, count in node_types.items():
        print(f"  {node_type}: {count}") 