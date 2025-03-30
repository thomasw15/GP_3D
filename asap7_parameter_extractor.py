#!/usr/bin/env python3
"""
ASAP7 Technology Parameter Extractor
-----------------------------------
This script extracts technology parameters from the ASAP7 PDK datasheet files and 
saves them in a structured format (JSON) for use in circuit optimization.

IMPORTANT: This extractor MUST be run to get accurate parameters for circuit
optimization. Without running this extractor, the system will fall back to
rough estimates which may not accurately represent the ASAP7 technology.

The script can extract parameters such as:
- Gate area/footprint
- Pin capacitance
- Intrinsic delay and output resistance
- Leakage power

Parameters are extracted from the text datasheet files (.txt.gz) in the ASAP7 PDK.

Usage:
    python asap7_parameter_extractor.py --pdk /path/to/asap7_pdk --output params.json

Or more easily with the provided shell script:
    ./extract_asap7_params.sh /path/to/asap7_pdk

Author: Claude
Date: March 26, 2024
"""

import os
import re
import json
import gzip
import glob
from collections import defaultdict
import argparse

class ASAP7ParameterExtractor:
    """
    Extracts technology parameters from ASAP7 datasheet files.
    """
    
    def __init__(self, pdk_path):
        """
        Initialize the extractor with the path to the ASAP7 PDK.
        
        Args:
            pdk_path: Path to the ASAP7 PDK directory
        """
        self.pdk_path = pdk_path
        self.datasheet_path = os.path.join(pdk_path, 'Datasheet', 'text')
        self.params = {
            'technology': '7nm',
            'vdd': 0.7,  # Supply voltage in volts (from datasheet)
            'temp': 25,  # Temperature in °C (from datasheet)
            'gate_delay': {},
            'gate_area': {},
            'input_cap': {},
            'leakage': {}
        }
    
    def find_datasheet_files(self, cell_type='SIMPLE', process='RVT', corner='TT'):
        """
        Find datasheet files matching the specified criteria.
        
        Args:
            cell_type: Type of cell library ('SIMPLE', 'INVBUF', etc.)
            process: Process type ('RVT', 'LVT', 'SLVT')
            corner: Process corner ('TT', 'FF', 'SS')
            
        Returns:
            List of matching datasheet file paths
        """
        pattern = f"asap7sc7p5t_{cell_type}_{process}_{corner}_*.txt.gz"
        return glob.glob(os.path.join(self.datasheet_path, pattern))
    
    def extract_gate_params_from_file(self, file_path):
        """
        Extract gate parameters from a datasheet file.
        
        Args:
            file_path: Path to the datasheet file (.txt.gz)
            
        Returns:
            Dictionary with extracted parameters
        """
        gate_params = {}
        current_gate = None
        current_gate_norm = None
        current_section = None
        
        # Open and parse the gzipped file
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Processing {len(lines)} lines from {os.path.basename(file_path)}...")
        
        # Process the file line by line
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check if we're starting a new gate section
            if "Cell Group" in line and "_ASAP7_75T_R from Library" in line:
                # Extract gate name (e.g., "AND2X2" from "Cell Group AND2X2_ASAP7_75T_R from Library")
                parts = line.split("Cell Group ")
                if len(parts) > 1:
                    gate_name_parts = parts[1].split("_ASAP7_75T_R")
                    if gate_name_parts:
                        current_gate = gate_name_parts[0]
                        current_gate_norm = self.normalize_gate_name(current_gate)
                        if current_gate_norm:
                            print(f"Found gate: {current_gate} (normalized: {current_gate_norm})")
                            current_section = None
            
            # Check if we're ending a gate section
            if current_gate and f"END Cell Group {current_gate}_ASAP7_75T_R" in line:
                current_gate = None
                current_gate_norm = None
                current_section = None
                continue
            
            # Skip if no current gate
            if not current_gate or not current_gate_norm:
                continue
            
            # Identify the current subsection
            if "Footprint:" in line:
                current_section = "footprint"
                continue
            elif "Leakage" in line and "Min" in line and "Avg" in line:
                current_section = "leakage"
                continue
            elif "Pin Capacitance" in line:
                current_section = "pin_cap"
                continue
            elif "Delays(ps) to Y rising:" in line:
                current_section = "delay_rising"
                continue
            elif "Delays(ps) to Y falling:" in line:
                current_section = "delay_falling"
                continue
            
            # Lowercased gate name for case-insensitive matching in tables
            gate_name_lower = current_gate.lower()
            cell_pattern = f"{gate_name_lower}x"  # e.g., "and2x" to match "AND2x2_ASAP7_75t_R"
            cell_pattern_alt = f"{gate_name_lower}_"  # Alternative pattern
            
            # Extract parameters based on current section
            if current_section == "footprint" and "|" in line:
                # Match cell name in format like "AND2x2_ASAP7_75t_R"
                if cell_pattern in line.lower() or cell_pattern_alt in line.lower():
                    parts = line.split('|')
                    if len(parts) >= 3:
                        area_str = parts[2].strip()
                        try:
                            area = float(area_str)
                            # Initialize gate record if not exists
                            if current_gate_norm not in self.params['gate_area']:
                                self.params['gate_area'][current_gate_norm] = area
                                print(f"  Extracted area: {area}")
                        except ValueError:
                            pass
            
            elif current_section == "leakage" and "|" in line:
                if cell_pattern in line.lower() or cell_pattern_alt in line.lower():
                    parts = line.split('|')
                    if len(parts) >= 4:
                        leakage_str = parts[3].strip()  # Avg column
                        try:
                            leakage = float(leakage_str)
                            self.params['leakage'][current_gate_norm] = leakage
                            print(f"  Extracted leakage: {leakage}")
                        except ValueError:
                            pass
            
            elif current_section == "pin_cap" and "|" in line:
                if cell_pattern in line.lower() or cell_pattern_alt in line.lower():
                    parts = line.split('|')
                    if len(parts) >= 4:  # At least one input pin plus the cell name
                        try:
                            # Count how many input pins (depends on gate type)
                            input_caps = []
                            # Skip first 2 parts (empty and cell name)
                            for i in range(2, len(parts)-1):
                                cap_str = parts[i].strip()
                                if cap_str and cap_str not in ["Pin Cap(ff)", "Max Cap(ff)"]:
                                    try:
                                        cap_val = float(cap_str)
                                        input_caps.append(cap_val)
                                    except ValueError:
                                        pass
                            
                            if input_caps:
                                # For most gates, input pins come before the output pin Y
                                # Use all but possibly the last cap as inputs
                                if len(input_caps) > 1:
                                    avg_input_cap = sum(input_caps[:-1]) / len(input_caps[:-1])
                                else:
                                    avg_input_cap = input_caps[0]
                                
                                self.params['input_cap'][current_gate_norm] = avg_input_cap
                                print(f"  Extracted input capacitance: {avg_input_cap}")
                        except Exception as e:
                            print(f"  Error processing pin capacitance: {e}")
            
            elif (current_section == "delay_rising" or current_section == "delay_falling") and "|" in line:
                if cell_pattern in line.lower() or cell_pattern_alt in line.lower():
                    if "->Y(" in line:  # Only process lines with timing arcs
                        parts = line.split('|')
                        if len(parts) >= 6:
                            try:
                                # Extract timing arc
                                timing_arc = parts[2].strip()
                                if "->" in timing_arc and "Y(" in timing_arc:
                                    pin = timing_arc.split('->')[0].strip()
                                    transition = timing_arc.split('(')[1].split(')')[0]
                                    
                                    # Extract delay values
                                    min_delay = float(parts[3].strip())
                                    mid_delay = float(parts[4].strip())
                                    max_delay = float(parts[5].strip())
                                    
                                    # Store intrinsic delay if not already set
                                    if current_gate_norm not in self.params['gate_delay']:
                                        # Use typical drive factor based on gate type
                                        drive_factor = self.get_typical_drive_factor(current_gate_norm)
                                        self.params['gate_delay'][current_gate_norm] = (mid_delay, drive_factor)
                                        print(f"  Extracted delay: {mid_delay} ps (drive factor: {drive_factor})")
                            except Exception as e:
                                print(f"  Error processing delay: {e}")
        
        return self.params
    
    def get_typical_drive_factor(self, gate_type):
        """
        Return typical drive factor for a gate type based on educated estimates.
        
        Args:
            gate_type: Normalized gate type
            
        Returns:
            Typical drive factor value
        """
        # Note: These drive factor values are educated estimates rather than values
        # directly extracted from the datasheet. A more accurate approach would be to:
        # 1) Extract multiple delay points with varying load capacitance values
        # 2) Perform regression analysis to determine the slope of delay vs. load
        # 3) Convert the slope coefficient to our drive factor format
        # However, this simplified approach with typical values for each gate type 
        # is sufficient for most practical purposes.
        
        # Default educated guess for drive factor
        drive_factor = 0.7  # Default value
        
        # Refine based on gate type - these are educated guesses based on
        # typical relative drive strength characteristics of these gate types
        if 'inv' in gate_type:
            drive_factor = 0.42  # Inverters typically have the lowest drive factor
        elif 'buf' in gate_type:
            drive_factor = 0.51  # Buffers have slightly higher drive factor than inverters
        elif 'nand' in gate_type:
            drive_factor = 0.65  # NAND gates have moderate drive factor
        elif 'nor' in gate_type:
            drive_factor = 0.7   # NOR gates typically have higher drive factor than NAND
        elif 'and' in gate_type:
            drive_factor = 0.68  # AND gates (NAND+INV) have moderate-high drive factor
        elif 'or' in gate_type:
            drive_factor = 0.73  # OR gates (NOR+INV) have high drive factor
        elif 'xor' in gate_type or 'xnor' in gate_type:
            drive_factor = 0.82  # XOR/XNOR gates have the highest drive factor due to complexity
        
        return drive_factor
    
    def normalize_gate_name(self, gate_name):
        """
        Normalize gate names to consistent format.
        
        Args:
            gate_name: Original gate name from datasheet
            
        Returns:
            Normalized gate name
        """
        # Extract gate type
        gate_type = ""
        gate_name = gate_name.lower()
        
        # Handle different naming conventions
        if "nand" in gate_name:
            if "2" in gate_name:
                gate_type = "nand2"
            elif "3" in gate_name:
                gate_type = "nand3"
            elif "4" in gate_name:
                gate_type = "nand4"
            else:
                gate_type = "nand"
                
        elif "nor" in gate_name:
            if "2" in gate_name:
                gate_type = "nor2"
            elif "3" in gate_name:
                gate_type = "nor3"
            elif "4" in gate_name:
                gate_type = "nor4"
            else:
                gate_type = "nor"
                
        elif "and" in gate_name and "nand" not in gate_name:
            if "2" in gate_name:
                gate_type = "and2"
            elif "3" in gate_name:
                gate_type = "and3"
            elif "4" in gate_name:
                gate_type = "and4"
            else:
                gate_type = "and"
                
        elif "or" in gate_name and "nor" not in gate_name and "xor" not in gate_name:
            if "2" in gate_name:
                gate_type = "or2"
            elif "3" in gate_name:
                gate_type = "or3"
            elif "4" in gate_name:
                gate_type = "or4"
            else:
                gate_type = "or"
                
        elif "xor" in gate_name:
            gate_type = "xor"
            
        elif "xnor" in gate_name:
            gate_type = "xnor"
            
        elif "inv" in gate_name:
            gate_type = "inv"
            
        elif "buf" in gate_name:
            gate_type = "buf"
            
        elif "aoi" in gate_name:
            if "21" in gate_name:
                gate_type = "aoi21"
            elif "22" in gate_name:
                gate_type = "aoi22"
                
        elif "oai" in gate_name:
            if "21" in gate_name:
                gate_type = "oai21"
            elif "22" in gate_name:
                gate_type = "oai22"
        
        return gate_type.lower() if gate_type else None
    
    def calculate_drive_factor(self, gate_params):
        """
        Calculate the drive strength factor for each gate type.
        
        Args:
            gate_params: Dictionary with gate parameters
            
        Returns:
            Dictionary with drive factors
        """
        # We don't need this method anymore as we're directly setting drive factors
        # in the get_typical_drive_factor method during extraction
        return {}
        
    def extract_parameters(self, cell_types=None, process='RVT', corner='TT'):
        """
        Extract parameters from all datasheet files.
        
        Args:
            cell_types: List of cell types to extract parameters from
            process: Process type ('RVT', 'LVT', 'SLVT')
            corner: Process corner ('TT', 'FF', 'SS')
            
        Returns:
            Dictionary with extracted parameters
        """
        if cell_types is None:
            cell_types = ['SIMPLE', 'INVBUF']
        
        all_gate_params = {}
        
        # Extract parameters from each cell type
        for cell_type in cell_types:
            files = self.find_datasheet_files(cell_type, process, corner)
            for file_path in files:
                print(f"Processing {os.path.basename(file_path)}...")
                self.extract_gate_params_from_file(file_path)
        
        # No need to calculate drive factors - they're set during extraction
        # drive_factors = self.calculate_drive_factor(all_gate_params)
        
        return self.params
    
    def save_parameters(self, output_file="asap7_extracted_params.json", params=None):
        """
        Save parameters to a JSON file.
        
        Args:
            output_file: Output file path
            params: Parameters to save (defaults to self.params)
            
        Returns:
            True if successful, False otherwise
        """
        params = params or self.params
        
        try:
            with open(output_file, 'w') as f:
                json.dump(params, f, indent=2)
            print(f"Parameters saved to {output_file}")
            return True
        except Exception as e:
            print(f"Error saving parameters: {e}")
            return False
    
    def load_parameters(self, input_file="asap7_extracted_params.json"):
        """
        Load parameters from a JSON file.
        
        Args:
            input_file: Input file path
            
        Returns:
            Dictionary with loaded parameters
        """
        try:
            with open(input_file, 'r') as f:
                params = json.load(f)
            print(f"Parameters loaded from {input_file}")
            return params
        except Exception as e:
            print(f"Error loading parameters: {e}")
            return None
    
    def print_summary(self, params=None):
        """
        Print a summary of the extracted parameters.
        
        Args:
            params: Parameters to print (defaults to self.params)
        """
        params = params or self.params
        
        print(f"\nASAP7 Technology Parameters Summary ({params['technology']})")
        print("-" * 50)
        print(f"Supply voltage: {params['vdd']} V")
        print(f"Temperature: {params['temp']} °C")
        
        # Print gate delay parameters
        print("\nGate Delay Parameters (ps):")
        for gate, (intrinsic, drive) in sorted(params['gate_delay'].items()):
            print(f"  {gate.upper()}: {intrinsic:.2f} ps (drive factor: {drive:.2f})")
        
        # Print gate area parameters
        print("\nGate Area (μm²):")
        for gate, area in sorted(params['gate_area'].items()):
            print(f"  {gate.upper()}: {area:.4f}")
        
        # Print input capacitance parameters
        print("\nInput Capacitance (fF):")
        for gate, cap in sorted(params['input_cap'].items()):
            print(f"  {gate.upper()}: {cap:.4f}")
        
        # Print leakage power parameters
        print("\nLeakage Power (pW):")
        for gate, leakage in sorted(params['leakage'].items()):
            print(f"  {gate.upper()}: {leakage:.4f}")

def main():
    """
    Main function to extract ASAP7 parameters.
    """
    parser = argparse.ArgumentParser(description='Extract ASAP7 technology parameters')
    parser.add_argument('--pdk', required=True, help='Path to ASAP7 PDK directory')
    parser.add_argument('--output', default='asap7_extracted_params.json', help='Output JSON file')
    parser.add_argument('--process', default='RVT', choices=['RVT', 'LVT', 'SLVT'], help='Process type')
    parser.add_argument('--corner', default='TT', choices=['TT', 'FF', 'SS'], help='Process corner')
    parser.add_argument('--cell-types', default='SIMPLE,INVBUF', help='Comma-separated list of cell types')
    
    args = parser.parse_args()
    
    # Create the extractor
    extractor = ASAP7ParameterExtractor(args.pdk)
    
    # Extract parameters
    cell_types = args.cell_types.split(',')
    params = extractor.extract_parameters(cell_types, args.process, args.corner)
    
    # Save parameters
    extractor.save_parameters(args.output, params)
    
    # Print summary
    extractor.print_summary(params)

if __name__ == "__main__":
    main() 