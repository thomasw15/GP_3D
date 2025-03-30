# Geometric Programming for Circuit Design Optimization

This repository contains the implementation of geometric programming (GP) approaches for various circuit design optimization problems as presented in our paper. The code generates the experimental results and visualizations used in the publication.

## Overview

The scripts in this repository implement GP-based optimization for:
- 2D/3D circuit floorplanning with thermal considerations
- Transistor sizing with delay-area tradeoffs
- Gate sizing for ISCAS benchmark circuits
- Interconnect sizing with optimal width determination

## Key Files and Functionality

### Floorplanning Optimization

- **floorplanning.py**: Generates plots comparing 2D and 3D floorplanning optimization for an artificially constructed 4-module arrangement.

- **floorplanning_functions.py**: Core implementation of 3D floorplanning optimization with thermal consideration.

- **floorplanning_temperature.py**: Produces visualizations showing the impact of floorplanning on module temperature control.

### Transistor and Gate Sizing

- **transistor_sizing.py**: Implements transistor sizing optimization and plots volume vs. delay trade-off curves.

- **ISCAS_gate_sizing.py**: Optimizes gate sizing for ISCAS benchmark circuits and analyzes delay-volume tradeoffs.

- **run_c17_gate_sizes_plot.py**: Generates detailed plots comparing individual gate sizes for the C17 benchmark circuit as volume constraints change.

### Interconnect Sizing

- **interconnect_sizing.py**: Implements interconnect sizing optimization for constructed interconnect networks and visualizes width-delay relationships.

- **interconnect_iscas.py**: Applies interconnect sizing optimization to the C17 circuit from ISCAS-85 benchmarks.

## Setup and Dependencies

This project requires the following Python packages:
- CVXPY (with GP support)
- NumPy
- Matplotlib
- pandas
- tqdm
- concurrent.futures

You can install the required packages using:

```bash
pip install cvxpy numpy matplotlib pandas tqdm
```

## Running the Experiments

Each script can be executed independently to reproduce specific figures from the paper:

```bash
# Run floorplanning experiments
python floorplanning.py

# Run temperature-aware floorplanning experiments
python floorplanning_temperature.py --num_experiments 5 --num_modules 150 --alpha 0.6

# Run transistor sizing experiments
python transistor_sizing.py

# Run gate sizing for C17 benchmark
python run_c17_gate_sizes_plot.py

# Run interconnect sizing experiments
python interconnect_sizing.py

# Run interconnect sizing for ISCAS benchmarks
python interconnect_iscas.py
```

The generated plots and data will be saved in the `results/` directory by default.

## Citation

If you use this code in your research, please cite our paper:

```
[Citation information will be provided upon publication. The paper is currently in preparation.]
```

## References

In this work, we use the following benchmark circuits and process design kits:

1. **ISCAS-85 Benchmark Circuits**:  
   F. Brglez, D. Bryan, and K. Kozminski, "Combinational profiles of sequential benchmark circuits," in *1989 IEEE International Symposium on Circuits and Systems (ISCAS)*, 1989, pp. 1929-1934 vol.3, doi: [10.1109/ISCAS.1989.100747](https://doi.org/10.1109/ISCAS.1989.100747).

2. **ASAP7 Process Design Kit**:  
   L. T. Clark, V. Vashishtha, L. Shifren, A. Gujja, S. Sinha, B. Cline, C. Ramamurthy, and G. Yeric, "ASAP7: A 7-nm finFET predictive process design kit," *Microelectronics Journal*, vol. 53, pp. 105-115, 2016, doi: [10.1016/j.mejo.2016.04.006](https://doi.org/10.1016/j.mejo.2016.04.006).

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details. 