import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def run_floorplanning_experiment():
    """
    This experiment compares 2D and 3D floorplanning optimization.
    It evaluates the trade-off between area and performance for different values of alpha.
    """
    list3d = []
    list2d = []
    
    alpha_values = [alpha/100 for alpha in range(1, 100, 2)]
    
    for alpha in alpha_values:
        p1 = 0  # 3D performance measure
        p2 = 0  # 2D performance measure
        print(f"percentage is: {alpha}")
        
        for i in range(100):        
            # Generate random parameters
            par = np.random.rand(3, 4)
            zpar = np.random.rand(4) * 0.2
            zcap = np.sum(zpar) * 3
            
            # 3D floorplanning optimization
            x = cp.Variable(4, pos=True)
            y = cp.Variable(4, pos=True)
            z = cp.Variable(4, pos=True)
            W = cp.Variable(pos=True)
            L = cp.Variable(pos=True)
            H = cp.Variable(pos=True)
            
            # Objective function: trade-off between area and performance
            objective_fn = alpha * W * L + (1-alpha) * (z[0]/(x[0] * y[0]) + z[1]/(x[1] * y[1]) + z[2]/(x[2] * y[2]) + x[3]/(z[3] * y[3]))
            
            # Constraints for 3D floorplanning
            constraints = [
                x[0] >= par[0][0], x[1] >= par[0][1], x[2] >= par[0][2], x[3] >= zpar[3],
                y[0] >= par[1][0], y[1] >= par[1][1], y[2] >= par[1][2], y[3] >= par[1][3],
                z[0] >= zpar[0], z[1] >= zpar[1], z[2] >= zpar[2], z[3] >= par[0][3],
                W >= x[0] + x[3], W >= x[1] + x[3], W >= x[2] + x[3],
                L >= y[0], L >= y[1], L >= y[2], L >= y[3], 
                H >= z[0] + z[1] + z[2], H >= z[3],
                H <= zcap
            ]
            
            problem = cp.Problem(cp.Minimize(objective_fn), constraints)
            problem.solve(gp=True)
            
            if not np.isinf(problem.value):
                p1 = p1 + problem.value

            # 2D floorplanning optimization
            x = cp.Variable(4, pos=True)
            y = cp.Variable(4, pos=True)
            W = cp.Variable(pos=True)
            L = cp.Variable(2, pos=True)
            
            # Objective function for 2D
            objective_fn2 = alpha * W * (L[0] + L[1]) + (1 - alpha) * (zpar[0]/(x[0] * y[0]) + zpar[1]/(x[1] * y[1]) + zpar[2]/(x[2] * y[2]) + zpar[3]/(x[3] * y[3]))
            
            # Constraints for 2D floorplanning
            constraints = [
                x[0] >= par[0][0], x[1] >= par[0][1], x[2] >= par[0][2], x[3] >= zpar[3],
                y[0] >= par[1][0], y[1] >= par[1][1], y[2] >= par[1][2], y[3] >= par[1][3],
                W >= x[0] + x[1], W >= x[2] + x[3],
                L[0] >= y[0], L[0] >= y[1], L[1] >= y[2], L[1] >= y[3]
            ]
            
            problem = cp.Problem(cp.Minimize(objective_fn2), constraints)
            problem.solve(gp=True)
            p2 = p2 + problem.value
            
        # Store average performance across 100 trials
        list3d.append(p1/100)
        list2d.append(p2/100)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, list3d, 'b-', label='3D floorplanning')
    plt.plot(alpha_values, list2d, 'r--', label='2D floorplanning')

    # Labels and title
    plt.xlabel('Alpha (area weight)')
    plt.ylabel('Objective Function Value')
    plt.title('2D vs 3D Floorplanning Performance')
    plt.legend()
    plt.grid(True)

    # Save the plot as a PDF
    plt.savefig("floorplanning_comparison.pdf", format="pdf")
    
    # Show the plot
    plt.show()
    
    return list3d, list2d, alpha_values

if __name__ == "__main__":
    run_floorplanning_experiment() 