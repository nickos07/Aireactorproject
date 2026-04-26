# Import required dependencies
import numpy as np
import argparse
import json
from pathlib import Path
from Reactor import Reactor
from ControlModule import ControlModule
from DemandGenerator import generate_demand
from Metrics import *
from Plotter import *

def get_args() -> tuple:
    # Define the parser object
    parser = argparse.ArgumentParser()

    # Define the expected arguments to parse and their data types
    parser.add_argument("--input-reactor", "-i", type=str, default="Reactors/R0.json", help="Path of the reactor's JSON file")
    parser.add_argument("--gamma", "-g", type=float, default=0.9, help="Discount factor used in the MDP")
    parser.add_argument("--random-seed", "-r", type=int, default=42, help="Pseudo-random number generator seed")
    parser.add_argument("--test-all", "-a", action="store_true", help="Test all reactors instead of single reactor")

    # Parse the arguments
    args = parser.parse_args()
    
    return args

def run_single_reactor(reactor_path: str, gamma: float, random_seed: int) -> None:
    """Run control loop for a single reactor with full visualization"""
    np.random.seed(random_seed)
    
    # Some verbose to check the correct parsing of the input arguments
    print(f"\nLoading reactor from file: {reactor_path}")
    print(f"Using gamma (discount factor): {gamma}")
    print(f"Using {random_seed} as random seed\n")

    # Build the Reactor object by reading the reactor's JSON file
    with open(reactor_path, 'r', encoding='utf-8') as file:
        json_data = json.load(fp=file)
        reactor   = Reactor(model=json_data['model'],
                            effective_section=float(json_data['effective_section']),
                            neutron_flux=float(json_data['neutron_flux']),
                            core_volume=float(json_data['core_volume']),
                            fision_energy=float(json_data['fision_energy']),
                            probabilities=dict(json_data['probabilities']))
        
    # Some verbose of the reactor loaded
    print(reactor)  # Overloaded in the __str__ method of Reactor's class

    # Get the probabilities from the reactor's dynamics
    probs = np.array([reactor.probabilities['decrease'], 
                      reactor.probabilities['maintain'], 
                      reactor.probabilities['increase']], dtype=np.float64)
    
    # Make a radar-plot with the reactor probabilities
    plot_reactor_as_radar(probs=probs)
    
    # Generate a random power demand
    demand = generate_demand(n_samples=512)

    # Define the number of MDP's states, actions and the discount factor (gamma)
    n_states  = 100
    n_actions = 3

    # Get the response time-series (answer to the demand time-series)
    response  = ControlModule.control_loop(demand=demand, 
                                           probs=probs,
                                           n_states=n_states,
                                           n_actions=n_actions,
                                           gamma=gamma)
    
    # Test print statement to verify control_loop is working
    print(f"✓ Control loop executed successfully!")
    print(f"  Response shape: {response.shape}, Min: {response.min():.4f}, Max: {response.max():.4f}, Mean: {response.mean():.4f}")
    
    # Calculate and print the four regression metrics for the current demand-response data
    _MAE  = MAE(y_true=demand, y_pred=response)
    _MSE  = MSE(y_true=demand, y_pred=response)
    _R2   = R2(y_true=demand, y_pred=response)
    _Corr = Corr(y_true=demand, y_pred=response)
    print(f"\n✓ Evaluation Metrics:")
    print(f"  MAE={_MAE:.6f}")
    print(f"  MSE={_MSE:.6f}")
    print(f"  R2={_R2:.6f}")
    print(f"  Corr={_Corr:.6f}\n")
    
    # Plot the original power demand
    plot_demand(demand=demand)

    # Plot the original power demand and the corresponding response
    plot_demand_response(demand=demand, response=response)

    # Plot the power response and the control bar percentaje employed
    plot_control_bars_usage(reactor=reactor, response=response)

    # Plot the correlation scatter-plot of both the demand and the response time-series
    plot_correlation(demand=demand, response=response)

    # Plot the MAE and the MSE in a bar-plot
    plot_mae_and_mse(MAE=_MAE, MSE=_MSE)

    # Plot the R2 and the Corr in a bar-plot
    plot_r2_and_pearson(R2=_R2, Pearson=_Corr)

def run_test_all_reactors(gamma: float, random_seed: int) -> None:
    """Test all reactors and generate comparison results"""
    # Generate demand curve once (same for all tests)
    np.random.seed(random_seed)
    demand = generate_demand(n_samples=512)
    
    # Find all reactor JSON files
    reactor_dir = Path('Reactors')
    reactor_files = sorted(reactor_dir.glob('R*.json'))
    
    print("\n" + "=" * 80)
    print("REACTOR CONTROL SYSTEM - COMPREHENSIVE EXPERIMENTATION")
    print("=" * 80)
    print(f"Testing {len(reactor_files)} reactors with {len(demand)} demand points\n")
    
    results = []
    
    for reactor_file in reactor_files:
        print(f"Testing {reactor_file.name}...", end=" ", flush=True)
        try:
            np.random.seed(random_seed)
            
            # Load reactor data
            with open(reactor_file, 'r', encoding='utf-8') as file:
                reactor_data = json.load(file)
            
            # Extract probabilities as array
            probs = np.array([
                reactor_data['probabilities']['decrease'],
                reactor_data['probabilities']['maintain'],
                reactor_data['probabilities']['increase']
            ], dtype=np.float64)
            
            # Run control loop
            response = ControlModule.control_loop(demand, probs, 100, 3, gamma)
            
            # Calculate metrics
            mae = MAE(demand, response)
            mse = MSE(demand, response)
            r2 = R2(demand, response)
            corr = Corr(demand, response)
            
            results.append({
                'model': reactor_data['model'],
                'path': str(reactor_file),
                'MAE': mae,
                'MSE': mse,
                'R2': r2,
                'Corr': corr
            })
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Print comprehensive results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<30} {'MAE':<12} {'MSE':<12} {'R2':<12} {'Corr':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['model']:<30} "
              f"{result['MAE']:<12.6f} "
              f"{result['MSE']:<12.6f} "
              f"{result['R2']:<12.6f} "
              f"{result['Corr']:<12.6f}")
    
    print("-" * 80)
    
    # Find best and worst performers
    best_r2 = max(results, key=lambda x: x['R2'])
    worst_r2 = min(results, key=lambda x: x['R2'])
    best_mae = min(results, key=lambda x: x['MAE'])
    best_corr = max(results, key=lambda x: x['Corr'])
    
    print(f"\nBest R² Score:       {best_r2['model']:<25} ({best_r2['R2']:.6f})")
    print(f"Worst R² Score:      {worst_r2['model']:<25} ({worst_r2['R2']:.6f})")
    print(f"Best MAE:            {best_mae['model']:<25} ({best_mae['MAE']:.6f})")
    print(f"Best Correlation:    {best_corr['model']:<25} ({best_corr['Corr']:.6f})")
    print("=" * 80 + "\n")

def main() -> None:
    # Parse the main arguments
    args = get_args()

    # Set the random seed
    np.random.seed(args.random_seed)
    
    # Run either single reactor or all reactors test
    if args.test_all:
        run_test_all_reactors(args.gamma, args.random_seed)
    else:
        run_single_reactor(args.input_reactor, args.gamma, args.random_seed)

if __name__ == '__main__':
    main()
