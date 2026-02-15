import json
import os
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    """
    Manages experiment metadata, results, and output organization.
    Each experiment run gets its own timestamped directory.
    """
    
    def __init__(self, base_output_dir='experiments'):
        self.base_output_dir = base_output_dir
        self.experiment_dir: str | None = None
        self.metadata = {}
        
    def start_experiment(self, net, args, dataset_name=None):
        """
        Create a new experiment directory and save metadata.
        
        Parameters
        ----------
        net : HopfieldNet
            The network instance (to extract A, B, C, D, u0/alpha)
        args : Namespace
            Command-line arguments
        dataset_name : str, optional
            Name of the dataset file (extracted from args.data if None)
        """
        # Create timestamped experiment folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = dataset_name or Path(args.data).stem
        
        exp_name = f"{dataset_name}_{timestamp}"
        self.experiment_dir = os.path.join(self.base_output_dir, exp_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Gather metadata
        self.metadata = {
            'experiment_name': exp_name,
            'timestamp': timestamp,
            'dataset': {
                'file': args.data,
                'name': dataset_name,
                'n_cities': net.size
            },
            'network_parameters': {
                'A': net.a,
                'B': net.b,
                'C': net.c,
                'D': net.d,
                'u0_alpha': getattr(net, 'alpha', getattr(net, 'u0', None)),
                'sigma': getattr(net, 'sigma', getattr(net, 'size_adj', 0)),
                'timestep': net.timestep
            },
            'simulation_settings': {
                'steps': args.steps,
                'seed': args.seed,
                'snapshot_frequency': args.freq
            }
        }
        
        # Save metadata immediately
        self._save_metadata()
        
        print(f"\n📁 Experiment directory: {self.experiment_dir}")
        return self.experiment_dir
    
    def _save_metadata(self):
        """Write metadata to JSON file."""
        assert self.experiment_dir is not None, "Call start_experiment() first"
        metadata_path = os.path.join(self.experiment_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def log_results(self, tour, tour_distance, final_energy, is_valid):
        """
        Save final results to metadata.
        
        Parameters
        ----------
        tour : list
            Decoded tour (city indices)
        tour_distance : float
            Total tour distance
        final_energy : float
            Final energy value
        is_valid : bool
            Whether tour is a valid Hamiltonian cycle
        """
        assert self.experiment_dir is not None, "Call start_experiment() first"
        
        self.metadata['results'] = {
            'tour': tour,
            'tour_distance': tour_distance,
            'final_energy': final_energy,
            'is_valid_tour': is_valid
        }
        self._save_metadata()
        
        # Also write a human-readable summary
        summary_path = os.path.join(self.experiment_dir, 'RESULTS.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("EXPERIMENT RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Dataset: {self.metadata['dataset']['name']}\n")
            f.write(f"Cities: {self.metadata['dataset']['n_cities']}\n\n")
            f.write("Network Parameters:\n")
            for k, v in self.metadata['network_parameters'].items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\nTour distance: {tour_distance:.6f}\n")
            f.write(f"Final energy: {final_energy:.6f}\n")
            f.write(f"Valid tour: {'✓ YES' if is_valid else '✗ NO'}\n")
            f.write(f"\nTour: {tour}\n")
    
    def get_output_subdir(self, subdir_name):
        """Get path to a subdirectory within the experiment folder."""
        assert self.experiment_dir is not None, "Call start_experiment() first"
        path = os.path.join(self.experiment_dir, subdir_name)
        os.makedirs(path, exist_ok=True)
        return path
    
    def append_to_comparison_csv(self):
        """Append this experiment to a global CSV for easy comparison."""
        assert self.experiment_dir is not None, "Call start_experiment() first"
        
        csv_path = os.path.join(self.base_output_dir, 'all_experiments.csv')
        
        import csv
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'dataset', 'A', 'B', 'C', 'D', 'u0', 
                               'steps', 'distance', 'energy', 'valid'])
            
            writer.writerow([
                self.metadata['timestamp'],
                self.metadata['dataset']['name'],
                self.metadata['network_parameters']['A'],
                self.metadata['network_parameters']['B'],
                self.metadata['network_parameters']['C'],
                self.metadata['network_parameters']['D'],
                self.metadata['network_parameters']['u0_alpha'],
                self.metadata['simulation_settings']['steps'],
                self.metadata['results']['tour_distance'],
                self.metadata['results']['final_energy'],
                self.metadata['results']['is_valid_tour']
            ])
