import json
import csv
import os
from datetime import datetime
from pathlib import Path


class MaxCutExperimentLogger:
    """
    Manages experiment metadata, results, and output organization
    for the Hopfield-Tank Max-Cut solver.
    """

    def __init__(self, base_output_dir='experiments_maxcut'):
        self.base_output_dir = base_output_dir
        self.experiment_dir: str | None = None
        self.metadata = {}

    def start_experiment(self, net, args, dataset_name=None):
        """
        Create a timestamped experiment directory and save metadata.

        Parameters
        ----------
        net          : HopfieldNetMaxCut instance
        args         : argparse.Namespace
        dataset_name : str, optional
        """
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = (dataset_name
                        or (Path(args.data).stem if args.data else 'random_graph'))
        exp_name     = f"{dataset_name}_{timestamp}"

        self.experiment_dir = os.path.join(self.base_output_dir, exp_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.metadata = {
            'experiment_name': exp_name,
            'timestamp':       timestamp,
            'dataset': {
                'file':    args.data if args.data else 'random',
                'name':    dataset_name,
                'n_nodes': net.n,
            },
            'network_parameters': {
                'u0':       net.u0,
                'tau':      net.tau,
                'timestep': net.timestep,
                'initialisation method' : net.init_mode
            },
            'simulation_settings': {
                'steps':              args.steps,
                'seed':               args.seed,
                'snapshot_frequency': args.freq,
            }
        }
        self._save_metadata()
        print(f"\n📁 Experiment directory: {self.experiment_dir}")
        return self.experiment_dir

    def _save_metadata(self):
        assert self.experiment_dir is not None, "Call start_experiment() first"
        with open(os.path.join(self.experiment_dir, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def log_results(self, partition, cut_value, final_energy):
        """
        Save final results to metadata and write RESULTS.txt.

        Parameters
        ----------
        partition    : list of ±1 integers (one per node)
        cut_value    : float, binary cut weight after sign() binarisation
        final_energy : float, final Lyapunov energy value
        """
        assert self.experiment_dir is not None, "Call start_experiment() first"

        set_A = [i for i, s in enumerate(partition) if s  ==  1]
        set_B = [i for i, s in enumerate(partition) if s  == -1]

        self.metadata['results'] = {
            'partition':    partition,
            'set_A':        set_A,
            'set_B':        set_B,
            'cut_value':    cut_value,
            'final_energy': final_energy,
        }
        self._save_metadata()

        with open(os.path.join(self.experiment_dir, 'RESULTS.txt'), 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("EXPERIMENT RESULTS — Hopfield-Tank Max-Cut\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Dataset  : {self.metadata['dataset']['name']}\n")
            f.write(f"Nodes    : {self.metadata['dataset']['n_nodes']}\n\n")
            f.write("Network Parameters:\n")
            for k, v in self.metadata['network_parameters'].items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\nBinary cut value : {cut_value:.6f}\n")
            f.write(f"Final energy     : {final_energy:.6f}\n")
            f.write(f"\nSet A (s = +1) : {set_A}\n")
            f.write(f"Set B (s = -1) : {set_B}\n")

    def get_output_subdir(self, subdir_name):
        """Get (and create) a subdirectory within the experiment folder."""
        assert self.experiment_dir is not None, "Call start_experiment() first"
        path = os.path.join(self.experiment_dir, subdir_name)
        os.makedirs(path, exist_ok=True)
        return path

    def append_to_comparison_csv(self):
        """
        Append this experiment to a global CSV for easy cross-run comparison.
        Mirrors TSP append_to_comparison_csv().
        """
        assert self.experiment_dir is not None, "Call start_experiment() first"
        csv_path    = os.path.join(self.base_output_dir, 'all_experiments.csv')
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'dataset', 'n_nodes', 'u0',
                                 'steps', 'cut_value', 'energy'])
            r = self.metadata
            writer.writerow([
                r['timestamp'],
                r['dataset']['name'],
                r['dataset']['n_nodes'],
                r['network_parameters']['u0'],
                r['simulation_settings']['steps'],
                r['results']['cut_value'],
                r['results']['final_energy'],
            ])
