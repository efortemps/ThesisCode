import json
import csv
import os
from datetime import datetime
from pathlib import Path


class MaxCutExperimentLogger:
    """
    Manages experiment metadata, results, and output organisation
    for the OIM Max-Cut solver (mu-parametrisation).

    Directory structure created under base_output_dir
    --------------------------------------------------
    <base_output_dir>/
      <dataset>_<timestamp>/          # single-mu run  (start_experiment)
        metadata.json
        RESULTS.txt
        runs.csv
        plots/
          experiment_maxcut_phases.png
          experiment_maxcut_thresholds.png

      mu_sweep_<dataset>_<timestamp>/ # interactive / sweep  (start_mu_sweep_experiment)
        metadata.json
        mu_sweep_results.csv
        state_<timestamp>.json        # snapshot saved on each slider change

    Public API
    ----------
    Single-mu run
        start_experiment(net, args, dataset_name)
        log_results(partition, cut_value, final_energy)
        log_all_runs(records)
        save_plot(src_path, subfolder="plots")

    Interactive / sweep
        start_mu_sweep_experiment(args, dataset_name, n_nodes)
        log_mu_sweep_results(results)
        save_interactive_state(state_dict)
    """

    def __init__(self, base_output_dir: str = "experiments_MaxCut_OIM"):
        self.base_output_dir = base_output_dir
        self.experiment_dir: str | None = None
        self.metadata: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_metadata(self) -> None:
        assert self.experiment_dir is not None, "Call start_experiment() first."
        path = os.path.join(self.experiment_dir, "metadata.json")
        with open(path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def get_output_subdir(self, subdir_name: str) -> str:
        """Return (and create) a named subdirectory inside the experiment folder."""
        assert self.experiment_dir is not None, "Call start_experiment() first."
        path = os.path.join(self.experiment_dir, subdir_name)
        os.makedirs(path, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Single-mu experiment
    # ------------------------------------------------------------------

    def start_experiment(self, net, args, dataset_name: str | None = None) -> str:
        """
        Create a timestamped directory and record metadata for a single-mu run.

        Parameters
        ----------
        net          : OIMMaxCut instance (after construction, before simulate).
        args         : argparse.Namespace from experiment_maxcut.py
                       Expected fields: graph, mu, n_init, t_end, n_points, seed.
        dataset_name : Optional override for the graph name used in the folder name.

        Returns
        -------
        str : path to the experiment directory.
        """
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = dataset_name or Path(args.graph).stem
        exp_name     = f"{dataset_name}_{timestamp}"

        self.experiment_dir = os.path.join(self.base_output_dir, exp_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.metadata = {
            "experiment_name" : exp_name,
            "experiment_type" : "single_mu",
            "timestamp"       : timestamp,
            "dataset": {
                "file"   : args.graph,
                "name"   : dataset_name,
                "n_nodes": net.n,
            },
            "network_parameters": {
                "mu"       : net.mu,
                "Ks_equiv" : net.mu / 2.0,
                "K_equiv"  : 1.0,
                "init_mode": net.init_mode,
                "timestep" : net.timestep,
            },
            "simulation_settings": {
                "n_init"  : args.n_init,
                "t_end"   : args.t_end,
                "n_points": args.n_points,
                "seed"    : args.seed,
            },
        }

        self._save_metadata()
        print(f"\n📁 Experiment directory: {self.experiment_dir}")
        return self.experiment_dir

    def log_results(self, partition: list, cut_value: float,
                    final_energy: float, mu_star: float | None = None) -> None:
        """
        Persist the best-run result and write a human-readable RESULTS.txt.

        Parameters
        ----------
        partition    : list of ±1 ints (one per node) from get_spins().
        cut_value    : binary cut weight after binarisation.
        final_energy : final Lyapunov energy L(theta).
        mu_star      : global binarisation threshold (optional but recommended).
        """
        assert self.experiment_dir is not None, "Call start_experiment() first."

        set_A = [i for i, s in enumerate(partition) if s ==  1]
        set_B = [i for i, s in enumerate(partition) if s == -1]

        self.metadata["results"] = {
            "partition"   : partition,
            "set_A"       : set_A,
            "set_B"       : set_B,
            "cut_value"   : cut_value,
            "final_energy": final_energy,
        }
        if mu_star is not None:
            mu = self.metadata["network_parameters"]["mu"]
            self.metadata["results"]["mu_star"]   = mu_star
            self.metadata["results"]["binarised"] = bool(mu > mu_star)

        self._save_metadata()

        txt_path = os.path.join(self.experiment_dir, "RESULTS.txt")
        with open(txt_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("EXPERIMENT RESULTS — OIM Max-Cut (mu parametrisation)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Dataset   : {self.metadata['dataset']['name']}\n")
            f.write(f"Nodes     : {self.metadata['dataset']['n_nodes']}\n\n")
            f.write("Network parameters:\n")
            for k, v in self.metadata["network_parameters"].items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\nBinary cut value : {cut_value:.6f}\n")
            f.write(f"Final energy     : {final_energy:.6f}\n")
            if mu_star is not None:
                f.write(f"mu*              : {mu_star:.6f}\n")
                f.write(f"Binarised        : {bool(self.metadata['network_parameters']['mu'] > mu_star)}\n")
            f.write(f"\nSet A (sigma = +1) : {set_A}\n")
            f.write(f"Set B (sigma = -1) : {set_B}\n")

    def log_all_runs(self, records: list) -> str:
        """
        Write per-IC run data to runs.csv.

        Parameters
        ----------
        records : list of dicts, each with keys:
                  ic, cut, binarized, energy, phases_final.

        Returns
        -------
        str : path to the CSV file.
        """
        assert self.experiment_dir is not None, "Call start_experiment() first."
        if not records:
            return ""

        n_nodes   = len(records[0]["phases_final"])
        csv_path  = os.path.join(self.experiment_dir, "runs.csv")
        ph_headers = [f"theta_{i}" for i in range(n_nodes)]
        fieldnames = ["ic", "cut", "binarized", "energy"] + ph_headers

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                row = {
                    "ic"       : r["ic"],
                    "cut"      : r["cut"],
                    "binarized": int(r["binarized"]),
                    "energy"   : r["energy"],
                }
                for i, h in enumerate(ph_headers):
                    row[h] = r["phases_final"][i]
                writer.writerow(row)

        self.metadata.setdefault("results", {})["runs_csv"] = csv_path
        self._save_metadata()
        return csv_path

    def save_plot(self, src_path: str, subfolder: str = "plots") -> str:
        """
        Copy a saved plot PNG into the experiment directory.

        Parameters
        ----------
        src_path  : path to the PNG produced by the experiment script.
        subfolder : sub-directory name inside the experiment folder.

        Returns
        -------
        str : destination path.
        """
        import shutil
        assert self.experiment_dir is not None, "Call start_experiment() first."
        dest_dir  = self.get_output_subdir(subfolder)
        dest_path = os.path.join(dest_dir, os.path.basename(src_path))
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
        return dest_path

    # ------------------------------------------------------------------
    # Interactive / mu-sweep experiment
    # ------------------------------------------------------------------

    def start_mu_sweep_experiment(self, args, dataset_name: str,
                                   n_nodes: int) -> str:
        """
        Create a timestamped directory and record metadata for an interactive
        or mu-sweep run.

        Parameters
        ----------
        args         : argparse.Namespace from experiment_maxcut_interactive.py
                       Expected fields: graph, mu_min, mu_max, n_mu, n_init,
                       t_end, n_points, seed.
        dataset_name : graph / dataset label used in the folder name.
        n_nodes      : number of nodes in the graph.

        Returns
        -------
        str : path to the experiment directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name  = f"mu_sweep_{dataset_name}_{timestamp}"

        self.experiment_dir = os.path.join(self.base_output_dir, exp_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.metadata = {
            "experiment_name" : exp_name,
            "experiment_type" : "mu_sweep",
            "timestamp"       : timestamp,
            "dataset": {
                "file"   : args.graph,
                "name"   : dataset_name,
                "n_nodes": n_nodes,
            },
            "sweep_parameters": {
                "mu_min"  : args.mu_min,
                "mu_max"  : args.mu_max,
                "n_mu"    : args.n_mu,
                "n_init"  : args.n_init,
                "t_end"   : args.t_end,
                "n_points": args.n_points,
                "seed"    : args.seed,
            },
        }

        self._save_metadata()
        print(f"\n📁 Experiment directory: {self.experiment_dir}")
        return self.experiment_dir

    def log_mu_sweep_results(self, results: dict,
                              experiment_dir: str | None = None) -> str:
        """
        Write per-mu sweep diagnostics to mu_sweep_results.csv.

        Parameters
        ----------
        results : dict with keys:
                  mu                   — list[float]
                  binarization_residual— list[float]
                  is_binarized         — list[bool]
                  energy               — list[float]
                  binary_cut_value     — list[float]
                  gradient_norm        — list[float]
                  phases               — list[list[float]]  (final phases per mu)
                  jacobian_eigenvalues — list[list[float]]  (eigenvalues of A)
        experiment_dir : override output path (defaults to self.experiment_dir).

        Returns
        -------
        str : path to the CSV file.
        """
        out_dir  = experiment_dir or self.experiment_dir or "."
        csv_path = os.path.join(out_dir, "mu_sweep_results.csv")

        mu_arr  = results["mu"]
        n_rows  = len(mu_arr)
        n_nodes = len(results["phases"][0])

        ph_headers  = [f"theta_{i}"   for i in range(n_nodes)]
        eig_headers = [f"jac_eig_{i}" for i in range(n_nodes)]
        fieldnames  = (
            ["mu", "binarization_residual", "is_binarized",
             "energy", "binary_cut_value", "gradient_norm"]
            + ph_headers + eig_headers
        )

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for k in range(n_rows):
                row = {
                    "mu"                    : results["mu"][k],
                    "binarization_residual" : results["binarization_residual"][k],
                    "is_binarized"          : int(results["is_binarized"][k]),
                    "energy"                : results["energy"][k],
                    "binary_cut_value"      : results["binary_cut_value"][k],
                    "gradient_norm"         : results["gradient_norm"][k],
                }
                for i, h in enumerate(ph_headers):
                    row[h] = results["phases"][k][i]
                for i, h in enumerate(eig_headers):
                    row[h] = results["jacobian_eigenvalues"][k][i]
                writer.writerow(row)

        if self.experiment_dir:
            self.metadata.setdefault("results_summary", {}).update({
                "n_mu_values"       : n_rows,
                "fraction_binarized": float(sum(results["is_binarized"]) / n_rows),
            })
            self._save_metadata()

        return csv_path

    def save_interactive_state(self, state_dict: dict) -> str:
        """
        Save an interactive-session snapshot to a timestamped JSON file.
        Called every time the slider changes in experiment_maxcut_interactive.py.

        Parameters
        ----------
        state_dict : dict produced by the interactive script's save_state().

        Returns
        -------
        str : path to the written JSON file.
        """
        out_dir = self.experiment_dir or "."
        os.makedirs(out_dir, exist_ok=True)

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(out_dir, f"state_{ts}.json")
        with open(path, "w") as f:
            json.dump(state_dict, f, indent=2)
        return path
