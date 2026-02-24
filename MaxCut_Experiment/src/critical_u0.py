from pathlib import Path
import pandas as pd

# critical_u0.py is in: Experiments/MaxCut_Experiment/src/critical_u0.py
# So parents[0]=src, parents[1]=MaxCut_Experiment, parents[2]=Experiments
EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]  

name1 = EXPERIMENTS_DIR / "experiments_maxcut" / "u0_sweep" / "u0_sweep_k33_20260224_151715.csv"
name2 = EXPERIMENTS_DIR / "experiments_maxcut" / "u0_sweep" / "u0_sweep_petersen_20260224_153517.csv"
target1 = 11
target2 = 9
target3 = 7

df = pd.read_csv(name2)

df = df.sort_values(["u0", "seed"], kind="stable").reset_index(drop=True)

for target_cut in [target1, target2, target3]:
    match = df.loc[df["cut"].eq(target_cut)]
    if match.empty:
        print(f"cut == {target_cut}: not found")
    else:
        first_row = match.iloc[0]  
        print(f"cut == {target_cut}:")
        print(first_row.to_string())
        print("-" * 40)
