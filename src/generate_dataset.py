import os
import json
import yaml
import pandas as pd
from gurobipy import read, GRB
from tqdm import tqdm


# ===FUNCTIONS===
# load config.yaml
def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# load instances.txt
def load_instances(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip().replace(".mps.gz", "").lower() for line in f]

# 
def load_benchmark_features(csv_path, instance_list, feature_cols):
    df = pd.read_csv(csv_path)
    df["instance"] = df["InstanceInst."].str.lower()
    df_filtered = df[df["instance"].isin(instance_list)]
    return df_filtered[["instance"] + feature_cols]

#
def load_solu_dict(solu_path, instance_list):
    solu = {}
    with open(solu_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] in ("=opt=", "=best="):
                name = parts[1].lower()
                if name in instance_list:
                    solu[name] = float(parts[2])
    return solu

def load_param_sets(param_path):
    with open(param_path, "r") as f:
        return json.load(f)

#
def solve_instances(instances, param_sets, solu_dict, folder, time_limit, gap_threshold):
    records = []
    for name in tqdm(instances):
        filename = os.path.join(folder, name + ".mps")
        if not os.path.exists(filename):
            continue
        best_obj = solu_dict[name]
        for param_id, param in param_sets.items():
            try:
                model = read(filename)
                model.setParam("OutputFlag", 0)
                model.setParam("TimeLimit", time_limit)
                model.setParam("MIPGap", gap_threshold)
                for k, v in param.items():
                    model.setParam(k, v)
                model.optimize()

                if model.Status == GRB.OPTIMAL or model.SolCount > 0:
                    obj = model.ObjVal
                    runtime = model.Runtime
                    gap = abs(obj - best_obj) / max(abs(best_obj), 1e-6)
                    runtime_rec = runtime if gap <= gap_threshold else 3600
                    
                    records.append({
                        "instance": name,
                        "param_id": int(param_id),
                        "runtime": runtime_rec,
                        "objval": obj,
                        "status": int(model.Status),
                        "gap": gap
                    })
            except Exception as e:
                print(f"[ERROR!]{name} | param {param_id} failed: {e}")
    return pd.DataFrame(records)

def build_training_dataset(df_results, df_features, feature_cols, output_path):
    df_merged = df_results.merge(df_features, on="instance", how="left")
    df_final = df_merged[feature_cols + ["param_id", "runtime", "gap", "status", "objval"]]
    df_final.to_csv(output_path, index=False)
    print(f"Training dataset saved to {output_path} (rows: {len(df_final)})")
    return df_final


# ===MAIN===
def main():
    config = load_config()
    
    # ===CONFIGURATIONS===
    # paths config
    starter_path = config["PATHS"]["STARTER_DIR"]
    starter_inst = config["PATHS"]["STARTER_INSTANCES"]
    solu_path = config["PATHS"]["SOLU"]
    param_set_path = config["PATHS"]["PARAMS_SETS"]
    benchmark_csv = config["PATHS"]["BENCHMARK_CSV"]
    output_csv = config["PATHS"]["OUTPUT_CSV"]

    # settiings config
    time_limit = config["SETTINGS"]["TIME_LIMITS"]
    mip_gap = config["SETTINGS"]["GAP_THRESHOLD"]

    # feature cols
    FEATURE_COLS = config["FEATURE_COLS"]

    instances = load_instances(starter_inst)
    df_features = load_benchmark_features(benchmark_csv, instances, FEATURE_COLS)
    solu_dict = load_solu_dict(solu_path, instances)
    param_sets = load_param_sets(param_set_path)
    df_results = solve_instances(instances, param_sets, solu_dict, starter_path, time_limit, mip_gap)
    build_training_dataset(df_results, df_features, FEATURE_COLS, output_csv)

if __name__ == "__main__":
    main()