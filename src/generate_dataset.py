import os
import json
import yaml
import time
import pandas as pd
from gurobipy import read, GRB
from tqdm import tqdm
from datetime import datetime

# ===FUNCTIONS===
# logging file
def get_log_path():
    date_str = datetime.now().strftime("%Y-%m-%d")
    return f"logs/solve_log_{date_str}.txt"

def log_message(msg, log_path):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"{timestamp} {msg}\n")

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
def solve_single_instance(name, param_sets, solu_dict, model_folder, time_limit, gap_threshold):
    filename = os.path.join(model_folder, name + ".mps")
    if not os.path.exists(filename):
        return []

    best_obj = solu_dict.get(name)
    records = []

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
            print(f"[ERROR!] {name} | param {param_id} failed: {e}")
    return records

#
def solve_instances_incremental(instances, param_sets, solu_dict, folder, time_limit, gap_threshold, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = get_log_path()
    total_start = time.time()

    for name in tqdm(instances):
        outfile = os.path.join(output_dir, f"{name}.csv")
        if os.path.exists(outfile):
            log_message(f"[SKIPPED] {name}.csv", log_path)
            continue

        try:
            results = solve_single_instance(name, param_sets, solu_dict, folder, time_limit, gap_threshold)
            if results:
                pd.DataFrame(results).to_csv(outfile, index=False)
                log_message(f"[SUCCESS] {name}.csv ({len(results)} rows)", log_path)
            else:
                log_message(f"[EMPTY] {name}.csv", log_path)
        except Exception as e:
            log_message(f"[ERROR] {name}.csv - {str(e)}", log_path)

    total_time = time.time() - total_start
    log_message(f"[DONE] Solved {len(instances)} instances in {total_time:.1f} seconds.", log_path)

#
def build_training_dataset(df_results, df_features, feature_cols, output_path):
    df_merged = df_results.merge(df_features, on="instance", how="left")
    df_final = df_merged[feature_cols + ["param_id", "runtime", "gap", "status", "objval"]]
    df_final.to_csv(output_path, index=False)
    print(f"Training dataset saved to {output_path} (rows: {len(df_final)})")
    return df_final

#
def merge_instance_results(output_dir):
    all_dfs = []
    for file in os.listdir(output_dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(output_dir, file))
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


# ===MAIN===
def main():
    config = load_config()
    
    # ===CONFIGURATIONS===
    # paths config
    param_set_path = config["PATHS"]["PARAMS_SETS"]
    solu_path = config["PATHS"]["SOLU"]
    benchmark_csv = config["PATHS"]["BENCHMARK_CSV"]
    starter_path = config["PATHS"]["STARTER_DIR"]
    starter_inst = config["PATHS"]["STARTER_INSTANCES"]
    starter_output_csv = config["PATHS"]["STARTER_OUTPUT_CSV"]
    model_path = config["PATHS"]["MODEL_DIR"]
    model_inst = config["PATHS"]["MODEL_INSTANCES"] 
    model_output_csv = config["PATHS"]["MODEL_OUTPUT_CSV"]
    partial_csv_dir = config["PATHS"]["PARTIAL_DIR"]

    # settiings config
    time_limit = config["SETTINGS"]["TIME_LIMITS"]
    mip_gap = config["SETTINGS"]["GAP_THRESHOLD"]

    # feature cols
    FEATURE_COLS = config["FEATURE_COLS"]

    instances = load_instances(model_inst)
    df_features = load_benchmark_features(benchmark_csv, instances, FEATURE_COLS)
    solu_dict = load_solu_dict(solu_path, instances)
    param_sets = load_param_sets(param_set_path)
    solve_instances_incremental(instances, param_sets, solu_dict, model_path, time_limit, mip_gap, partial_csv_dir)
    df_results = merge_instance_results(partial_csv_dir)
    build_training_dataset(df_results, df_features, FEATURE_COLS, model_output_csv)

if __name__ == "__main__":
    main()