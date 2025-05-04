import os
import time
import json
import pandas as pd
from gurobipy import *
from src.features_utils import extract_features

def collect_dataset(instance_dir, param_file, output_path, time_limit=60):
    # print("Files in data/instances/:", os.listdir(instance_dir))
    with open(param_file, "r") as f:
        param_sets = json.load(f)

    all_records = []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for filename in os.listdir(instance_dir):
        if filename.endswith(".mps"):
            instance_name = filename.replace(".mps", "")
            filepath = os.path.join(instance_dir, filename)
            print(f"\n Processing instance: {instance_name}")
            
            try:
                model = read(filepath)
                model.update()
                features = extract_features(model)
            except Exception as e:
                print(f"[ERROR] Cannot process {filename}: {e}")
                continue

            for param_id, param_dict in param_sets.items():
                print(f"  → Running param_id: {param_id}")
                try:
                    model.reset()
                    for k, v in param_dict.items():
                        model.setParam(k, v)
                    model.setParam("OutputFlag", 0)
                    model.setParam("TimeLimit", time_limit)

                    start = time.time()
                    model.optimize()
                    end = time.time()

                    result = {
                        "instance": instance_name,
                        "param_id": int(param_id),
                        "runtime": model.Runtime,
                        "gap": model.MIPGap if model.SolCount > 0 else 1.0,
                        "status": model.Status,
                        "obj": model.ObjVal if model.SolCount > 0 else None,
                        "success": 1
                    }

                except Exception as e:
                    print(f"[FAIL] param_id {param_id} failed: {e}")
                    result = {
                        "instance": instance_name,
                        "param_id": int(param_id),
                        "runtime": None,
                        "gap": 1.0,
                        "status": -1,
                        "obj": None,
                        "success": 0
                    }

                full_record = {**features, **param_dict, **result}
                all_records.append(full_record)

    df = pd.DataFrame(all_records)
    print(f"[DEBUG] Final number of records: {len(all_records)}")
    df.to_csv(output_path, index=False)
    print(f"\n Final dataset saved to: {output_path}")