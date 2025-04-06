import json
import os
import pathlib

import numpy as np
import torch

results_path = f"{os.path.dirname(os.path.realpath(__file__))}/timings"
results_path = pathlib.Path(results_path)

results_final = dict()

for res in results_path.iterdir():
    if not res.is_file():
        continue

    with open(res) as file:
        results = json.load(file)

    configs = sorted(
        enumerate(results["configs"]),
        key=lambda x: (
            np.mean(runs) if len(runs := results["result"][x[0]]) > 0 else float("inf")
        ),
    )
    best_config = configs[0][1]

    name = res.name
    prefix, _, compute, dtype, dim = name.split("-")
    dtype = eval(dtype, dict(torch=torch))
    dim = int(dim.split(".")[0])

    prefix = f"{compute} {prefix}"
    results_final.setdefault(prefix, {})
    results_final[prefix][(dtype, dim)] = tuple(best_config)

for prefix in sorted(results_final):
    print(prefix)
    for key in sorted(results_final[prefix], key=lambda x: (str(x[0]), *x[1:])):
        print(f"{key}: {results_final[prefix][key]},")
