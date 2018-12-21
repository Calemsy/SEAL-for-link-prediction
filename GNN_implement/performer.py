import main
import os.path
import numpy as np

data_set_name = ["mutag", "proteins", "cni1", "dd"]
performance = {}
for name_file in data_set_name:
    args = main.parse_args(name_file)
    if name_file not in performance.keys():
        performance[name_file] = []
    for i in range(10):
        print("-" * 30, name_file, "test: ", i)
        test_acc, _, _, _ = main.gnn(args)
        performance[name_file].append(test_acc)
print()
for key, value in performance.items():
    with open(os.path.join("logs", key+".txt"), "a") as f:
        for v in value:
            f.write(str(v) + "\n")
    print("%s: average acc is %f, variance is %f." % (key, np.average(value), np.std(value)))