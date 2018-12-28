import numpy as np
import main

TIMES = 10

data_names = ["Delicious", "FW1", "FW2", "FW3", "Kohonen", "SciMet"]
for data in data_names:
    auc_list = []
    for time in range(TIMES):
        print("-" * 30, data, " ", time)
        args = main.parse_args(data)
        auc = main.seal(args)
        auc_list.append(auc)
        with open("./logs/" + data + ".txt", "a") as f:
            f.write(str(auc) + "\n")
        print("average auc: %f." % (np.average(auc_list)))