import numpy as np
import pandas as pd
from tqdm import tqdm
from operator import itemgetter


def excursion_twitter_id():
    # pre-process the twitter data
    # the original data split by comma, what `seal` need is "\t".
    data = np.array(pd.read_table("./raw_data/twitter_raw.txt", delimiter=",", header=None))
    # vertices_set = list(set(sum(data, []))) # time and memory consumption
    vertices_set = set()
    for line in data:
        vertices_set.add(line[0])
        vertices_set.add(line[1])
    vertices_set = list(vertices_set)
    vertex_map = dict([(x, vertices_set.index(x)) for x in vertices_set])

    for new_index, old_index in tqdm(enumerate(data)):
        data[new_index] = list(itemgetter(*old_index)(vertex_map))
    np.savetxt("./raw_data/twitter.txt", data, delimiter="\t", fmt="%d")
