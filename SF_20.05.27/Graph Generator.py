import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from tqdm import tqdm
import pandas as pd
from pyvis.network import Network



G2 = nx.scale_free_graph(2500)
G = G2.to_undirected()

nx.write_gml(G, path="SFG_data.gml")
