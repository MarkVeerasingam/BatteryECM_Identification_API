import matplotlib.pyplot as plt
import numpy as np
from utils.data_plotter import plot_soc_ocv_from_capacity_test


if __name__ == "__main__":
    plot_soc_ocv_from_capacity_test(battery_label='G1', degree=11)  