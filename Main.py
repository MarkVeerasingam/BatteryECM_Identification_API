from App.Service.CapacityTest import CapacityTest
from App.Service.HPPCTest import HPPCTest

if __name__ == "__main__":
    battery_label = "G1"

    capacity_test = CapacityTest(battery_label=battery_label, degree=11)
    soc, ocv = capacity_test.fit_soc_ocv_polynomial()
    capacity_test.show_ocv_soc_plot(soc, ocv)
    # capacity_test.save_to_csv(soc, ocv) 

    hppc_test = HPPCTest(battery_label="G1")
    hppc_test.run_analysis(pulse_number=9)
    hppc_test.plot_hppc_analysis()