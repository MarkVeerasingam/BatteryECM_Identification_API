from App.Service.CapacityTest import CapacityTest
from App.Service.HPPCTest import HPPCTest

if __name__ == "__main__":
    battery_label = "G1"

    # Capacity Test
    # capacity_test = CapacityTest(battery_label=battery_label, degree=11)
    # soc, ocv = capacity_test.fit_soc_ocv_polynomial()
    # capacity_test.show_ocv_soc_plot(soc, ocv)
    # capacity_test.save_to_csv(soc, ocv) 

    # HPPC Test
    cycle_number = 3
    hppc_test = HPPCTest(battery_label=battery_label, cycle_number=cycle_number)
    pulse_count = hppc_test.get_pulse_count()
    for pulse in range(pulse_count):
        hppc_test.run_analysis(pulse_number=pulse) 
        hppc_test.save_to_csv()