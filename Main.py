from App.Service.CapacityTest import CapacityTest
from App.Service.HPPCTest import HPPCTest

if __name__ == "__main__":
    # Choose Battery Label:
    battery_label = "V4"

    # Capacity Test:
    capacity_test = CapacityTest(battery_label=battery_label)
    capacity_test.fit_soc_ocv_polynomial(degree=11)
    capacity_test.plot_ocv_soc_fitting()
    capacity_test.save_to_csv() 

    # HPPC Test:
    cycle_number = 0
    hppc_test = HPPCTest(battery_label=battery_label, cycle_number=cycle_number)
    pulse_count = hppc_test.get_pulse_count()
    for pulse in range(pulse_count):
        hppc_test.run_analysis(pulse_number=pulse) 
        hppc_test.save_to_csv()
