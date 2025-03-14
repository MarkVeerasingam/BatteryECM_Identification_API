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
    hppc_test = HPPCTest(battery_label=battery_label)
    # Get the total number of available pulses
    total_pulses = hppc_test.get_pulse_count()
    print(f"Total pulses available: {total_pulses}")
    first_pulse = min(0, total_pulses)
    hppc_test.run_analysis(first_pulse)

    # Plot the results
    hppc_test.plot_hppc_analysis()