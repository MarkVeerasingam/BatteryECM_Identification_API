from App.Service.CapacityTest import CapacityTest
from App.Service.HPPCTest import HPPCTest
from App.Service.ECMTheveninParameterizer import ECMTheveninParameterizer

if __name__ == "__main__":
    # Choose Battery Label:
    battery_label = "V4"
    cycle_number = 0

    """
    Capacity Test:
    """
    capacity_test = CapacityTest(battery_label=battery_label)
    capacity_test.fit_soc_ocv_polynomial(degree=11)
    capacity_test.plot_ocv_soc_fitting()
    capacity_test.save_to_csv() 

    """
    Hybrid pulse power characterization (HPPC) Test:
    """
    hppc_test = HPPCTest(battery_label=battery_label, cycle_number=cycle_number)
    pulse_count = hppc_test.get_pulse_count()
    for pulse in range(pulse_count):
        hppc_test.run_analysis(pulse_number=pulse) 
        hppc_test.save_to_csv()
    hppc_test.plot_hppc_analysis()

    """
    ECM Thevenin Parameterization:
    """
    ecm_parameterizer = ECMTheveninParameterizer(battery_label=battery_label, cycle_number=cycle_number)

    # single pulse parameter optimization:
    ecm_parameterizer.load_pulses(pulse_number=0)
    ecm_parameterizer.setup_model()
    ecm_parameterizer.update_parameters()
    ecm_parameterizer.setup_problem()
    ecm_parameterizer.optimize()
    ecm_parameterizer.plot_results()
    ecm_parameterizer.export_results()

    # for pulse_number in range(pulse_count):
    #         print(f"Processing pulse {pulse_number}...")
    #         ecm_parameterizer.load_data(pulse_number)
    #         ecm_parameterizer.setup_model()
    #         ecm_parameterizer.setup_problem()
    #         ecm_parameterizer.optimize()
        
    # # Export the final results to JSON
    # ecm_parameterizer.export_results()
