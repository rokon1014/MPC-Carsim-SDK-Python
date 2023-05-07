""" Carsim Run class
This can be used to run Carsim Simulation step given a defined import and export array

"""

import argparse
import ctypes
from time import sleep
import vs_solver
import math


class CarsimRUn:
    def __init__(self, sim_file_filename, path_to_vs_dll):
        self.sim_file_filename = sim_file_filename
        self.vs = vs_solver.vs_solver()
        self.vs_dll = ctypes.cdll.LoadLibrary(path_to_vs_dll)
        self.import_array = [0.0, 0.0, 0.0]
        self.export_array = []
        self.configuration = None
        self.t_current = 0
        self.tc = 0

        if self.vs_dll is None:
            print("dll file problem")
            return

    def check_configuration(self):
        if self.vs.get_api(self.vs_dll):
            self.configuration = self.vs.read_configuration(self.sim_file_filename)
            self.t_current = self.configuration.get("t_start")
            t_step_size = self.configuration.get("t_step")
            self.export_array = self.vs.copy_export_vars(
                self.configuration.get("n_export")
            )  # get export variables from vs solver

            if len(self.export_array) < 1:
                print("At least three export parameters needed.")
                return

            return self.t_current, t_step_size, self.export_array

    def run(self, delta_in_rad):
        self.t_current = self.t_current + self.configuration.get("t_step")
        t_step_size = self.configuration.get("t_step")

        # Converted to degerees and Multiplied for steering ratio (steering wheel to vehicle wheel??)
        self.import_array[0] = math.degrees(delta_in_rad) * 20
        status, self.export_array = self.vs.integrate_io(
            self.t_current, self.import_array, self.export_array
        )

        return self.t_current, t_step_size, self.export_array

    def terminate(self):
        self.vs.terminate_run(self.t_current)


if __name__ == "__main__":
    sim_file_filename = "gps_path.sim"
    path_to_vs_dll = "carsim_64.dll"

    sim_time = 0
    try:
        csr = CarsimRUn(sim_file_filename, path_to_vs_dll)
        init_state = csr.check_configuration()

        while sim_time < 5:
            sim_time, time_step, states = csr.run(math.radians(10))
            print(sim_time, time_step, states)
        csr.terminate()
    except Exception as e:
        csr.terminate()
        print(e)
        raise
