import numba
import numpy as np

parameter_index_mapping = {
    "c_ee": 0, "c_ei": 1, "c_ie": 2, "c_ii": 3, 
    "tau_e": 4, "tau_i": 5,
    "theta_e": 6, "theta_i": 7,
    "a_e": 8, "a_i": 9,
    "k_e": 10, "k_i": 11, "r_e": 12, "r_i": 13,
    "noise_E": 14, "noise_I": 15,
}
parameter_latex_mapping = {
    "c_ee": "$c_{{{ee}}}$", "c_ei": "$c_{{{ei}}}$", "c_ie": "$c_{{{ie}}}$", "c_ii": "$c_{{{ii}}}$", 
    "tau_e": "$\\tau_e$", "tau_i": "$\\tau_i$",
    "theta_e": "$\\theta_e$", "theta_i": "$\\theta_i$",
    "a_e": "$a_e$", "a_i": "$a_i$",
    "k_e": "$k_e$", "k_i": "$k_i$", "r_e": "$r_e$", "r_i": "$r_i$",
    "noise_E": "$Noise_E$", "noise_I": "$Noise_I$",
}
parameter_names = [par for par in parameter_index_mapping]


# Model dynamics
@numba.njit
def fn_Edot(E,I, params, global_input):
    return (-E + (params[10] - params[12] * E) * 1.0 / (1 + np.exp(-params[8] * (params[0] * E - params[1] * I - params[6] + global_input))))/params[4]
@numba.njit
def fn_Idot(E, I, params):
    return (-I + (params[11] - params[13] * I) * 1.0 / (1 + np.exp(-params[9] * (params[2] * E - params[3] * I - params[7]))))/params[5]

limit_cycle40Hz = np.array([
                                16.0, 12.00, 15.0, 3.00,         #c_xx
                                8.0, 8.0,                        #tau_x
                                2., 3.7,                       #theta_x
                                1.3, 2.00,                       #a_x
                                1.0, 1.0, 1.0, 1.0,              #k_x, r_x
                                0.0000e-06, 0.0000e-06           #noise
                    ])