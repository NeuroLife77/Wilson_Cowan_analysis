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


@numba.njit
def get_WC_ts(
            parameters,
            global_input = 0.0,
            length: float = 12,
            dt: float = 0.5,
            initial_conditions = np.array([0.25,0.25]),
            noise_seed: int = 42,
            store_I: bool = False,
            is_noise_log_scale = True,
            
    ):
    # Input parameters are of shape (num_param_set, num_parameters) to match parameter estimation output, but we need (num_parameters,num_param_set) to simulate
    params = np.zeros_like(parameters.T) + parameters.T
    if is_noise_log_scale:
        params[params.shape[0]-2:] = 10**(params[params.shape[0]-2:])
        params[-2,params[-2]==1] = 0
        params[-1,params[-1]==1] = 0 
    # Set seed
    np.random.seed(noise_seed)
    # White noise
    DE, DI = np.sqrt(2*params[-2]* dt), np.sqrt(2*params[-1]* dt)

    num_param_set = parameters.shape[0] 

    # Equivalent to allocating memory
    sim_length = int(1000/dt*length)

    time_series_E = np.zeros((sim_length+1, int(num_param_set)))
    time_series_I = np.zeros((3,int(num_param_set)))
    if store_I:
        time_series_I = np.empty((sim_length+1,int(num_param_set)))
    time_series_E_temp = np.zeros((1,int(num_param_set)))
    time_series_I_temp = np.zeros((1,int(num_param_set)))
    time_series_E_corr = np.zeros((1,int(num_param_set)))
    time_series_I_corr = np.zeros((1,int(num_param_set)))
    time_series_E_noise = np.zeros((1,int(num_param_set)))
    time_series_I_noise = np.zeros((1,int(num_param_set)))

    # Set initial conditions
    time_series_E[0] = initial_conditions[0]
    time_series_I[0] = initial_conditions[1]
    
    # Heun performed in-place within the time_series_X arrays to maximize speed
    for i in range(int(1000/dt*length)-1):
        # Forward Euler
        j_0 = i
        j_1 = (i+1)
        if not store_I:
            j_0 = i%2
            j_1 = (i+1)%2
        # Calculating input from other nodes
                #               c_ee   *       E          -   c_ei    *        I           -  theta_e  + global_input
        time_series_E[i+1] = params[0] * time_series_E[i] - params[1] * time_series_I[j_0] - params[6]  + global_input
        #                       c_ie   *       E          -   c_ii    *        I           -  theta_i
        time_series_I[j_1] = params[2] * time_series_E[i] - params[3] * time_series_I[j_0] - params[7]
        #                    c_e /  1 +    exp(-  a_e     *    node input E   )      
        time_series_E[i+1] = 1.0 / (1 + np.exp(-params[8] * time_series_E[i+1]))
        #                    c_i /  1 +    exp(-  a_i     *    node input I    )
        time_series_I[j_1] = 1.0 / (1 + np.exp(-params[9] * time_series_I[j_1]))
        #                         (     k_e     -    r_e     *        E       ) *  S_e(input node E) -       E         ) /   tau_e
        time_series_E[i+1] = dt*(((params[10] - params[12] * time_series_E[i]) * time_series_E[i+1]) - time_series_E[i]) / params[4]
        #                         (     k_i     -    r_i     *        I       )  *  S_i(input node I)  -       I       )    /   tau_i 
        time_series_I[j_1] = dt*(((params[11] - params[13] * time_series_I[j_0]) * time_series_I[j_1]) - time_series_I[j_0]) / params[5] 
        time_series_E_temp = time_series_E[i] + time_series_E[i+1] 
        time_series_I_temp = time_series_I[j_0] + time_series_I[j_1]
        # Corrector point
        #                       c_ee   *       E          -   c_ei      *        I           -  theta_e  + global_input
        time_series_E_corr = params[0] * time_series_E_temp - params[1] * time_series_I_temp - params[6] + global_input
        #                       c_ie   *       E            -   c_ii    *        I           -  theta_i
        time_series_I_corr = params[2] * time_series_E_temp - params[3] * time_series_I_temp - params[7]
        #                    c_e /  1 +    exp(-  a_e     *    node input E    )  
        time_series_E_corr = 1.0 / (1 + np.exp(-params[8] * time_series_E_corr))
        #                    c_i /  1 +    exp(-  a_i     *    node input I    )  
        time_series_I_corr = 1.0 / (1 + np.exp(-params[9] * time_series_I_corr))
        #                         (   k_e     -    r_e     *           E       ) *  S_e(input node E)  -       E           ) /   tau_e
        time_series_E_corr = dt*(((params[10] - params[12] * time_series_E_temp) * time_series_E_corr) - time_series_E_temp) / params[4] 
        #                         (   k_i     -    r_i     *           I       ) *  S_i(input node I)  -       I           ) /   tau_i
        time_series_I_corr = dt*(((params[11] - params[13] * time_series_I_temp) * time_series_I_corr) - time_series_I_temp) / params[5]
        # Heun point
        time_series_E_noise = np.random.normal(0,1,size=num_param_set) *  DE 
        time_series_I_noise = np.random.normal(0,1,size=num_param_set) *  DI
        time_series_E[i+1] = time_series_E[i] + (time_series_E[i+1]+time_series_E_corr)/2 + time_series_E_noise
        time_series_I[j_1] = time_series_I[j_0] + (time_series_I[j_1]+time_series_I_corr)/2 + time_series_I_noise
    return time_series_E[:-1].T, time_series_I[:-1].T