import numba
import numpy as np
fp_coding = {
    -4:'k *', # Should not exist (numerical artifact)
    -3:'b *', # Either a limit cycle or a very weakly stable/unstable spiral
    -2:'g *', # Stable spiral
    -1:'r *', # Unstable spiral
    -0:'r .',  # No classification condition was met, but it spins
    0:'r .', # No classification condition was met
    1:'r X',  # Unstable node
    2:'g X',  # Stable node
    3:'b X',  # Either a neutral point or a very weakly stable/unstable node
    4:'k X'   # Saddle node
}
fp_code_meaning = {
    -4:'Should not exist (numerical artifact)',
    -3:'Either a limit cycle or a very weakly stable/unstable spiral',
    -2:'Stable spiral',
    -1:'Unstable spiral',
    -0:'No classification condition was met, but it spins',
    0:' No classification condition was met',
    1:'Unstable node',
    2:'Stable node',
    3:'Either a neutral point or a very weakly stable/unstable node',
    4:'Saddle node',
}
# Jacobian
@numba.njit
def fn_Edot_E(E, I, params, global_input): # Just the math
    Edot_E = 1/params[4] * (-1 - (params[12]/(1 + np.exp(-params[8] * (params[0] * E - params[1] * I + global_input - params[6])))) + (params[10] - params[12] * E) * (( np.exp(-params[8] * (params[0] * E - params[1] * I + global_input - params[6])) * (params[8] * params[0]))/(1 + np.exp(-params[8] * (params[0] * E - params[1] * I + global_input - params[6])))**2))
    return Edot_E
@numba.njit
def fn_Edot_I(E, I, params, global_input): # Just the math
    Edot_I = 1/params[4] * ((params[10] - params[12] * E) * (( np.exp(-params[8] * (params[0] * E - params[1] * I + global_input - params[6])) * (-params[8] * params[1]))/(1 + np.exp(-params[8] * (params[0] * E - params[1] * I + global_input - params[6])))**2))
    return Edot_I
@numba.njit
def fn_Idot_E(E, I, params): # Just the math
    Idot_E = 1/params[5] * ((params[11] - params[13] * I) * (( np.exp(-params[9] * (params[2] * E - params[3] * I - params[7])) * (params[9] * params[2]))/(1 + np.exp(-params[9] * (params[2] * E - params[3] * I - params[7])))**2))
    return Idot_E
@numba.njit
def fn_Idot_I(E, I, params): # Just the math
    Idot_I = 1/params[5] * (-1 - (params[13]/(1 + np.exp(-params[9] * (params[2] * E - params[3] * I - params[7])))) + (params[11] - params[13] * I) * (( np.exp(-params[9] * (params[2] * E - params[3] * I - params[7])) * (-params[9] * params[3]))/(1 + np.exp(-params[9] * (params[2] * E - params[3] * I - params[7])))**2))
    return Idot_I
@numba.njit
def get_jac(E,I, params, global_input): # Just the math
    Edot_E = fn_Edot_E(E, I, params, global_input)
    Edot_I = fn_Edot_I(E, I, params, global_input)
    Idot_E = fn_Idot_E(E, I, params)
    Idot_I = fn_Idot_I(E, I, params)
    return Edot_E,Edot_I,Idot_E,Idot_I

# Nullclines
@numba.njit
def fn_E_nullcline(I,params, global_input): # Just the math
    log_component = np.log((-params[13] * I + params[11])/I -1)
    return (params[9]*params[3]*I + params[9]*params[7] - log_component)/(params[9]*params[2])
@numba.njit
def fn_I_nullcline(E,params, global_input): # Just the math
    log_component = np.log((-params[12] * E + params[10])/E -1)
    return -(params[8]*params[6] - params[8]*params[0]*E - log_component - params[8]*global_input)/(params[8]*params[1])

# Newton method

# h_E and h_I
@numba.njit
def fn_h_E(E, params, global_input): # Just the math
    return E - fn_E_nullcline(fn_I_nullcline(E, params, global_input), params, global_input)

@numba.njit
def fn_h_I(I, params, global_input): # Just the math
    return I - fn_I_nullcline(fn_E_nullcline(I, params, global_input), params, global_input)

@numba.njit
def fn_E_nullcline_prime(I,params, global_input): # Just the math
    numerator = params[11] + params[3] + params[7]
    denominator = 1 + params[9] * params[2]**2 * params[11] * I - params[9] * params[2]**2 * I**2 * (params[13] + 1)
    return numerator/denominator
@numba.njit
def fn_I_nullcline_prime(E,params, global_input): # Just the math
    numerator = params[10] - params[0] + params[6] - global_input
    denominator = 1 + params[8] * params[1]**2 * params[10] * E - params[8] * params[1]**2 * E**2 * (params[12] + 1)
    return numerator/denominator

@numba.njit
def fn_h_E_prime(E,params, global_input): # Just the math
    dInull_dE = fn_I_nullcline_prime(E,params, global_input)
    Inull_E = fn_I_nullcline(E,params, global_input)
    dE_null_dI_at_Inull_E = fn_E_nullcline_prime(Inull_E,params, global_input)
    return 1 - dE_null_dI_at_Inull_E * dInull_dE

@numba.njit
def fn_h_I_prime(I,params, global_input): # Just the math
    dEnull_dI = fn_E_nullcline_prime(I,params, global_input)
    Enull_I = fn_E_nullcline(I,params, global_input)
    dI_null_dE_at_Enull_I = fn_I_nullcline_prime(Enull_I,params, global_input)
    return 1 - dI_null_dE_at_Enull_I * dEnull_dI

# Newton step
@numba.njit
def fn_newton_step_h_E(E, params, global_input, damp = 1.0): # Just the math
    h_E = fn_h_E(E, params, global_input)
    h_E_prime = fn_h_E_prime(E, params, global_input)
    return E - damp * h_E/h_E_prime
@numba.njit
def fn_newton_step_h_I(I, params, global_input, damp = 1.0): # Just the math
    h_I = fn_h_I(I, params, global_input)
    h_I_prime = fn_h_I_prime(I, params, global_input)
    return I - damp * h_I/h_I_prime

@numba.njit
def newton_iterate_h_E(E_0, params, global_input, damp = 0.25, n_iters = 10000):
    # "params" has the shape (16,N) where N is the number of distinct parameter sets to find the fixed points for,
    # all N parameter sets will be run in a vectorized way
    E_vals = np.zeros((2,params.shape[-1]))
    I_vals = np.zeros((2,params.shape[-1]))
    # Initialize E, I coordinates with E_0 and the associated estimate of I
    E_vals[0] = E_0
    I_vals[0] =  fn_I_nullcline(E_vals[0], params, global_input)
    for newt_step in range(n_iters): # Run newton steps for n_iters 
        # Using a circular array to keep RAM usage low
        newt_step_0 = newt_step%2 
        newt_step_1 = (newt_step+1)%2
        # Find next E estimate
        E_vals[newt_step_1] = fn_newton_step_h_E(E_vals[newt_step_0], params, global_input, damp = damp)
        # Find associated I estimate
        I_vals[newt_step_1] =  fn_I_nullcline(E_vals[newt_step_1], params, global_input)
    return E_vals[newt_step_1], I_vals[newt_step_1]

@numba.njit
def newton_iterate_h_I(I_0, params, global_input, damp = 0.25, n_iters = 10000):
    # "params" has the shape (16,N) where N is the number of distinct parameter sets to find the fixed points for,
    # all N parameter sets will be run in a vectorized way
    E_vals = np.zeros((2,params.shape[-1]))
    I_vals = np.zeros((2,params.shape[-1]))
    # Initialize E, I coordinates with I_0 and the associated estimate of I
    I_vals[0] = I_0
    E_vals[0] =  fn_E_nullcline(I_vals[0], params, global_input)
    for newt_step in range(n_iters): # Run newton steps for n_iters 
        # Using a circular array to keep RAM usage low
        newt_step_0 = newt_step%2
        newt_step_1 = (newt_step+1)%2
        # Find next I estimate
        I_vals[newt_step_1] = fn_newton_step_h_I(I_vals[newt_step_0], params, global_input, damp = damp)
        # Find associated E estimate
        E_vals[newt_step_1] =  fn_E_nullcline(I_vals[newt_step_1], params, global_input)
    return E_vals[newt_step_1], I_vals[newt_step_1]

@numba.njit
def multi_start_h_X(
                    parameters,
                    state_space,
                    n_starts,n_iters,
                    global_input = 0, damp = 0.05,
                    edge_res = -3, init_at_nullclines = True
    ):
    # "parameters" has the shape (N,16) as it's more intuitive to generate
    # "params" needs to be of shape (16,N) for the helper functions
    params = parameters.T 
    
    # Running the newton method for multiple starting point locations, covering the whole state space, 
    # helps minimize the chances of missing some fixed points. However, when initializing at the nullclines,
    # the resolution at the edges of the nullclines is much lower due to the sigmoid activation function shape.
    # the code below splits the initialization points (n_starts) in 1/5, 3/5, 1/5 proportions, allocating 1/5 of 
    # the points each edge to be spaced in a log-space fashion to better cover the edges of the nullclines. The 
    # middle 3/5 are spaced linearly.
    edge_starts = n_starts//5
    mid_starts = int(n_starts - (2 * edge_starts))
    Es = np.zeros((n_starts,params.shape[-1]))
    Is = np.zeros((n_starts,params.shape[-1]))
    log_step = np.logspace(-15,edge_res,edge_starts)
    # First edge
    Es[:edge_starts] = (state_space[0][0] + log_step)[:,None]
    Is[:edge_starts] = (state_space[1][0] + log_step)[:,None]
    # Middle range
    Es[edge_starts:edge_starts + mid_starts] = np.linspace(
                                                            state_space[0][0]+log_step[-1],
                                                            state_space[0][1]-log_step[-1],
                                                            mid_starts
                                                )[:,None]
    Is[edge_starts:edge_starts + mid_starts] = np.linspace(
                                                            state_space[1][0]+log_step[-1],
                                                            state_space[1][1]-log_step[-1],
                                                            mid_starts
                                                )[:,None]
    # Second edge
    Es[edge_starts + mid_starts:] = (state_space[0][1] - log_step[::-1])[:,None]
    Is[edge_starts + mid_starts:] = (state_space[1][1] - log_step[::-1])[:,None]

    if init_at_nullclines:
        init_E = fn_E_nullcline(Is, params, global_input)
        init_I = fn_I_nullcline(Es, params, global_input)
    else:
        init_E = Es
        init_I = Is
    
    x_h_X = np.zeros((2,2,n_starts,parameters.shape[0]))
    # Run newton for each initialization on each nullcline (or simply the Es,Is initialization if "init_at_nullclines==False")
    for j,E_0 in enumerate(init_E):
        x_h_X[0,0,j],x_h_X[1,0,j] = newton_iterate_h_E(E_0, params, global_input, damp = damp, n_iters = n_iters)
    for j,I_0 in enumerate(init_I):
        x_h_X[0,1,j],x_h_X[1,1,j] = newton_iterate_h_I(I_0, params, global_input, damp = damp, n_iters = n_iters)

    return x_h_X

def get_FPs(
                parameters,  global_input,
                state_space, n_starts, n_iters,
                unique_threshold=6, damp=0.05,
                init_at_nullclines=True,  edge_res = -3,
                return_counts = False
    ):
    # Run newton method from multiple starting points
    x_h_X = multi_start_h_X(
                                parameters,
                                state_space,
                                n_starts,n_iters,
                                global_input=global_input, damp=damp,
                                init_at_nullclines=init_at_nullclines, edge_res=edge_res
            )
    # Ensure that there is no nan or inf
    x_h_X[np.isnan(x_h_X)] = -1
    x_h_X[np.isinf(x_h_X)] = -1
    # Group up the starts on the E nullcline with the starts on the I nullcline
    x_h = x_h_X.reshape((2,-1,parameters.shape[0]))

    # Group up all points that are closer than "unique_threshold" allows together to define the set of unique fixed points 
    unique_points = []
    for s in range(parameters.shape[0]):
        pts, cts = np.unique(np.round(x_h[...,s],unique_threshold).T,axis = 0, return_counts=True)
        pts_ok = (pts==-1).sum(-1)==0 # Remove all points that were nan or inf
        pts_kept = pts[pts_ok,:]
        cts_kept = cts[pts_ok]
        if return_counts:
            unique_points.append([pts_kept,cts_kept])
        else:
            unique_points.append(pts_kept)
    return unique_points

def get_Es_Is_log_edges(state_space, n_points, edge_res = -3):
    # Running the newton method for multiple starting point locations, covering the whole state space, 
    # helps minimize the chances of missing some fixed points. However, when initializing at the nullclines,
    # the resolution at the edges of the nullclines is much lower due to the sigmoid activation function shape.
    # the code below splits the initialization points (n_starts) in 1/5, 3/5, 1/5 proportions, allocating 1/5 of 
    # the points each edge to be spaced in a log-space fashion to better cover the edges of the nullclines. The 
    # middle 3/5 are spaced linearly.
    edge_points = n_points//5
    mid_points = int(n_points - (2 * edge_points))
    Es = np.zeros((n_points))
    Is = np.zeros((n_points))
    log_step = np.logspace(-15,edge_res,edge_points)
    # First edge
    Es[:edge_points] = (state_space[0][0] + log_step)
    Is[:edge_points] = (state_space[1][0] + log_step)
    # Middle range
    Es[edge_points:edge_points + mid_points] = np.linspace(
                                                            state_space[0][0]+log_step[-1],
                                                            state_space[0][1]-log_step[-1],
                                                            mid_points
                                            )
    Is[edge_points:edge_points + mid_points] = np.linspace(
                                                            state_space[1][0]+log_step[-1],
                                                            state_space[1][1]-log_step[-1],
                                                            mid_points
                                            )
    # Second edge
    Es[edge_points + mid_points:] = (state_space[0][1] - log_step[::-1])
    Is[edge_points + mid_points:] = (state_space[1][1] - log_step[::-1])

    return Es, Is
    
# Fixed point classification 
@numba.njit
def get_lambdas(E, I, params, global_input):
    # Jacobian
    Edot_E, Edot_I, Idot_E, Idot_I = get_jac(E,I, params, global_input)
    # Trace, determinant & discriminant
    tr = np.asarray(Edot_E + Idot_I, dtype = np.complex128)
    det = np.asarray(Edot_E * Idot_I - Edot_I * Idot_E, dtype = np.complex128)
    discriminant = (tr**2 - 4 * det)
    # Eigenvalues
    lambda_0 = tr/2 + np.sqrt(discriminant)/2
    lambda_1 = tr/2 - np.sqrt(discriminant)/2
    return lambda_0, lambda_1, discriminant

def classify_fixed_points(
                            fixed_points, parameters, global_input,
                            imag_threshold = 0.85, real_threshold = 0.75,
                            norm_imag_lambdas = True
    ):
    fp_class = []
    eigval = []
    # For parameter set 
    for param_set in range(len(fixed_points)):
        fp_class.append([])
        eigval.append([])
        # Initialize the value for normalizing the real component of the parameter set's fixed point's eigenvalues 
        max_real_val = 1e-10
        # For each fixed point
        for fp in range(len(fixed_points[param_set])):
            # Handles the possibility of having distinct global input values for each parameter set
            if isinstance(global_input,float) or isinstance(global_input,int):
                g_inp = global_input
            else:
                g_inp = global_input[param_set]

            # Get the eigenvalues for the fixed point
            lambda_0, lambda_1, discriminant = get_lambdas(
                                                            fixed_points[param_set][fp][0], fixed_points[param_set][fp][1],
                                                            parameters[param_set], g_inp
                                            )
            # Split into real and imaginary components
            real_0, real_1 = np.real(lambda_0),np.real(lambda_1)
            imag_0, imag_1 = np.imag(lambda_0),np.imag(lambda_1)
            # Possibly normalize the imaginary component (taking its unit vector value)
            # to make thresholding more consisten
            if norm_imag_lambdas: 
                lambda_0_norm = np.sqrt(real_0**2+imag_0**2)
                lambda_1_norm = np.sqrt(real_1**2+imag_1**2)
                imag_0 = imag_0/lambda_0_norm
                imag_1 = imag_1/lambda_1_norm
                lambda_0 = real_0 + imag_0*1j
                lambda_1 = real_1 + imag_1*1j

            # Keeps track of the most stable FP's real value (magnitude)
            max_real_val = max(max_real_val, abs(min(real_1, real_0))) 

            eigval[param_set].append([lambda_0, lambda_1, discriminant])

        # Loop again over all fixed points (now with the eigenvalues)    
        for fp in range(len(fixed_points[param_set])):
            # Normalize the real values by scaling everything to have the largest real component of all eigenvalues to 1 
            # to make thresholding more consisten
            real_0, real_1 = np.real(eigval[param_set][fp][0])/max_real_val, np.real(eigval[param_set][fp][1])/max_real_val
            imag_0, imag_1 = np.imag(eigval[param_set][fp][0]), np.imag(eigval[param_set][fp][1])
            # Initialize classification features 
            spins = 1 # Cycle & spirals
            stable = 0 # Stable vs unstable
            
            if np.abs(imag_0) > imag_threshold or np.abs(imag_1) > imag_threshold: 
                # Complex values big enough
                spins = -1
            if real_0 > real_threshold and real_1 > real_threshold: 
                # Real values both positive and stronger than the 
                stable = 1 # Unstable 
            elif real_0 < 0 and real_1 < 0: 
                # Real values both negative
                stable = 2 # Stable 
            elif np.abs(real_0) < real_threshold and np.abs(real_1) < real_threshold: 
                # They are not negative but below threshold
                stable = 3 # Neutral or much weaker than the strongest stable FP
            elif (real_0 < 0 and real_1 >= 0) or (real_1 < 0 and real_0 >= 0):
                # One is positive and one is negative
                stable = 4 # Saddle 
            
            fp_class[param_set].append(spins * stable)

    return fp_class, eigval

