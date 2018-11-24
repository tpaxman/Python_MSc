import numpy as np

def lpm_to_m3s(q_lpm):
    '''converts flow rate from L/min to m3/s'''
    q_m3s = q_lpm / 60 / 1000
    return q_m3s

def flow_to_vel(q_lpm, area_m2):
    '''converts flow rate (m3/s) and area (m2) to velocity (m/s)'''
    q_m3s = lpm_to_m3s(q_lpm)
    vel_mps = q_m3s/area_m2
    return vel_mps    
    
def calc_area(diam_m):
    area_m2 = np.pi * diam_m**2 / 4
    return area_m2

def calc_cf(pdrop, q_lpm, diam_m, rho):
    """See Slutsky et al. (1980)"""
    area_m2 = calc_area(diam_m)
    vel_mps = flow_to_vel(q_lpm, area_m2)
    cf = pdrop / (0.5 * rho * vel_mps**2)
    return cf

def calc_reynoldsnum(q_lpm, diam_m, rho, mu):
    area_m2 = calc_area(diam_m)
    vel_mps = flow_to_vel(q_lpm, area_m2)
    reynoldsnum = rho * vel_mps * diam_m / mu
    return reynoldsnum

#########################

def calc_pdrop_modpedley(q_lpm, diam_m, length_m, rho_kgm3, mu_pas, gen):
    """Modified-Pedley/van Ertbruggen model pressure drop"""
    GAMMA_DICT = {0:0.162, 1:0.239, 2:0.244, 3:0.295,
                  4:0.175, 5:0.303, 6:0.356, 7:0.566}
    gamma = GAMMA_DICT.get(gen, GAMMA_DICT[7]) # use gen7 gamma for higher gens
    reynoldsnum = calc_reynoldsnum(q_lpm, diam_m, rho_kgm3, mu_pas)
    vel_mps = flow_to_vel(q_lpm, calc_area(diam_m))
    pdrop = (gamma * (reynoldsnum*diam_m/length_m)**(1/2)
                   * 32.*mu_pas*length_m*vel_mps/diam_m**2)
    return pdrop
            
def calc_pdrop_pedley(q_lpm, diam_m, length_m, rho_kgm3, mu_pas):
    """Pedley model pressure drop"""
    q_m3s = lpm_to_m3s(q_lpm)
    reynoldsnum = calc_reynoldsnum(q_lpm, diam_m, rho_kgm3, mu_pas)
    Z = 1.85/4/np.sqrt(2) * (reynoldsnum*diam_m/length_m)**(1/2)
    pdrop_hagenpois = 128*mu_pas/np.pi * length_m*q_m3s/diam_m**4
    pdrop = Z*pdrop_hagenpois if Z > 1 else pdrop_hagenpois
    return pdrop

def calc_pdrop_blasius(q_lpm, diam_m, length_m, rho_kgm3, mu_pas, c_coef=0.316):
    """Turbulent Blasius model pressure drop"""
    ALPHA = 0.25
    reynoldsnum = calc_reynoldsnum(q_lpm, diam_m, rho_kgm3, mu_pas)
    f = c_coef*(reynoldsnum + 0.0000001)**-ALPHA
    vel_mps = flow_to_vel(q_lpm, calc_area(diam_m))
    pdrop = f * (length_m/diam_m) * (1/2) * (vel_mps**2)*rho_kgm3
    return pdrop