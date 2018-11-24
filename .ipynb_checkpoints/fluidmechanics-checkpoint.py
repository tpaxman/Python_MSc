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

def calc_pdropblasius(q_m3s, diam_m, length_m, rho_kgm3, mu_pas, c_coef):
    """Calculate pressure drop  with the turbulent Blasius model
    where the coefficient C can be varied (originally C = 0.316)
    f calculation"""
    ALPHA = 0.25
    reynoldsnum = calc_reynoldsnum(q_lpm, diam_m, rho_kgm3, mu_pas)
    f = c_coef * (reynoldsnum + eps) ** -ALPHA
    #% Velocity calculation
    area_m2 = diam2area(diam_m)
    V = flow_to_vel(q_lpm, area_m2)
    pDrop = f * (length_m / diam_m) * (1 / 2) * (V ** 2) * rho_kgm3