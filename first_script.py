import numpy as np
import pandas as pd

def helioxflowratecorrector(q,H):
    TSI_Q  = H.loc[:,('tsiAirReading','mean')]
    TRUE_Q = H.loc[:,('actualHelioxFlowRate','mean')]
    return np.interp(q, TSI_Q, TRUE_Q)

F = pd.read_csv("./SOURCE_DATA/FLUID_VALUES.csv")
S = pd.read_csv("./SOURCE_DATA/SubjectReplicas_AirwayDimensions/SUBJECT_VALUES.csv")
T = pd.read_csv("./SOURCE_DATA/SubjectReplicas_ExperimentalMeasurements/EXPERIMENT_DATA.csv")
H = pd.read_csv("./SOURCE_DATA/SubjectReplicas_ExperimentalMeasurements/HELIOX_FLOWRATE_CONVERT.csv")
E = pd.read_csv("./SOURCE_DATA/SubjectReplicas_ExperimentalMeasurements/PDROP_EMPTY.csv")


H = pd.pivot_table(H, index='alicatHeliumReading', aggfunc=(np.mean,np.std))

mask = T.fluidFlag==2
T.loc[mask, 'Qstp'] = helioxflowratecorrector(T.loc[mask, 'Qstp'], H)
    
    
    
    
    

# pd.read_csv('',sep=',',header=0)



# =============================================================================
# def MANUSCRIPT_I_CALCS():
#     ## HELIOX CONVERSION DATA IMPORT
#     #  accumulate and average the heliox Alicat-TSI flow rate conversion data
#     H = import_sourcedatafile(H_SRCFILE);
#     H = accumanddofun(H, H_VALUES, H_CRITERIA, H_ACCUMFUN);
# =============================================================================
    
    
#def accumanddofun():
    