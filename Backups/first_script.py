import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN

N_QNOM = 4

def correcthelioxflowerror(q,H):
    TSI_Q  = H.loc[:,('tsiAirReading','mean')]
    TRUE_Q = H.loc[:,('actualHelioxFlowRate','mean')]
    return np.interp(q, TSI_Q, TRUE_Q)

def findqnom(T, fluidFlag, N_QNOM):
    #get array values, vertically aligned, of Q for a given fluid type
    qValues = T.loc[T['fluidFlag']==fluidFlag,'Qstp'].values.reshape(-1,1)     
    #find clusters of data (i.e. 4 mean values)
    kmeans = KMeans(n_clusters=N_QNOM).fit(qValues)
    #round the mean values to nearest integer
    qNomVals = kmeans.cluster_centers_.round().astype(int)
    return qNomVals
    
def xx(T, fluidFlag, N_QNOM):
    #   get array values, vertically aligned, of Q for a given fluid type
    qValues = T.loc[T['fluidFlag']==fluidFlag,'Qstp'].values.reshape(-1,1)     
    #   find clusters of data (i.e. 4 mean values)
    qGroups = DBSCAN().fit(test).labels_.reshape(-1,1)
    x = pd.Series({'q':qValues, 'g':qGroups}).to_frame()
    return x
# =============================================================================
#     x = DBSCAN().fit(qValues).
#     return x
# =============================================================================


F = pd.read_csv("./SOURCE_DATA/FLUID_VALUES.csv")
S = pd.read_csv("./SOURCE_DATA/SubjectReplicas_AirwayDimensions/SUBJECT_VALUES.csv")
T = pd.read_csv("./SOURCE_DATA/SubjectReplicas_ExperimentalMeasurements/EXPERIMENT_DATA.csv")
H = pd.read_csv("./SOURCE_DATA/SubjectReplicas_ExperimentalMeasurements/HELIOX_FLOWRATE_CONVERT.csv")
E = pd.read_csv("./SOURCE_DATA/SubjectReplicas_ExperimentalMeasurements/PDROP_EMPTY.csv")

# Polish H
H = pd.pivot_table(H, index='alicatHeliumReading', aggfunc=(np.mean,np.std))

# Polish T
mask = T.fluidFlag==2
T.loc[mask, 'Qstp'] = correcthelioxflowerror(T.loc[mask, 'Qstp'], H)



# Find qNom values and add to main table

# FIND QNOM GROUPS
FLUID_FLAG = 2;
qValues = T.loc[T['fluidFlag']==FLUID_FLAG,'Qstp'].values
qGroup = DBSCAN().fit(qValues.reshape(-1,1)).labels_
df = pd.DataFrame({'qGroup':qGroup, 'Qstp':qValues})

#FIND MEAN OF EACH GROUP
df_groups = df.groupby(by='qGroup').mean()
df_groups['Qnom']=np.round(df_groups['Qstp']).astype(int)
df_groups = df_groups.drop(columns='Qstp')

##MAP QNOM VALUES TO MAIN TABLE
df2 = pd.merge(df, df_groups, how='left', left_on='qGroup', right_on='qGroup')




# =============================================================================
# test = xx(T,1,N_QNOM)
# =============================================================================

qNomAir = findqnom(T,1,N_QNOM)
qNomHel = findqnom(T,2,N_QNOM)
    
    
    
    
    

# pd.read_csv('',sep=',',header=0)



# =============================================================================
# def MANUSCRIPT_I_CALCS():
#     ## HELIOX CONVERSION DATA IMPORT
#     #  accumulate and average the heliox Alicat-TSI flow rate conversion data
#     H = import_sourcedatafile(H_SRCFILE);
#     H = accumanddofun(H, H_VALUES, H_CRITERIA, H_ACCUMFUN);
# =============================================================================
    
    
#def accumanddofun():


# SCRAP
# TRASH IT. DON'T NEED IT.

def get_property_dict(df,indexname,propertyname):
    property_dict = df[[indexname,propertyname]].set_index(indexname).to_dict()[propertyname]
    return property_dict

def get_property_df(df,indexname,propertyname):
    property_df = df[[indexname,propertyname]].set_index(indexname)
    return property_df    
 
areaattach_dict = get_property_dict(S,'subNum','areaAttach')
rho_dict        = get_property_dict(F,'fluidName','rho')

area_df = get_property_df(S,'subNum','areaAttach')
#areaattach_dict = S[['subNum','areaAttach']].set_index('subNum').to_dict()['areaAttach']
#rho_dict = F[['fluidName','rho']].set_index('fluidName').to_dict()['rho']
    