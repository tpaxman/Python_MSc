
# coding: utf-8

# # EXPERIMENTAL DATA

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import importlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# User-defined modules:
import fluidmechanics as fluids

sns.set()


# ## Corrections for heliox flowrates (DataFrame 'H')
# The flowmeter used in the experiments (TSI) does not have a built in setting for heliox. The vendor recommended a correction factor to use while using heliox while the flowmeter was set for 'Air'. This correction factor turned out to be erroneous when testing with an Alicat flow controller, which had been validated. In order to shift all the heliox measurement flowrate to their proper values, a table of points showing the reading on the Alicat controller (correct) and the simultaneous reading on the TSI flow meter (incorrect) were recorded (in `HELIOX_FLOWRATE_CONVERT.csv`)

# In[ ]:


# IMPORT CORRECTIONS FOR ERRONEOUS HELIOX READINGS
H = pd.read_csv("./SOURCE_DATA/SubjectReplicas_ExperimentalMeasurements/HELIOX_FLOWRATE_CONVERT.csv")
H = pd.pivot_table(H, index='alicatHeliumReading', aggfunc=(np.mean,np.std))  #aggregate and average repeated measurements 


# In[ ]:


# CORRECT HELIOX FLOW RATES
def helioxcorrection(q,H):
    TSI_Q  = H.loc[:,('tsiAirReading','mean')]
    TRUE_Q = H.loc[:,('actualHelioxFlowRate','mean')]
    return np.interp(q, TSI_Q, TRUE_Q)

def correcthelioxflowrates(df,H):
    df.loc[df['fluid']=='heliox', 'qstp'] = helioxcorrection(df.loc[df['fluid']=='heliox', 'qstp'], H)
    return df


# ## Experimental measurements (DataFrame 'T')
# Pressure drop measurements were done at various settings and recorded in `EXPERIMENT_DATA.csv`. The measurements were done for all combinations of:
# * fluid type (`fluid`): *air* or *heliox*
# * configuration (`config`): with the branching airway segment either *attached* or *detached*.
# * flow rates (`qstp`): 5-30 L/min for air, 7-45 L/min for heliox, recorded by the TSI meter for STP conditions
# * subject replicas (`subnum`): 10 child subjects numbered 2, 3, 5, 6, 9, 10, 11, 12, 13, 14.

# In[ ]:


# IMPORT EXPERIMENTAL DATA
T = pd.read_csv("./SOURCE_DATA/SubjectReplicas_ExperimentalMeasurements/EXPERIMENT_DATA.csv")

# rename headers, remove unneeded columns
T = T.rename({'fluidFlag':'fluid', 'subNum':'subnum', 'attFlag':'config', 'Qstp':'qstp', 'PDrop':'pdrop'}, axis='columns')
T = T.drop(columns=['date','dayRun','Tfluid','Pfluid'])

# change fluid and configuration flags to actual descriptions
T['fluid']  = T['fluid'].map( {1:'air', 2:'heliox'})
T['config'] = T['config'].map({0:'detach', 1:'attach'})

# correct all heliox flow rate readings to their proper values
T = correcthelioxflowrates(T,H)


# In[ ]:


T[T['fluid']=='heliox'].head()


# ## Compute nominal flowrate (`qnom`)

# In[ ]:


# DIVIDE EACH READING INTO Q_NOMINAL GROUP USING 'DBSCAN'
def assignqnomgroups(df,fluid):
    qValues = df.loc[df['fluid']==fluid,'qstp'].values
    qNomGroups = DBSCAN(min_samples=3).fit(qValues.reshape(-1,1)).labels_   # min samples set to 3 allows it to work for table E
    df.loc[df['fluid']==fluid,'qnom'] = qNomGroups
    return df

T = assignqnomgroups(T,'air')          # attach
T = assignqnomgroups(T,'heliox')
T['qnom'] = T['qnom'].astype(int)  # convert index value to integer afterward


# In[ ]:


T.head()


# In[ ]:


# FIND AVERAGE QNOM VALUES FOR EACH QNOM GROUP
def get_qnom_dict(df,fluid):
    df_fluid = df.loc[df['fluid']==fluid, ['qnom','qstp']]      
    df_groups = df_fluid.groupby(by='qnom').mean()                 # average of each nominal Q group
    df_groups['qnom']=np.round(df_groups['qstp']).astype(int)    # convert qnom from float to int
    df_groups = df_groups.drop(columns='qstp')                   # 
    df_groups_dict = df_groups.to_dict()
    return df_groups_dict['qnom']

# Assign mapping of qnom groups -> values to a master dectionary for both fluids
QNOM_GROUPS_DICT_AIR = get_qnom_dict(T,'air')
QNOM_GROUPS_DICT_HEL = get_qnom_dict(T,'heliox')
QNOM_GROUPS_DICT = {'air':QNOM_GROUPS_DICT_AIR, 'heliox':QNOM_GROUPS_DICT_HEL}


# In[ ]:


QNOM_GROUPS_DICT


# In[ ]:


# REPLACE QNOM GROUP INDICES WITH ACTUAL VALUES USING THE DICTIONARY
def replace_qnom_groups_with_values(df,QNOM_GROUPS_DICT,fluid):
    df.loc[df['fluid']==fluid,'qnom'] = df.loc[df['fluid']==fluid,'qnom'].map(QNOM_GROUPS_DICT[fluid])
    return df

T = replace_qnom_groups_with_values(T,QNOM_GROUPS_DICT,'air')
T = replace_qnom_groups_with_values(T,QNOM_GROUPS_DICT,'heliox')


# In[ ]:


T = T.groupby(by=['fluid','config','subnum','qnom']).mean().reset_index()


# # Join attached and detached configurations

# In[ ]:


# Break table apart into 2 pieces for each configuration (attached, detached) 
def slicetable_on_config(T,config):
    Tconfig = T[T['config']==config]
    Tconfig = Tconfig.drop(columns=['config'])
    return Tconfig

Tattach = slicetable_on_config(T,'attach')
Tdetach = slicetable_on_config(T,'detach')


# In[ ]:


# Join tables together based on fluid type, subject number and nominal flow rate
T = pd.merge(Tattach, Tdetach,  how='left', left_on=['fluid','subnum','qnom'], right_on = ['fluid','subnum','qnom'])
T = T.rename({'pdrop_x':'pdrop_attach', 'pdrop_y':'pdrop_detach'}, axis='columns')
T = T.rename({'qstp_x' :'qstp_attach' , 'qstp_y' :'qstp_detach'},  axis='columns')


# In[ ]:


# Average each flow rate reading to use as the flow rate
qstp_list = ['qstp_attach','qstp_detach']
T['qstp'] = T[qstp_list].mean(axis=1)
T = T.drop(columns=qstp_list)


# In[ ]:


# Rearrange columns
T = T[['fluid','subnum','qnom','qstp','pdrop_attach','pdrop_detach']]


# In[ ]:


T.head()


# # Calculate $\Delta P$ nose-throat and branching

# ## Calculate sudden expansion pressure drop

# In[ ]:


F = pd.read_csv("./SOURCE_DATA/FLUID_VALUES.csv")
F = F.rename({'fluidName':'fluid', 'K_SE':'k_se','latexName':'latexname'}, axis='columns')
F = F.drop(columns='fluidFlag')


# In[ ]:


S = pd.read_csv("./SOURCE_DATA/SubjectReplicas_AirwayDimensions/SUBJECT_VALUES.csv")
S = S.rename({'subNum':'subnum', 'areaAttach':'attacharea'}, axis='columns')


# In[ ]:


def map_df_small_to_big(df_big,df_small,indexname,propertyname,keepwhat='wholetable'):
    '''map values of propertyname from a small df to a big df based on matching the indexname
    df_big - destination data frame / df_small - source data frame /
    indexname - column name to match between both tables / propertyname - value to map to bigger table based on smaller table'''
    if isinstance(indexname,str): indexname = [indexname]
    if isinstance(propertyname,str): propertyname = [propertyname]
    property_df = df_small[indexname + propertyname]
    df_aug =  pd.merge(df_big, property_df, how='left', left_on=indexname, right_on=indexname)
    if keepwhat == 'wholetable':
        return df_aug  #return whole table with new property values added
    elif keepwhat == 'newvalue':
        return df_aug[propertyname]  # or keep only the new property values


# In[ ]:


# add necessary values to table for vector calculation of p_se, then drop them after. 
T = map_df_small_to_big(T,S,'subnum','attacharea')
T = map_df_small_to_big(T,F,'fluid',['rho','mu','k_se'])

def calc_pdrop_se(q_lpm, attacharea_m2, rho, k_se):
    '''calculates sudden expansion into plenum 
    at the attachment point of replica'''
    velattach = fluids.flow_to_vel(q_lpm, attacharea_m2)
    pdrop_se = (1/2) * rho * k_se * (velattach**2)
    return pdrop_se

T['pdrop_se'] = calc_pdrop_se(T['qstp'].values, T['attacharea'].values, T['rho'].values, T['k_se'].values)
T = T.drop(['attacharea','rho','mu','k_se'], axis=1)


# In[ ]:


T.head()


# ## Import empty pressure drop readings

# In[ ]:


# Import data
E = pd.read_csv("./SOURCE_DATA/SubjectReplicas_ExperimentalMeasurements/PDROP_EMPTY.csv")

# Rename columns and categories
E = E.rename({'fluidFlag':'fluid', 'Qstp':'qstp', 'PDrop':'pdrop', 'attFlag':'config'}, axis='columns')
E['fluid']  = E['fluid'].map( {1:'air', 2:'heliox'})

# Correct erroneous heliox values
E = correcthelioxflowrates(E,H)

# Tag each reading with a qnom group
E = assignqnomgroups(E,'air')
E = assignqnomgroups(E,'heliox')
E['qnom'] = E['qnom'].astype(int)

# replace qnom groups by actual values using dictionary derived from table T
E = replace_qnom_groups_with_values(E,QNOM_GROUPS_DICT,'air')
E = replace_qnom_groups_with_values(E,QNOM_GROUPS_DICT,'heliox')

# rearrange columns
E = E[['fluid','qnom','qstp','pdrop']]
assert all((np.abs(E.qnom - E.qstp)/E.qnom).values < 0.1), 'qnom assignment for table E failed'  # check that assignment of qnom worked

# group repeated values done at the same flow rate 
E = pd.pivot_table(E, index=['fluid','qnom'], aggfunc=np.mean)  #aggregate and average repeated measurements 
E = E.reset_index()


# In[ ]:


def powerfit(x,y):
    ''' For form: y = a*x^b
    Inputs:  (x, y) / Outputs: [a, b]'''
    C = np.polyfit(np.log(x), np.log(y), 1);
    a = np.exp(C[1]);
    b = C[0];  
    return a, b

def make_pdrop_empty_lambda_for_fluid(E,fluid):
    Etemp = E[E['fluid']==fluid]
    a,b = powerfit(Etemp['qstp'].values, Etemp['pdrop'].values)
    pdrop_empty_func = lambda q: a*(q**b)
    return pdrop_empty_func

def make_pdrop_empty_main_func(E):
    functionList = []
    uniqueFluids = E.fluid.unique()
    for fluid in uniqueFluids:
        functionList.append(make_pdrop_empty_lambda_for_fluid(E,fluid))
    funcDict = dict(zip(uniqueFluids,functionList))
    pdrop_empty_func = lambda q, fluid: funcDict[fluid](q)   # Return a function to calculate pdrop_empty based on only q and fluid
    return pdrop_empty_func

# Create a function to calculate pdrop_empty based on only q and fluid
calc_pdrop_empty = make_pdrop_empty_main_func(E)


# In[ ]:


# Calculate 'pdrop_empty' to account for the exit pipe dimensions
T['pdrop_empty'] = np.vectorize(calc_pdrop_empty)(T.qstp,T.fluid)


# ## Calculate Branching and Nose-Throat pressure drop

# In[ ]:


# Calculate pdrop_dist and pdrop_nt as a function of the other recorded pressure drop values
T['pdrop_branching'] = T['pdrop_attach'] - T['pdrop_detach'] + T['pdrop_se']
T['pdrop_nosethroat'] = T['pdrop_detach'] - T['pdrop_se'] - T['pdrop_empty']
T = T.drop(['pdrop_attach','pdrop_detach','pdrop_se','pdrop_empty'], axis=1)


# In[ ]:


T.head()


# # Reynolds number ($Re$) and Coefficient of Friction ($C_F$)

# Function definitions

# In[ ]:


# add necessary values to main table in preparation for calculations
T = map_df_small_to_big(T,S,'subnum','diam')
T = map_df_small_to_big(T,F,'fluid',['rho','mu'])    

# Calculate C_F and Re and add to main table
T['cf_branching']  = fluids.calc_cf(T.pdrop_branching,  T.qstp, T.diam, T.rho)
T['cf_nosethroat'] = fluids.calc_cf(T.pdrop_nosethroat, T.qstp, T.diam, T.rho)
T['reynoldsnum']   = fluids.calc_reynoldsnum(T.qstp, T.diam, T.rho, T.mu)

# Remove temporary calculation values
T = T.drop(['diam','rho','mu'], axis=1)


# # ANALYTICAL CALCULATIONS

# In[ ]:


x = T[(T.subnum==3) & (T.fluid=='heliox')]
x.plot('reynoldsnum','cf_branching');


# In[ ]:


T.head()


# In[ ]:


#plt.rcParams['figure.figsize'] = [16,7]
#uniqueFluids = T.fluid.unique()
#uniqueSubnums = T.subnum.unique()
#legendTitles = []
#for fluid in uniqueFluids:
#   for subnum in uniqueSubnums:
#        dfx = T[(T.fluid==fluid) & (T.subnum==subnum)]
#        plt.plot(dfx.qstp,dfx.pdrop_branching)
#        legendTitles.append(fluid + ', ' + str(subnum))
#plt.legend(legendTitles,loc='best')
#plt.show()


# In[ ]:


#sns.set_style('white')
#colpal = sns.color_palette('pastel')
#sns.relplot(x="qstp", y="pdrop_branching", data=T, col='fluid', hue='subnum', palette=colpal, markers='.', kind='line');
#x=T.set_index(['fluid','qnom'],append=True)
#x.unstack('fluid')
#x.plot()2


# # TO DO #
# - Change file hierarchy in folders
# - Combine base files with MATLAB so that both can work on the data at once.

# In[ ]:


importlib.reload(fluids)


# In[ ]:


fluids.calc_pdrop_blasius(5,1,2,3,4,5)


# In[ ]:


fluids.calc_pdrop_modpedley(5,1,2,3,4,5)


# In[ ]:


fluids.calc_pdrop_pedley(5,1,2,3,4)


# In[ ]:


for gen in range(11):
    print(gen, fluids.calc_pdrop_modpedley(10,10/1000,0.1,1.2,1.8e-5,gen))


# In[ ]:


def import_branching_airway_dimensions(subnum):
    fpath = './SOURCE_DATA/SubjectReplicas_AirwayDimensions/'
    fname = 'Sub' + str(subnum)
    fext  = '.txt'
    df = pd.read_csv(fpath + fname + fext, sep='\t', 
                     names=['name','length_mm','diam_mm'])
    df['diam_m'] = df.diam_mm / 1000
    df['length_m'] = df.length_mm / 1000 
    df['gen'] = df.name.str.replace('(\d+).*', '\g<1>').astype(int)
    df = df.drop(columns=['diam_mm','length_mm'])
    return df    

def find_parents(df):
    df['parent'] = (df.gen - 1).astype(str) + df.name.str.replace('^\d+|.$','')
    df['iparent'] = ''
    for parent in df.parent[1:]:
        df.loc[df.parent==parent, 'iparent'] = df.loc[df.name == parent].index[0]
    return df

def find_daughters(df):
    daughternames = []
    daughterindex = []
    for name in a.name:
        daughternames.append(a.loc[a.parent==name,'name'].values)
        daughterindex.append(a.loc[a.parent==name].index.values)
    df['daughters'] = daughternames
    df['idaughters'] = daughterindex
    return df

a=import_branching_airway_dimensions(2)
a=find_parents(a)
a=find_daughters(a)


# In[ ]:


for x in a.daughters:
    #print(x,type(x))
    print(a.name[a.name.isin(x.tolist())].index.values)


# In[ ]:


x=a.daughters[0].tolist()
x in a.name.values.tolist()


# In[ ]:


a[a.name.isin(['1a','1b'])].index.values


# In[ ]:


a.daughters.values

