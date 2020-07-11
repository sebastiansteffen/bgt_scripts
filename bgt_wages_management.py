#!/usr/bin/env python
# coding: utf-8

# In[1]:


# TODO

# skill_list also contains isSpecialized, isBaseline, isSoftware. Use this.
# occ_list also contains OccFam, Employer, Sector. Use this.
# tf-idf counts

# FINISHED
# incorporated predicted wages.
# incorporated mean, variance for BGTJobId actual wages, # skills.
# incorporated BLS OES wages.


# # Libraries

# In[2]:


#import sys
#!{sys.executable} -m pip install --upgrade --user numpy
#!{sys.executable} -m pip install --upgrade --user pandas
import pandas as pd
import numpy as np
import os, io, requests, re, csv, zipfile, xlrd
from collections import Counter, defaultdict


# # Constants

# In[104]:


# home_dir = "/mnt/hgfs/Dropbox (MIT)/research/occupational_drift"
home_dir = "/home/ssteffen/projects/ssteffen_proj/bgt_occ_change"
data_dir = "/nfs/pool001/ssteffen/data"
bgt_data_dir = os.path.join(data_dir, 'bgt')
other_data_dir = os.path.join(data_dir, 'other')
bgt_derived_data_dir = os.path.join(data_dir, 'bgt_derived')
os.chdir(home_dir)

# FLAGS
is_debug = False
main_vars = ['naics3', 'soc6'] # can include 'socX', 'naicsX', 'occ_type', 'fips', 'firm'
groupby_vars = ['scf', 'sc', 's', 't'] + main_vars
time_var = 'y' # Cannot be changed to monthly since we do yearly variances.
use_predicted_wages = False #whether to import and merge on BGT predicted wages (currently only exist for 2017, 2018).
use_job_ad_wages = False
use_oes_wages = False 
use_deming_skills = False

if is_debug:
    NROWS = 10000
else:
    NROWS = None

# SOC aggregation - exclude military
soc_list = ['55']
# NAICS2 aggregation
naics_list = []

if use_job_ad_wages:
    save_dir = os.path.join(bgt_derived_data_dir, '_'.join(main_vars + [time_var]) + '_wage')
else:
    save_dir = os.path.join(bgt_derived_data_dir, '_'.join(main_vars + [time_var]))
print(f'Output Directory: {save_dir}')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
if is_debug:
    df_main_y_s_f_name = os.path.join(save_dir, 'test_df_main_y_s.csv')
    df_main_y_sc_f_name = os.path.join(save_dir, 'test_df_main_y_sc.csv')
    df_main_y_scf_f_name = os.path.join(save_dir, 'test_df_main_y_scf.csv')
    
    df_main_y_s_w_f_name = os.path.join(save_dir, 'test_df_main_y_s_w.csv')
    df_main_y_sc_w_f_name = os.path.join(save_dir, 'test_df_main_y_sc_w.csv')
    df_main_y_scf_w_f_name = os.path.join(save_dir, 'test_df_main_y_scf_w.csv')
    
    df_main_y_f_name = os.path.join(save_dir, 'test_df_main_y.csv')
    df_y_f_name = os.path.join(save_dir, 'test_df_y.csv')
    
else:
    df_main_y_s_f_name = os.path.join(save_dir, 'df_main_y_s.csv')
    df_main_y_sc_f_name = os.path.join(save_dir, 'df_main_y_sc.csv')
    df_main_y_scf_f_name = os.path.join(save_dir, 'df_main_y_scf.csv')
    
    df_main_y_s_w_f_name = os.path.join(save_dir, 'df_main_y_s_w.csv')
    df_main_y_sc_w_f_name = os.path.join(save_dir, 'df_main_y_sc_w.csv')
    df_main_y_scf_w_f_name = os.path.join(save_dir, 'df_main_y_scf_w.csv')
    
    df_main_y_f_name = os.path.join(save_dir, 'df_main_y.csv')
    df_y_f_name = os.path.join(save_dir, 'df_y.csv')


# 
# # Functions

# In[7]:


def _index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
              return i
            
def _get_bls_year_data(year):
    '''
    Download zipped BLS data for a given year.
    Then unzip, clean, and return it as a dataframe.
    Example url to the BLS zip file: https://www.bls.gov/oes/special.requests/oesm10nat.zip
    Source: https://www.bls.gov/oes/tables.htm#2010
    TODO: use A_MEAN since it has less missing and may account for hourly seasonality.
    '''
    if len(str(year)) == 4:
        # Allow 4-digit years.
        year = int(str(year)[-2:])
    url = 'https://www.bls.gov/oes/special.requests/oesm{0}nat.zip'.format(year)
    if is_debug:
        print(url, year)
    response = requests.get(url)
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    files = zip_file.namelist()
    index = _index_containing_substring(files, 'national')
    with zip_file.open(files[index], 'r') as f:
            print(files)
            try:
                oes = pd.read_excel(f)
            except:
                oes = pd.read_excel(f)
    if year in list(range(10, 12)):
            oes = oes[oes['GROUP'].isnull()]
            oes = oes.replace('*',np.nan)
            oes = oes.replace('#',np.nan)
            oes = oes[oes['H_MEAN'].notnull()]
            oes = oes[oes['A_MEAN'].notnull()]
            #oes = oes[oes['H_MEDIAN'].notnull()]
            #oes = oes[oes['A_MEDIAN'].notnull()]
            
            # Calculate values to be reported as descriptive statistics
            wageE2010 = oes[oes['H_MEAN'].notnull()]['TOT_EMP'].sum()
            wageO2010 = oes[oes['H_MEAN'].notnull()]['TOT_EMP'].sum()
            oes[oes['H_MEAN'].notnull()]['TOT_EMP'].count()
            wage2010 = oes[oes['H_MEAN'].isnull()]['TOT_EMP']
            # Add a year column
            oes['year'] = '20{}'.format(year)
            # Calculate employment share
            oes['occEmpShare'] = oes['TOT_EMP'] / oes['TOT_EMP'].sum()
            # Drop unnecessary columns
            oes = oes[['OCC_CODE', 'OCC_TITLE', 'TOT_EMP', 'H_MEAN', 'A_MEAN', 'year', 'occEmpShare']]
    elif year in list(range(12, 19)):
        oes = oes[oes['OCC_GROUP']=='detailed']
        oes = oes.replace('*',np.nan)
        oes = oes.replace('#',np.nan)
        oes = oes[oes['H_MEAN'].notnull()]
        oes = oes[oes['A_MEAN'].notnull()]
            # Calculate values to be reported as descriptive statistics                                                   
        tot2017 = oes['TOT_EMP']
        wage2017 = oes[oes['H_MEAN'].notnull()]['TOT_EMP']
        # Add a year column
        oes['year'] = '20{}'.format(year)
        # Calculate employment share
        oes['occEmpShare'] = oes['TOT_EMP'] / oes['TOT_EMP'].sum()
        # Drop unnecessary columns
        oes = oes[['OCC_CODE', 'OCC_TITLE', 'TOT_EMP', 'H_MEAN', 'A_MEAN', 'year', 'occEmpShare']]
    else:
        raise

    oes['year'] = oes['year'].astype(np.int64)
    oes['year_tot_emp'] = oes.groupby('year')['TOT_EMP'].transform('sum')
    oes['emp_share'] = oes['TOT_EMP'] / oes['year_tot_emp']
    return(oes)

def _get_bls_data(years = range(10, 19)):
    '''
    Get all of the BLS data and append it.
    '''
    oes = pd.DataFrame()
    for year in list(years):
        if is_debug:
            print('Importing BLS data for May 20{0}.'.format(year))  
        oes = oes.append(_get_bls_year_data(year))
    print("Finished importing BLS data.")
    return(oes)

if is_debug:
    oes = _get_bls_data()
    fp = data_dir + "/other/soc_emp.csv"
    print(f'Saving BLS wages, employment data to: {fp}')
    oes.to_csv(fp, sep = '\t', index = False)
    # oes = pd.read_csv(fp, sep = '\t')
    display(oes.head(5))

    
def m2(x):
    mean = np.mean(x)
    count = len(x)
    result = np.sum(np.power(np.subtract(x, [mean] * count), 2))
    
    return result

# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
def update(newValues, existingAggregate = (0, 0.0, 0.0)):
    if isinstance(newValues, (int, float, complex)):
        # Handle single digits.
        newValues = [newValues]
        
    (count, mean, M2) = existingAggregate
    count += len(newValues) 
    # newvalues - oldMean
    delta = np.subtract(newValues, [mean] * len(newValues))
    mean += np.sum(delta / count)
    # newvalues - newMeant
    delta2 = np.subtract(newValues, [mean] * len(newValues))
    M2 += np.sum(delta * delta2)

    return (count, mean, M2)

def update1(newValue, existingAggregate = (0, 0.0, 0.0)):
# Only for single number updates.
    (count, mean, M2) = existingAggregate
    count += 1 
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2

    return (count, mean, M2)

# retrieve the mean, standard deviation and sample standard deviation
def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    (mean, variance, sampleVariance) = (mean, M2/count, M2/(count - 1)) 
    if count == 0:
        return float('nan')
    elif count == 1:
        return (mean, 0.0, 0.0)
    else:
        return (mean, np.sqrt(variance), np.sqrt(sampleVariance))
    
def lower_case_list(l):
    return [x.lower() for x in l]   

def pretty_plot_top_n(series, top_n=5, index_level=0):
    '''
    Source: https://sigdelta.com/blog/text-analysis-in-pandas/
    Usage: pretty_plot_top_n(counts['n_w'])
    Requires a (grouped) pd.Series
    '''
    r = series    .groupby(level=index_level)    .nlargest(top_n)    .reset_index(level=index_level, drop=True)
    r.plot.bar()
    return r.to_frame()


# # Main

# ## Aggregation

# In[8]:


# BGT wage predictions
# need to merge these on and see what year they are for.

# BGT data
file_list = {}
for data_files in next(os.walk(bgt_data_dir))[1]:
    print(data_files)
    files = []
    for r, d, f in os.walk(os.path.join(bgt_data_dir, data_files)):
        for file in f:
            if '.zip' in file:
                files.append(os.path.join(r, file)) 
        file_list[data_files] = files

main_file_list = file_list['Main']
skill_file_list = file_list['Skill']
# if is_debug:
#     main_file_list = main_file_list[5:36:12] 
#     skill_file_list = skill_file_list[5:36:12]


# In[ ]:


# df_tax = pd.read_csv(os.path.join(other_data_dir, 'skill_taxonomy_deming.csv'), sep='\t')
# y = 2010
# m = 10
# y_m = '{0}-{1:02d}'.format(y, m)
# relevant_skill_file = [s for s in skill_file_list if y_m in s]
# skill_list = pd.read_csv(relevant_skill_file[0], compression='zip', encoding='latin_1', sep='\t', na_values = [-999, 'na'], dtype={'BGTJobId': np.int64, 'Skill': str, 'SkillCluster': str, 'SkillClusterFamily': str})
# skill_list = skill_list.merge(right=df_tax, on='Skill', how='inner')
# display(skill_list[skill_list['num_deming_skills']>0].head(5))


# In[102]:


#%%script false
# Skills are unique at the BGTJobId level, but skill clusters and skill cluster families are not. 
# Thus, if a BGTJobId contains many buzzword skills, we would overcount the actual number of skill clusters and skill cluster families.
# To get around this, I create binary counts at the skill cluster and skill cluster family level separately.

print(f'Unique (skill, ym, {", ".join(str(x) for x in main_vars)}) counts.')
# Download Skill data.
# Loop over each month file to compile the list we want.

if use_predicted_wages:
    df_wages1 = pd.read_csv(os.path.join(other_data_dir, 'salary1_clean.csv'), encoding='latin_1', sep='\t', na_values = [-999, 'na'])
    df_wages2 = pd.read_csv(os.path.join(other_data_dir, 'salary-reproc-111018_clean.csv'), encoding='latin_1', sep='\t', na_values = [-999, 'na'])

if use_oes_wages:
    oes_fp = data_dir + "/other/soc_emp.csv"
    #oes = _get_bls_data()
    oes = pd.read_csv(oes_fp, sep = '\t')

if use_deming_skills:
    df_tax = pd.read_csv(os.path.join(other_data_dir, 'skill_taxonomy_deming.csv'), sep='\t')
    
#try: del big_ct
#except: pass
if is_debug:
    years = [2015, 2017]
    months = [1, 2]
else:
    years = [2007] + list(range(2010, 2021))
    # years = list(range(2016, 2020))
    months = list(range(1, 13))

df_main_y_s_all = []
df_main_y_sc_all = []
df_main_y_scf_all = []

df_main_y_s_w_all = []
df_main_y_sc_w_all = []
df_main_y_scf_w_all = []

df_main_y_all = []
df_y_all = []

for y in years:
    df_y = []
    for m in months:
        y_m = '{0}-{1:02d}'.format(y, m)
        print('Ingesting data for {0}.'.format(y_m))
        # Find the relevant file
        relevant_main_file = [s for s in main_file_list if y_m in s]
        relevant_skill_file = [s for s in skill_file_list if y_m in s]

        # Import main file
        if not relevant_main_file:
            continue
            
        occ_list = pd.read_csv(relevant_main_file[0], compression='zip', nrows = NROWS, 
                               encoding='latin_1', sep='\t', na_values = [-999, 'na'])
        occ_list.rename(columns = {'SOC':'soc', 'SOCName':'soc_name', 'Employer':'firm',
                                   'Sector':'sector', 'SectorName':'sector_name', 'NAICS3':'naics3', 
                                   'NAICS4':'naics4', 'NAICS5':'naics5', 'NAICS6':'naics6',
                                  'City':'city', 'State':'state', 'County':'county', 'FIPSState':'fips_state',
                                  'FIPSCounty':'fips_county', 'FIPS':'fips', 'Lat':'lat', 'Lon':'lon',
                                  'BestFitMSA':'best_fit_msa', 'BestFitMSAName':'best_fit_msa_name',
                                   'BestFitMSAType':'best_fit_msa_type', 'MSA':'msa', 'MSAName':'msa_name'
                                  }, 
                        inplace = True)
        
        occ_list = occ_list[['BGTJobId', 'soc', 'soc_name', 'firm', 'sector', 'sector_name', 'naics3', 'naics4', 'naics5',
                            'naics6', 'city', 'state', 'county', 'fips_state', 'fips_county', 'fips', 'lat',
                            'lon', 'msa', 'msa_name']]
        occ_list['t'] = int(y_m[:4])

        if use_job_ad_wages:
            if is_debug:
                print('Dropping job ads with empty wages.')  
            occ_list['wage_a'] = occ_list[['MinSalary', 'MaxSalary']].mean(axis=1, skipna=True)
            occ_list['wage_h'] = occ_list[['MinHrlySalary', 'MaxHrlySalary']].mean(axis=1, skipna=True)
            occ_list.dropna(subset=['BGTJobId', 'wage_a'], inplace=True)
            occ_list = occ_list[occ_list['wage_a' != 0.0]]
            occ_list = occ_list[occ_list['wage_h' != 0.0]]

        if use_predicted_wages:
            # TODO: Incorporate this into the aggregation. Right now they get dropped.
            if is_debug:
                print('Merging on predicted wages by BGTJobId from Chewning, Liu, Gaurav (KDD 2018).')
        # Filter out job ads for which we do not have wages.
            occ_list = occ_list.merge(right=df_wages1, on='BGTJobId', how='inner')
            occ_list = occ_list.merge(right=df_wages2, on='BGTJobId', how='inner')
            if occ_list.empty:
                print(f'No wages in {y_m}.')
                continue  
        
        if np.any(['soc' in word for word in main_vars]):
        # SOC Level cleaning.
            occ_list.dropna(subset=['soc'], inplace=True)
            occ_list['soc_name'] = occ_list['soc_name'].astype(str)
            occ_list['soc'] = occ_list['soc'].astype(str)
            occ_list['soc6'] = occ_list['soc'].str.replace('-', '', regex=False)
            occ_list['soc4'] = occ_list['soc6'].str[:4]
            occ_list['soc2'] = occ_list['soc6'].str[:2]
            # Filter SOC codes.
            if soc_list:
                occ_list = occ_list[~occ_list['soc2'].isin(soc_list)]
                
        if np.any(['firm' in word for word in main_vars]):
            if is_debug:
                print('Dropping empty firm names.')
                occ_list['firm'] = occ_list['firm'].str.replace('[\t\n\r\f\v]', '')
            occ_list.dropna(subset=['firm'], inplace=True)
            
        if np.any(['fips' in word for word in main_vars]):
            if is_debug:
                print('Dropping empty FIPS.')
            occ_list.dropna(subset=['fips'], inplace=True)
            
        if use_oes_wages:
            if is_debug:
                print('Merging on the BLS data by soc code.')
            occ_list = occ_list.merge(right=oes, left_on=['soc', 't'], right_on=['OCC_CODE', 'year'], how='inner') 
            occ_list['wage_a_weighted'] = occ_list['A_MEAN'] * occ_list['emp_share']
            occ_list['wage_h_weighted'] = occ_list['H_MEAN'] * occ_list['emp_share']

        if np.any(['naics' in word for word in main_vars]): 
            # NAICS Level cleaning.
            occ_list.dropna(subset=['naics3'], inplace=True)
            occ_list['naics3'] = occ_list['naics3'].astype(np.int64)
            occ_list['naics2'] = occ_list['naics3'].astype(str).str[:2].astype(int)
            # Filter NAICS codes.
            if naics_list:
                occ_list = occ_list[~occ_list['naics2'].isin(naics_list)] 
        
        if np.any(['occ_type' in word for word in main_vars]):
            occ_list.dropna(subset=['soc'], inplace=True)
            # Management measures
            # https://stackoverflow.com/questions/19913659/pandas-conditional-creation-of-a-series-dataframe-column
            conditions = [
            (occ_list['soc'].str.startswith('11')),
            ((occ_list['soc'].str.slice(0, 2).apply(int) >= 13) & (occ_list['soc'].str.slice(0, 2).apply(int) <= 31)),
            ((occ_list['soc'].str.slice(0, 2).apply(int) >= 33) & (occ_list['soc'].str.slice(0, 2).apply(int) < 55) & (occ_list['soc'].str.contains('-10')))
            ]
            choices = ['is_top_manager', 'is_professional', 'is_middle_manager']
            occ_list['occ_type'] = np.select(conditions, choices, default='other')
        
        # Import skill file
        skill_list = pd.read_csv(relevant_skill_file[0], compression='zip', nrows = NROWS,
                                 encoding='latin_1', sep='\t', na_values = [-999, 'na'], dtype={'BGTJobId': np.int64, 'Skill': str, 'SkillCluster': str, 'SkillClusterFamily': str})
        skill_list.rename(columns = {'SkillClusterFamily':'scf', 'SkillCluster':'sc', 'Skill':'s'},
                         inplace = True)
        skill_list = skill_list.dropna(subset=['BGTJobId', 's'])
        # Clean the skills.
        skill_list['s'] = skill_list['s'].str.replace('[\t\n\r\f\v]', '')
        skill_list['sc'] = skill_list['sc'].str.replace('[\t\n\r\f\v]', '')
        skill_list['scf'] = skill_list['scf'].str.replace('[\t\n\r\f\v]', '')
        
        skill_list = skill_list[['BGTJobId', 's', 'sc', 'scf']]
        
        skill_list['s_n'] = skill_list.groupby('BGTJobId')['s'].transform('nunique')
        skill_list['sc_n'] = skill_list.groupby('BGTJobId')['sc'].transform('nunique')
        skill_list['scf_n'] = skill_list.groupby('BGTJobId')['scf'].transform('nunique')
        
        
        # Merge together by BGTJobId
        if is_debug:
            print('Merging on the skills data.')
        df = occ_list.merge(right=skill_list, on='BGTJobId', how='inner')  
        
        if is_debug:
            print('Appending to the yearly file.')
        df_y.append(df)
    
    df_y = pd.concat(df_y, axis=0)
    

    
    if is_debug:
        print('Aggregating to (main_vars, time, skill) level.')
    # Binary Count by main_vars X t X skills
    df_main_y_s = df_y[['BGTJobId'] + main_vars + ['t', 's']].groupby(main_vars + ['t', 's']).count()
    
    # Binary Count by main_vars X t X skill cluster
    df_main_y_sc = df_y[['BGTJobId'] + main_vars + ['t', 'sc']].groupby(['BGTJobId'] + main_vars + ['t', 'sc']).nunique()
    df_main_y_sc = df_main_y_sc[[]].reset_index()
    df_main_y_sc = df_main_y_sc.groupby(main_vars + ['t', 'sc']).count()
    
    # Binary Count by main_vars X t X skill cluster family
    df_main_y_scf = df_y[['BGTJobId'] + main_vars + ['t', 'scf']].groupby(['BGTJobId'] + main_vars + ['t', 'scf']).nunique()
    df_main_y_scf = df_main_y_scf[[]].reset_index()
    df_main_y_scf = df_main_y_scf.groupby(main_vars + ['t', 'scf']).count()
    
    # Posting-S-Count-Weighted S Counts by main_vars X t X S
    df_main_y_s_w = df_y[['s_n'] + main_vars + ['t', 's']].copy()
    df_main_y_s_w['s_n'] = 1.0 / df_main_y_s_w['s_n']
    df_main_y_s_w.rename(columns = {'s_n': 'BGTJobId'}, inplace = True)
    df_main_y_s_w = df_main_y_s_w.groupby(main_vars + ['t', 's']).sum()
    
    # Posting-SC-Count-Weighted SC Counts by main_vars X t X SC
    df_main_y_sc_w = df_y[['sc_n'] + main_vars + ['t', 'sc']].copy()
    df_main_y_sc_w['sc_n'] = 1.0 / df_main_y_sc_w['sc_n']
    df_main_y_sc_w.rename(columns = {'sc_n': 'BGTJobId'}, inplace = True)
    df_main_y_sc_w = df_main_y_sc_w.groupby(main_vars + ['t', 'sc']).sum()
    
    # Posting-SCF-Count-Weighted SCF Counts by main_vars X t X SCF
    df_main_y_scf_w = df_y[['scf_n'] + main_vars + ['t', 'scf']].copy()
    df_main_y_scf_w['scf_n'] = 1.0 / df_main_y_scf_w['scf_n']
    df_main_y_scf_w.rename(columns = {'scf_n': 'BGTJobId'}, inplace = True)
    df_main_y_scf_w = df_main_y_scf_w.groupby(main_vars + ['t', 'scf']).sum()
    
    
    # Skill, Wage Variables by main_vars X t
    if is_debug:
        print('Aggregating to (main_vars, time) level.')
        
    if (not use_oes_wages):
        df_main_y = df_y.groupby(main_vars + ['t'], as_index=False).agg({'BGTJobId': 'count',
                                              's_n': ['mean', 'std', 'median'],
                                              'sc_n': ['mean', 'std', 'median'],
                                              'scf_n': ['mean', 'std', 'median']
                                             })
    else:
        df_main_y = df_y.groupby(main_vars + ['t'], as_index=False).agg({'BGTJobId': 'count',
                                              'wage_a': ['mean', 'std'],
                                              'wage_h': ['mean', 'std'],
                                              'wage_a_weighted': ['mean'],
                                              'wage_h_weighted': ['mean'],                   
                                              's_n': ['mean', 'std', 'median'],
                                              'sc_n': ['mean', 'std', 'median'],
                                              'scf_n': ['mean', 'std', 'median']
                                             })      
    
    
    # Rename columns.
    df_main_y.columns = ["_".join(x) for x in df_main_y.columns.ravel()]
    df_main_y.columns = [re.sub('_$', '', x) for x in df_main_y.columns]
    df_main_y.set_index(main_vars + ['t'], inplace=True)
    
    # Overall Variables by t
    if is_debug:
        print('Aggregating to (time) level.')
    if (not use_oes_wages):
        df_y1 = df_y.groupby(['t'], as_index=False).agg({'BGTJobId': 'count',
                                         's_n': ['mean', 'std', 'median'],
                                         'sc_n': ['mean', 'std', 'median'],
                                         'scf_n': ['mean', 'std', 'median']
                                         })
    else:
        df_y1 = df_y.groupby(['t'], as_index=False).agg({'BGTJobId': 'count',
                                         'wage_a': ['mean', 'std'],
                                         'wage_h': ['mean', 'std'],
                                         'wage_a_weighted': ['mean', 'std'],
                                         'wage_h_weighted': ['mean', 'std'], 
                                         's_n': ['mean', 'std', 'median'],
                                         'sc_n': ['mean', 'std', 'median'],
                                         'scf_n': ['mean', 'std', 'median']
                                         })
    
    # Rename columns.
    df_y1.columns = ["_".join(x) for x in df_y1.columns.ravel()]
    df_y1.columns = [re.sub('_$', '', x) for x in df_y1.columns]
    df_y1.set_index(['t'], inplace=True)
    
    if is_debug:
        print('Appending to the overall file.')
    df_main_y_s_all.append(df_main_y_s)
    df_main_y_sc_all.append(df_main_y_sc)
    df_main_y_scf_all.append(df_main_y_scf)
    
    df_main_y_s_w_all.append(df_main_y_s_w)
    df_main_y_sc_w_all.append(df_main_y_sc_w)
    df_main_y_scf_w_all.append(df_main_y_scf_w)
    
    df_main_y_all.append(df_main_y)
    df_y_all.append(df_y1)

df_main_y_s_all = pd.concat(df_main_y_s_all, axis=0)
df_main_y_sc_all = pd.concat(df_main_y_sc_all, axis=0)
df_main_y_scf_all = pd.concat(df_main_y_scf_all, axis=0)

df_main_y_s_w_all = pd.concat(df_main_y_s_w_all, axis=0)
df_main_y_sc_w_all = pd.concat(df_main_y_sc_w_all, axis=0)
df_main_y_scf_w_all = pd.concat(df_main_y_scf_w_all, axis=0)


df_main_y_all = pd.concat(df_main_y_all, axis=0)
df_y_all = pd.concat(df_y_all, axis=0)

print('Done!')


# In[103]:


# display(df_main_y_scf_w_all.head(10))
# display(df_main_y_scf_all.head(10))


# ## Save Data

# In[ ]:


# Save all
df_main_y_s_all.to_csv(df_main_y_s_f_name, sep='\t')
df_main_y_sc_all.to_csv(df_main_y_sc_f_name, sep='\t')
df_main_y_scf_all.to_csv(df_main_y_scf_f_name, sep='\t')

df_main_y_s_w_all.to_csv(df_main_y_s_w_f_name, sep='\t')
df_main_y_sc_w_all.to_csv(df_main_y_sc_w_f_name, sep='\t')
df_main_y_scf_w_all.to_csv(df_main_y_scf_w_f_name, sep='\t')

df_main_y_all.to_csv(df_main_y_f_name, sep='\t')
df_y_all.to_csv(df_y_f_name, sep='\t')


# ## (Optional) Load Data

# In[ ]:


# Load Data
# df_main_y_s_all = pd.read_csv(df_main_y_s_f_name, sep='\t', index_col=[0, 1, 2])
# df_main_y_sc_all = pd.read_csv(df_main_y_sc_f_name, sep='\t', index_col=[0, 1, 2])
# df_main_y_scf_all = pd.read_csv(df_main_y_scf_f_name, sep='\t', index_col=[0, 1, 2])

# df_main_y_s_w_all = pd.read_csv(df_main_y_s_w_f_name, sep='\t', index_col=[0, 1, 2])
# df_main_y_sc_w_all = pd.read_csv(df_main_y_sc_w_f_name, sep='\t', index_col=[0, 1, 2])
# df_main_y_scf_w_all = pd.read_csv(df_main_y_scf_w_f_name, sep='\t', index_col=[0, 1, 2])


# df_main_y_all = pd.read_csv(df_main_y_f_name, sep='\t', index_col=[0, 1])
# df_y_all = pd.read_csv(df_y_f_name, sep='\t', index_col=[0])


# In[ ]:


# eof

