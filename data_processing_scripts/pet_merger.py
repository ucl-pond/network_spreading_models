'''
pet_merger.py

Merging PET CSVs (no VISCODE) with ADNIMERGE via IDASEARCH (Visit & ADNI_VISCODE2_Assignment_Methods.pdf) and REGISTRY (VISCODE <> VISCODE2)

Neil Oxtoby (github:noxtoby), CMICHACKS23, 9 November 2023
'''
import pandas as pd
from pathlib import Path

path_to_data = Path("./data")

# REGISTRY table maps VISCODE to VISCODE2
csv_registry = path_to_data / 'REGISTRY_09Nov2023.csv'
df_registry = pd.read_csv(csv_registry) # VISCODE to VISCODE2 (ADNIMERGE)
mapper_viscode_viscode2 = dict( [(c,d) for c,d in zip(df_registry['VISCODE'].values,df_registry['VISCODE2'].values)] )

# All PETs: original, preprocessed, and processed
csv_idaSearch = path_to_data / 'idaSearch_11_09_2023.csv'
df_idaSearch  = pd.read_csv(csv_idaSearch) # Image ID to LONIUID (pet)
df_idaSearch['RID'] = df_idaSearch['Subject ID'].map(lambda x: x.split('_')[-1]).astype(int)

# ADNIMERGE: for demographics, biomarkers, Years_bl, etc.
csv_merge = path_to_data / 'ADNIMERGE_08Nov2023.csv'
df_merge  = pd.read_csv(csv_merge,low_memory=False)

# Choose your PET poison
tau_pet_csv = path_to_data / 'UCBERKELEY_TAU_6MM_06Nov2023.csv'
amy_pet_csv = path_to_data / 'UCBERKELEY_AMY_6MM_06Nov2023.csv'
fdg_pet_csv = path_to_data / 'FDG_wrangled_by_Neil.csv'

pet_csv = tau_pet_csv
df_pet  = pd.read_csv(pet_csv)
df_pet['Image ID'] = df_pet['LONIUID'].map(lambda x: x.replace('I','')).astype(int) # 'Image ID' for joining with IDA Search

# See ADNI_VISCODE2_Assignment_Methods.pdf (ADNI 2 Visit Codes Assignment Methods (PDF) under Enrollment on LONI IDA)
mapper_visit_to_viscode = {
    #'ADNI4 Initial Visit - Cont Pt': '',
    #'ADNI4 Baseline - New Pt': '',
    'ADNI Baseline':      'bl',
    'ADNI1/GO Month 6':  'm06',
    'ADNI1/GO Month 12': 'm12',
    'ADNI1/GO Month 24': 'm24', 
    'ADNI1/GO Month 36': 'm36',
    'ADNI1/GO Month 48': 'm48',
    'ADNI1/GO Month 18': 'm18',
    'ADNIGO Month 60':   'm60',
    'ADNI2 Initial Visit-Cont Pt': 'v06',
    'ADNI2 Baseline-New Pt': 'v03',
    'ADNI2 Year 1 Visit': 'v11',
    'ADNI2 Year 2 Visit': 'v21',
    'ADNI2 Year 3 Visit': 'v31',
    'ADNI2 Year 4 Visit': 'v41',
    'ADNI2 Year 5 Visit': 'v51',
    'ADNI2 Tau-only visit': 'tau',
    'ADNI3 Initial Visit-Cont Pt': 'init',
    'ADNI3 Year 1 Visit': 'y1',
    'ADNI3 Year 2 Visit': 'y2',
    'ADNI3 Year 4 Visit': 'y4'
}

# Make columns for merging
df_idaSearch['VISCODE']  = df_idaSearch['Visit'].map(mapper_visit_to_viscode)
df_idaSearch['VISCODE2'] = df_idaSearch['VISCODE'].map(mapper_viscode_viscode2)
missing = df_idaSearch['VISCODE'].isnull() | df_idaSearch['VISCODE2'].isnull()
df_idaSearch.loc[missing,'Visit']
print(f"FYI, missing {df_idaSearch['VISCODE'].isnull().sum()} VISCODEs and {df_idaSearch['VISCODE2'].isnull().sum()} VISCODE2s:")
print(',\n'.join(df_idaSearch.loc[missing,'Visit'].unique()))

# Merge 1: add VISCODE to PET
cols_ida = ['RID','Image ID','VISCODE2']
df_pet_ida = pd.merge(
    df_pet,
    df_idaSearch[cols_ida],
    on = ['RID','Image ID']
)
print(f"Merging PET (df_pet.shape: {df_pet.shape}) and IDASearch for VISCODE: df_pet_ida.shape = {df_pet_ida.shape}")

df_pet_ida_merge = pd.merge(
    df_pet_ida,
    df_merge,
    left_on = ['RID','VISCODE2'],
    right_on = ['RID','VISCODE']
)
print(f"Merging PET_IDA (df_pet_ida.shape: {df_pet_ida.shape}) and ADNIMERGE: df_pet_ida_merge.shape = {df_pet_ida_merge.shape}")

