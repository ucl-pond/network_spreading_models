# Tiantian He, UCL CMICHACKS23, 9 November 2023
# a class to input raw csv files of abeta, tau, adnimerge, output the split of subject according to abeta + tau positivity status
# tau positivity status is determined by Gaussian Mixture Model

import pandas as pd
from sklearn.mixture import GaussianMixture

class SubjectGrouper:
    def __init__(self, tau_path, amyloid_path, info_path, ratio):
        self.tau_path = tau_path
        self.amyloid_path = amyloid_path
        self.info_path = info_path
        self.data = None
        self.ratio = ratio

    @staticmethod
    def create_merge_id(df, rid_col, viscode_col):
        df['merge_id'] = df[rid_col].astype(str) + '_' + df[viscode_col].astype(str)
        return df

    @staticmethod
    def GaussianMixtureTau(tau_df, region_names_df, ratio):
            # Transform the region names into the column names used in the tau dataset
            region_columns = (region_names_df['region'].str.upper() + '_SUVR').tolist()
            # Initialize the DataFrame to store tau status for each subject
            tau_status_df = pd.DataFrame(index=tau_df.index)
            # drop nan in region_columns of tau_df
            tau_df = tau_df.dropna(subset=region_columns)

            for region in region_columns:
                # Reshape the data for the GMM (it expects a 2D array)
                region_data = tau_df[region].values.reshape(-1, 1)

                # Fit a GMM with 2 components
                gmm = GaussianMixture(n_components=2, random_state=0)
                gmm.fit(region_data)

                # Identify the 'negative' distribution (the one with the smaller mean)
                negative_component_index = gmm.means_.argmin()
                negative_mean = gmm.means_[negative_component_index][0]
                negative_std = (gmm.covariances_[negative_component_index][0] ** 0.5)[0]

                # Calculate the threshold for positivity
                threshold = negative_mean + negative_std

                # Determine positivity for the region
                tau_status_df[region] = tau_df[region] > threshold

            # Calculate the total number of positive regions for each subject
            tau_status_df['positive_regions'] = tau_status_df.sum(axis=1)

            # Determine overall tau status based on the specified ratio
            total_regions = len(region_columns)
            tau_status_df['tau_status'] = tau_status_df['positive_regions'] >= (total_regions * ratio)

            # Convert boolean status to '+' or '-'
            tau_df['tau_status'] = tau_status_df['tau_status'].apply(lambda x: '+' if x else '-')

            return tau_df

    def read_and_prepare(self):
        # Read the data
        # IMPORTANT NOTICE: for subject with multiple visits, use the latest visit as splitting criteria
        tau_df = pd.read_csv(self.tau_path)
        amyloid_df = pd.read_csv(self.amyloid_path)
        info_df = pd.read_csv(self.info_path).drop_duplicates(subset=['RID'], keep='last')
        region_names_df = pd.read_csv('./data/ADNI/fs_name.csv')
        region_columns = (region_names_df['region'].str.upper() + '_SUVR').tolist()

        tau_df = tau_df.drop_duplicates(subset=['RID'], keep='last')
        tau_df = tau_df[list(tau_df.columns[:14])+region_columns + ['VISCODE2']]

        amyloid_df = amyloid_df.drop_duplicates(subset=['RID'], keep='last')
        amyloid_df = amyloid_df[list(amyloid_df.columns[:14])+region_columns+ ['VISCODE2']]

        region_names_df = pd.read_csv('./data/ADNI/fs_name.csv')

        # Create merge_id for each dataframe
        tau_df = self.create_merge_id(tau_df, 'RID', 'VISCODE2')
        amyloid_df = self.create_merge_id(amyloid_df, 'RID', 'VISCODE2')
        info_df = self.create_merge_id(info_df, 'RID', 'VISCODE')

        # Determine abeta_status based on WH_1.11CUTOFF
        amyloid_df['abeta_status'] = amyloid_df['AMYLOID_STATUS'].apply(lambda x: '+' if x == 1 else '-')

        # Run Gaussian Mixture Model for tau_status
        tau_df = self.GaussianMixtureTau(tau_df,region_names_df, self.ratio)

        # # more complex way: Merge on merge_id
        # merged_df = tau_df.merge(amyloid_df, on='merge_id', how='inner')
        # merged_df = merged_df.merge(info_df, on='merge_id', how='inner')

        # Merge on RID: since we only use cross-section data at the latest visit for splitting for now, we can simply use RID
        merged_df = tau_df.merge(amyloid_df, on='RID', how='inner')
        merged_df = merged_df.merge(info_df, on='RID', how='inner')
        self.data = merged_df

    def split_subjects(self):
        if self.data is None:
            raise ValueError("Data not loaded. Please run 'read_and_prepare' first.")

        group_1 = self.data[(self.data['abeta_status'] == '+') & (self.data['tau_status'] == '+')]
        group_2 = self.data[(self.data['abeta_status'] == '+') & (self.data['tau_status'] == '-')]
        group_3 = self.data[(self.data['abeta_status'] == '-') & (self.data['tau_status'] == '+')]
        group_4 = self.data[(self.data['abeta_status'] == '-') & (self.data['tau_status'] == '-')]

        RID_index =  {
            'ab+_tau+': list(group_1['RID']),
            'ab+_tau-': list(group_2['RID']),
            'ab-_tau+': list(group_3['RID']),
            'ab-_tau-': list(group_4['RID']),
        }
        return RID_index


if __name__ == '__main__':
    data_dir = './data/ADNI/'
    tau_file = data_dir+'df_tau_ida.csv'
    amyloid_file = data_dir+'df_amy_ida.csv'
    info_file = data_dir+'ADNIMERGE_08Nov2023.csv'

    grouper = SubjectGrouper(tau_file, amyloid_file, info_file, ratio=0.2)
    grouper.read_and_prepare()
    groups = grouper.split_subjects()
    print(groups)
    
    ## now include all the visits
    tau_long_df = pd.read_csv(tau_file)
    amyloid_long_df = pd.read_csv(amyloid_file)
    region_names_df = pd.read_csv('./data/ADNI/fs_name.csv')
    region_columns = (region_names_df['region'].str.upper() + '_SUVR').tolist()

    tau_long_df = tau_long_df[list(tau_long_df.columns[:14])+region_columns + ['VISCODE2']]
    amyloid_long_df = amyloid_long_df[list(amyloid_long_df.columns[:14])+region_columns+ ['VISCODE2']]

    # filter out abeta+ tau+ 
    idx = groups['ab+_tau+']
    tau_long_df = tau_long_df[tau_long_df['RID'].isin(idx)].iloc[:,1:]
    # move VISCODE2 to the first column of tau_long_df
    cols = tau_long_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    tau_long_df = tau_long_df[cols]
    tau_long_df
    