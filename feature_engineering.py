import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from uszipcode import SearchEngine


########################################################################################################################
# LOAD DATA
########################################################################################################################
def load_dataset(data_dir, file):
    """
    load json dataset to dataframe
    """
    df = pd.DataFrame([])
    data_chunk = pd.read_json(os.path.join(data_dir, file), lines=True, chunksize=100000)
    for chunk in data_chunk:
          df = pd.concat([df, chunk])
    return df


########################################################################################################################
# ENRICH FEATURES FOR BUSINESS IN US
########################################################################################################################
def enrich_business_features(df):
    """
    enrich business in the US with population, population density,
    land area and median household income by searching with zipcode
    """
    # list of US state abbreviations
    states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
              "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
              "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
              "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
              "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
    # keep only businesses in US
    df_business_us = df[df['state'].isin(states)]
    # get rid of missing zipcode items
    df_with_zipcode = df_business_us[~(df_business_us['postal_code'] == '')]
    # activate search search engine which would provide more features relate to zipcode
    search = SearchEngine(simple_zipcode=True)
    df_zipcode = pd.DataFrame()
    zipcode = df_with_zipcode['postal_code'].unique()
    df_zipcode['postal_code'] = zipcode
    # perform searching and enrich features
    df_zipcode['population'] = [search.by_zipcode(i).to_dict()['population'] for i in tqdm(zipcode)]
    df_zipcode['population_density'] = [search.by_zipcode(i).to_dict()['population_density'] for i in tqdm(zipcode)]
    df_zipcode['land_area_in_sqmi'] = [search.by_zipcode(i).to_dict()['land_area_in_sqmi'] for i in tqdm(zipcode)]
    df_zipcode['median_household_income'] = [search.by_zipcode(i).to_dict()['median_household_income'] for i in
                                             tqdm(zipcode)]
    # merge features to business dataframe
    df = df_business_us.merge(df_zipcode, how='left', on='postal_code').dropna()
    print('Business features have been enriched! \n')

    return df


########################################################################################################################
# EXTRACT REVIEW FEATURES
########################################################################################################################
def review_features(df):
    """
    extract features from review data including: cool, funny, useful
    calculate review statistics of review per month and duration
    extract year and month of first and last time points
    """
    # get total number of 'cool', 'funny' and 'useful'
    df_review_statistic = df.groupby('business_id')[['cool', 'funny', 'useful']].sum()
    # get average stars
    df_review_statistic['avg_stars_review'] = df.groupby('business_id')[['stars']].mean()
    # group review timestamps into a list
    df_review_statistic['date'] = df.groupby('business_id')['date'].apply(list)
    review_date = df_review_statistic['date'].values
    # get number of reviews
    df_review_statistic['review_count'] = [len(dates) for dates in review_date]
    # extract 1st and last review year and month
    review_start = [min(i) for i in review_date]
    review_latest = [max(i) for i in review_date]
    df_review_statistic['review_start_year'] = list(map(lambda x: x.year, review_start))
    df_review_statistic['review_start_month'] = list(map(lambda x: x.month, review_start))
    df_review_statistic['review_latest_year'] = list(map(lambda x: x.year, review_latest))
    df_review_statistic['review_latest_month'] = list(map(lambda x: x.month, review_latest))
    # get duration of review in years
    review_duration = np.array(review_latest) - np.array(review_start)
    print('Calculating business duration according to review time...')
    df_review_statistic['review_duration'] = [item / pd.Timedelta(days=365.25) for item in tqdm(review_duration)]
    # get review rate per month
    df_review_statistic['review_per_month'] = df_review_statistic['review_count'] / df_review_statistic[
        'review_duration'] / 12
    # drop redundant columns
    df_review_statistic.drop(columns=['date', 'review_count'], inplace=True)
    print('Review features extracted! \n')

    return df_review_statistic


########################################################################################################################
# EXTRACT CHECKIN FEATURES
########################################################################################################################
def checkin_features(df):
    """
    extract check-in features of duration in year, counts and year and month of first and last time points
    """
    checkin = df['date'].values
    checkin_split = list(map(lambda x: x.split(','), checkin))
    # extract 1st and last checkin time
    checkin_start = np.array([pd.Timestamp(i[0]) for i in checkin_split])
    checkin_stop = np.array([pd.Timestamp(i[-1]) for i in checkin_split])
    # get duration of checkin
    checkin_duration = checkin_stop - checkin_start
    df['checkin_duration'] = [item / pd.Timedelta(days=365.25) for item in checkin_duration]
    # extract 1st and last checkin year and month
    df['checkin_start_year'] = list(map(lambda x: x.year, checkin_start))
    df['checkin_start_month'] = list(map(lambda x: x.month, checkin_start))
    df['checkin_latest_year'] = list(map(lambda x: x.year, checkin_stop))
    df['checkin_latest_month'] = list(map(lambda x: x.month, checkin_stop))
    # get number of checkins
    df['checkin_count'] = list(map(lambda x: len(x), checkin_split))
    # calculate checkins per month
    df['checkin_per_month'] = df['checkin_count'] / df['checkin_duration'] / 12
    df.drop(columns=['date'], inplace=True)
    print('Checkin features have been extracted! \n')

    return df


########################################################################################################################
# CLUSTERING BUSINESS WITH GEO-COORDINATES
########################################################################################################################
def business_clustering(df):
    """
    DBSCAN - Density-Based Spatial Clustering of Applications with Noise
    cluster the business based on the minimum distance between two businesses
    """
    coords = df[['latitude', 'longitude']].values
    # radius of the earth in km
    kms_per_radian = 6371
    # the minimus distance between two business is 20m
    epsilon = .05 / kms_per_radian
    # perform clustering
    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
    df['coords_cluster_label'] = cluster_labels
    # number of business in each cluster
    counts_cluster = [len(cluster) for cluster in clusters]
    df['neighbors'] = [counts_cluster[i] for i in cluster_labels]
    # if the number of neighbors > 20, the business in a chain
    df['is_chain'] = df['neighbors'] >= 5
    df['is_chain'] = df['is_chain'].astype(int)
    print('Businesses are clustered into {} groups. \n'.format(num_clusters))

    return df


########################################################################################################################
def main(data_dir, save_data=False):
    """
    execute feature engineering
    """
    # Load data
    print('Loading business data ...')
    df_business = load_dataset(data_dir, 'business.json')
    print('Loading checkin data ...')
    df_checkin = load_dataset(data_dir, 'checkin.json')
    print('Loading review data ...')
    df_review = load_dataset(data_dir, 'review.json')
    print('All datasets have been loaded! \n')

    # Enrich business features
    print("Enriching business features ...")
    df_business = enrich_business_features(df_business)

    # Extract features from review and merge into business
    print('Extracting review features ...')
    df_review_statistic = review_features(df_review)
    df_business = df_business.merge(df_review_statistic, on='business_id', how='inner')

    # Extract features from checkin and merge into business
    print('Extracting checkin features...')
    df_checkin = checkin_features(df_checkin)
    df_business = df_business.merge(df_checkin, on='business_id', how='inner')

    # Cluster business with Geo-coordinates
    print("Clustering business according to Geo-coordinates ...")
    df_business = business_clustering(df_business)

    print('Features are ready!')

    # save business features into the folder contains original datasets
    if save_data:
        df_business.to_pickle(os.path.join(data_dir, 'features.pkl'))
        print('Features are saved as {}'.format(os.path.join(data_dir, 'features.pkl')))

    return df_business
