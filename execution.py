import os
import feature_engineering
import model

if __name__ == "__main__":
    # folder path contains the original data
    data_dir = os.getcwd() + '/yelp_dataset'

    # run feature engineering
    df_business = feature_engineering.main(data_dir, save_data=True)

    # run model training
    model.main(df_business, tune_parameter=True)
