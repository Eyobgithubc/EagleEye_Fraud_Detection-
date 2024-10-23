import pandas as pd
import numpy as np
def convert_ip_to_int(ip_df):
    """Convert lower_bound_ip_address and upper_bound_ip_address to integer format."""
   
    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(float).astype(int)
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(float).astype(int)
    
    return ip_df

def merge_ip_country(fraud_df, ip_df):
    """Merge fraud data with IP address ranges based on IP address range matching."""
  
    fraud_df['ip_address'] = fraud_df['ip_address'].astype(int)
    
  
    merged_df = pd.merge_asof(
        fraud_df.sort_values('ip_address'), 
        ip_df.sort_values('lower_bound_ip_address'),
        left_on='ip_address',
        right_on='lower_bound_ip_address',
        direction='backward'
    )
    

    merged_df = merged_df[(merged_df['ip_address'] >= merged_df['lower_bound_ip_address']) & 
                          (merged_df['ip_address'] <= merged_df['upper_bound_ip_address'])]
    

    row_count = merged_df.shape[0]
    print(f"Number of rows after merging and filtering: {row_count}")
    
    return merged_df


def correct_data_types(df):
    # Convert signup_time and purchase_time to datetime
    df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
    df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')

    # Convert ip_address to integer (if applicable)
    df['ip_address'] = df['ip_address'].fillna(0).astype(int)

    # Ensure age and purchase_value are integers (if required)
    df['age'] = df['age'].fillna(0).astype(int)
    df['purchase_value'] = df['purchase_value'].fillna(0).astype(int)

    # Convert categorical features (device_id, source, browser, sex) to category types for optimization
    df['device_id'] = df['device_id'].astype('category')
    df['source'] = df['source'].astype('category')
    df['browser'] = df['browser'].astype('category')
    df['sex'] = df['sex'].astype('category')

    return df


def feature_engineering(df):
    # Calculate time-to-purchase in seconds
    df['time_to_purchase'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()

    # Extract time-based features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek

    # Transaction frequency and velocity
    # Transaction frequency: number of transactions per user
    df['transaction_count'] = df.groupby('user_id')['user_id'].transform('count')

    # Transaction velocity: purchase value divided by time to purchase
    df['transaction_velocity'] = df['purchase_value'] / df['time_to_purchase'].replace(0, np.nan)

    return df

from sklearn.preprocessing import LabelEncoder

def encode_label_features(df):
    le = LabelEncoder()
    for column in ['source', 'browser', 'sex', 'device_id','country']:
        df[column] = le.fit_transform(df[column])
    return df



def normalize_and_scale(df, feature_columns):
    """Normalizes and scales selected features."""
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df