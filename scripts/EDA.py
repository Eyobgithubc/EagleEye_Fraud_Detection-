import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def univariate_analysis(df, columns):
    """
    Perform univariate analysis on specified columns in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    columns (list): List of column names to analyze.
    """
    for column in columns:
        plt.figure(figsize=(10, 5))
        
        # Histogram for distribution
        if df[column].dtype in ['int64', 'float64']:
            sns.histplot(df[column], bins=30, kde=True)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()
        
        # Summary statistics in a table
        summary_stats = df[column].describe().to_frame().reset_index()
        summary_stats.columns = ['Statistic', column]
        print(f'Summary statistics for {column}:')
        print(summary_stats, "\n")



def bivariate_analysis(df, x_col, y_col):
    """
    Perform bivariate analysis between two specified columns in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    x_col (str): The name of the first column (independent variable).
    y_col (str): The name of the second column (dependent variable).
    """
    plt.figure(figsize=(10, 6))
    
    # Scatter plot for bivariate relationship
    sns.scatterplot(data=df, x=x_col, y=y_col)
    plt.title(f'{x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()

    # Correlation heatmap if both columns are numeric
    if df[x_col].dtype in ['int64', 'float64'] and df[y_col].dtype in ['int64', 'float64']:
        correlation = df[[x_col, y_col]].corr()
        plt.figure(figsize=(5, 4))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', square=True)
        plt.title(f'Correlation Heatmap: {x_col} and {y_col}')
        plt.show()



def feature_engineering(df):
    """Perform feature engineering on the given DataFrame."""
    
    # Ensure purchase_time is in datetime format
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    # Extract hour of day and day of week
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.day_name()  # or use dt.dayofweek for numeric representation

    # Calculate transaction frequency (count of transactions per user)
    transaction_frequency = df.groupby('user_id')['purchase_time'].count().reset_index()
    transaction_frequency.columns = ['user_id', 'transaction_frequency']

    # Calculate transaction velocity (sum of purchase values per user)
    transaction_velocity = df.groupby('user_id')['purchase_value'].sum().reset_index()
    transaction_velocity.columns = ['user_id', 'transaction_velocity']

    # Merge frequency and velocity back to original DataFrame
    df = df.merge(transaction_frequency, on='user_id', how='left')
    df = df.merge(transaction_velocity, on='user_id', how='left')
    
    return df


def scale_features(df, columns_to_scale, method='min-max'):
    """Scale features in the DataFrame using the specified method."""
    if method == 'min-max':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Method should be 'min-max' or 'standard'")

    # Scale the specified columns
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    
    return df



