import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


# print(sys.version)


def encode_categorical_vars(dataset):
    # Get list of categorical variables
    s = (dataset.dtypes == 'object')
    object_cols = list(s[s].index)

    # print("Categorical variables:")
    # print(object_cols)

    # Apply ordinal encoder to each column with categorical data
    ordinal_encoder = OrdinalEncoder()
    dataset[object_cols] = ordinal_encoder.fit_transform(dataset[object_cols])

    return dataset


def clean_transactions_data(filename):
    """
    Cleans the transaction data
    :param filename: the filename with the transaction data
    :return: cleaned dataset
    """
    dtype_dict = {
        'TX_ID': str,
        'TX_TS': str,
        'CUSTOMER_ID': int,
        'TERMINAL_ID': int,
        'TX_AMOUNT': float,
        'TX_FRAUD': int,
        'TRANSACTION_GOODS_AND_SERVICES_AMOUNT': float,
        'TRANSACTION_CASHBACK_AMOUNT': float,
        'CARD_EXPIRY_DATE': str,
        'CARD_DATA': str,
        'CARD_BRAND': str,
        'TRANSACTION_TYPE': str,
        'TRANSACTION_STATUS': str,
        'FAILURE_CODE': str,
        'FAILURE_REASON': str,
        'TRANSACTION_CURRENCY': str,
        'CARD_COUNTRY_CODE': str,
        'MERCHANT_ID': str,
        'IS_RECURRING_TRANSACTION': bool,
        'ACQUIRER_ID': str,
        'CARDHOLDER_AUTH_METHOD': str,
        'ID_JOIN': float
    }

    # Load the data
    data = pd.read_csv(filename, dtype=dtype_dict)

    # Convert 'TX_TS' column to datetime
    data['TX_TS'] = pd.to_datetime(data['TX_TS'])

    # Extract and create new columns for day of the week, day, month, year, and time
    data['TX_DAY_OF_WEEK'] = data['TX_TS'].dt.dayofweek
    data['TX_DAY'] = data['TX_TS'].dt.day
    data['TX_MONTH'] = data['TX_TS'].dt.month
    data['TX_YEAR'] = data['TX_TS'].dt.year
    data['TX_TIME_SECONDS'] = data['TX_TS'].dt.hour * 3600 + data['TX_TS'].dt.minute * 60 + data['TX_TS'].dt.second

    # Drop the original 'TX_TS' column
    data.drop(columns=['TX_TS'], inplace=True)

    # Split the 'CARD_EXPIRY_DATE' into month and year components
    data['CARD_EXPIRY_MONTH'] = data['CARD_EXPIRY_DATE'].str.split('/').str[0].astype(int)
    data['CARD_EXPIRY_YEAR'] = data['CARD_EXPIRY_DATE'].str.split('/').str[1].astype(int)

    # Drop the original 'CARD_EXPIRY_DATE' column
    data.drop(columns=['CARD_EXPIRY_DATE'], inplace=True)

    # print(type(data))
    # print(data.columns)
    # print(data.shape)

    # Get names of columns with missing values
    missing_val_count_by_column = (data.isnull().sum())
    cols_with_missing = missing_val_count_by_column[missing_val_count_by_column > 0].index.tolist()
    data_in_cols_missing = missing_val_count_by_column[missing_val_count_by_column > 0].values.tolist()
    # print("Columns with number of missing data")
    # print(missing_val_count_by_column[missing_val_count_by_column > 0])

    # Removing missing values
    data_clean = data.drop(cols_with_missing, axis=1)
    # print(data_clean.columns)

    return data_clean


def clean_customers_data(filename):
    """
    Cleans the customer data
    :param filename: the filename with the customer data
    :return: cleaned dataset
    """
    dtype_dict = {
        'CUSTOMER_ID': str,
        'x_customer_id': float,
        'y_customer_id': float
    }

    # Load the data
    data = pd.read_csv(filename, dtype=dtype_dict)
    # print(data.columns)
    # print(data.shape)

    # Get names of columns with missing values
    missing_val_count_by_column = (data.isnull().sum())
    cols_with_missing = missing_val_count_by_column[missing_val_count_by_column > 0].index.tolist()
    # data_in_cols_missing = missing_val_count_by_column[missing_val_count_by_column > 0].values.tolist()
    # print("Columns with number of missing data")
    # print(missing_val_count_by_column[missing_val_count_by_column > 0])

    # Removing missing values
    data_clean = data.drop(cols_with_missing, axis=1)
    # print(data_clean.columns)

    return data_clean


def clean_merchants_data(filename):
    """
    Cleans the merchants data
    :param filename: the filename with the merchants data
    :return: cleaned dataset
    """
    dtype_dict = {
        'MERCHANT_ID': str,
        'BUSINESS_TYPE': str,
        'MCC_CODE': int,
        'LEGAL_NAME': str,
        'FOUNDATION_DATE': str,
        'TAX_EXCEMPT_INDICATOR': bool,
        'OUTLET_TYPE': str,
        'ACTIVE_FROM': str,
        'TRADING_FROM': str,
        'ANNUAL_TURNOVER_CARD': int,
        'ANNUAL_TURNOVER': int,
        'AVERAGE_TICKET_SALE_AMOUNT': float,
        'PAYMENT_PERCENTAGE_FACE_TO_FACE': int,
        'PAYMENT_PERCENTAGE_ECOM': int,
        'PAYMENT_PERCENTAGE_MOTO': int,
        'DEPOSIT_REQUIRED_PERCENTAGE': int,
        'DEPOSIT_PERCENTAGE': int,
        'DELIVERY_SAME_DAYS_PERCENTAGE': int,
        'DELIVERY_WEEK_ONE_PERCENTAGE': int,
        'DELIVERY_WEEK_TWO_PERCENTAGE': int,
        'DELIVERY_OVER_TWO_WEEKS_PERCENTAGE': int
    }

    # Load the data
    data = pd.read_csv(filename, parse_dates=['FOUNDATION_DATE', 'ACTIVE_FROM', 'TRADING_FROM'], dtype=dtype_dict)
    # print(data.columns)
    # print(data.shape)

    # Get names of columns with missing values
    missing_val_count_by_column = (data.isnull().sum())
    cols_with_missing = missing_val_count_by_column[missing_val_count_by_column > 0].index.tolist()
    # data_in_cols_missing = missing_val_count_by_column[missing_val_count_by_column > 0].values.tolist()
    # print("Columns with number of missing data")
    # print(missing_val_count_by_column[missing_val_count_by_column > 0])

    # Removing missing values
    data_clean = data.drop(cols_with_missing, axis=1)
    # print(data_clean.columns)

    return data_clean


def clean_terminals_data(filename):
    """
    Cleans the terminals data
    :param filename: the filename with the terminals data
    :return: cleaned dataset
    """
    dtype_dict = {
        'TERMINAL_ID': str,
        'x_terminal_id': float,
        'y_terminal__id': float
    }

    # Load the data
    data = pd.read_csv(filename, dtype=dtype_dict)
    # print(data.columns)
    # print(data.shape)

    # Get names of columns with missing values
    missing_val_count_by_column = (data.isnull().sum())
    cols_with_missing = missing_val_count_by_column[missing_val_count_by_column > 0].index.tolist()
    # data_in_cols_missing = missing_val_count_by_column[missing_val_count_by_column > 0].values.tolist()
    # print("Columns with number of missing data")
    # print(missing_val_count_by_column[missing_val_count_by_column > 0])

    # Removing missing values
    data_clean = data.drop(cols_with_missing, axis=1)
    # print(data_clean.columns)

    return data_clean


def load_and_clean_transactions_train_data(train_filename):
    """
    Loads and cleans the transaction train data
    :param train_filename: the filename with the transaction train data
    :return: the dataset
    """
    train_data = clean_transactions_data(train_filename)

    train_data = encode_categorical_vars(train_data)

    # Select target
    y = train_data.TX_FRAUD
    # print(y)

    # Removing the target
    predictors = train_data.drop(['TX_FRAUD'], axis=1)
    # print(predictors)
    print(predictors.columns)

    return y, predictors


def load_and_clean_transactions_test_data(test_filename):
    """
    Loads and cleans the transaction test data
    :param test_filename: the filename with the transaction test data
    :return: the dataset
    """
    test_data = clean_transactions_data(test_filename)
    # print(test_data.columns)

    train_data = encode_categorical_vars(test_data)

    # TODO: MERCHANT_ID??

    return train_data


def load_and_clean_customers_data(customers_filename):
    """
    Loads and cleans the customer data
    :param customers_filename: the filename with the customer data
    :return: the dataset
    """
    data = clean_customers_data(customers_filename)

    data = encode_categorical_vars(data)

    return data


def load_and_clean_merchants_data(merchants_filename):
    """
    Loads and cleans the merchants data
    :param merchants_filename: the filename with the merchants data
    :return: the dataset
    """
    data = clean_merchants_data(merchants_filename)

    data = encode_categorical_vars(data)

    return data


def load_and_clean_terminals_data(terminals_filename):
    """
    Loads and cleans the terminals data
    :param terminals_filename: the filename with the terminals data
    :return: the dataset
    """
    data = clean_terminals_data(terminals_filename)

    data = encode_categorical_vars(data)

    return data


def load_and_clean_data(filename):
    # Define a dictionary to map keywords to functions
    load_functions = {
        'merchants': load_and_clean_merchants_data,
        'terminals': load_and_clean_terminals_data,
        'customers': load_and_clean_customers_data,
        'transactions_train': load_and_clean_transactions_train_data,
        'transactions_test': load_and_clean_transactions_test_data
    }

    # Get the keyword from the filename
    for keyword, load_function in load_functions.items():
        if keyword in filename:
            return load_function(filename)
    else:
        raise Exception("Sorry, I do not know how to clean this dataset.")


def define_model():
    # Decision Tree
    my_model = DecisionTreeRegressor(max_leaf_nodes=5000, random_state=1)
    # play with the max_leaf_nodes (choose one that gives the lowest error [5, 50, 500, 5000]

    # Random forest
    # my_model = RandomForestRegressor(n_estimators=10, random_state=0)

    # XGBoost
    # n_estimators (too low = underfitting, too high = overfitting)
    # learning_rate (smaller = more accurate)
    # n_jobs (number of cores to use)
    # my_model = XGBRegressor(n_estimators=100, learning_rate=0.05, n_jobs=8)

    return my_model


def test_model(cols_to_use, model):
    # Test dataset
    data_filename = 'data/transactions_test.csv'
    data_test_original = load_and_clean_data(data_filename)
    print(data_test_original.columns)

    # Extracting columns model needs
    data_test = data_test_original[cols_to_use]
    print(data_test.columns)

    # Predicting
    y_ans = model.predict(data_test)

    # Save data to csv file
    predictions_df = pd.DataFrame(y_ans, columns=['TX_FRAUD'])
    predictions_df.insert(0, 'TX_ID', pd.read_csv(data_filename)['TX_ID'])
    predictions_df.to_csv('data/submission.csv', index=False)


def detect_fraud():
    # """
    data_filename = 'data/transactions_train.csv'
    data = load_and_clean_data(data_filename)
    print(len(data))

    # return

    # Separating the target variable
    y, X = data
    if len(data) == 2:
        y = data[0]
        X = data[1]

    print(y.shape)
    print(X.shape)

    print(X.iloc[0])

    # Not using: 'TX_ID', 'CUSTOMER_ID','MERCHANT_ID',
    cols_to_use = ['TERMINAL_ID', 'TX_AMOUNT', 'TRANSACTION_GOODS_AND_SERVICES_AMOUNT', 'TRANSACTION_CASHBACK_AMOUNT',
                   'CARD_DATA', 'CARD_BRAND', 'TRANSACTION_TYPE', 'TRANSACTION_STATUS',
                   'TRANSACTION_CURRENCY', 'CARD_COUNTRY_CODE', 'IS_RECURRING_TRANSACTION',
                   'ACQUIRER_ID', 'CARDHOLDER_AUTH_METHOD',
                   'TX_DAY_OF_WEEK', 'TX_DAY', 'TX_MONTH', 'TX_YEAR', 'TX_TIME_SECONDS',
                   'CARD_EXPIRY_MONTH', 'CARD_EXPIRY_YEAR']
    X = X[cols_to_use]
    # print("")
    # print(X.head())

    # print(X_test)

    # Define the model
    model = define_model()

    MAEs = []
    for i in range(0, 10):
        print('Run number ' + str(i))
        # Splitting the dataset into train and test
        # TODO: change random_state from 0 to 100 later
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=i)
        # print(X_train.iloc[0])

        # Fit the model
        print('Fitting model')
        model.fit(X_train, y_train)

        # Predict the model
        print('Predicting model')
        preds = model.predict(X_test)

        # Evaluate the model
        MAE = mean_absolute_error(y_test, preds)
        print("MAE: " + str(MAE))

        MAEs.append(MAE)

    print('MAEs')
    print(MAEs)

    print("TEST MODEL")
    test_model(cols_to_use, model)


if __name__ == '__main__':
    detect_fraud()

    pass



