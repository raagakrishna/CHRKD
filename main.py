import sys
import numpy as np
import pandas as pd

# print(sys.version)


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
    data = pd.read_csv(filename, parse_dates=['TX_TS', 'CARD_EXPIRY_DATE'], dtype=dtype_dict)
    print(type(data))
    print(data.columns)
    print(data.shape)

    # Get names of columns with missing values
    missing_val_count_by_column = (data.isnull().sum())
    cols_with_missing = missing_val_count_by_column[missing_val_count_by_column > 0].index.tolist()
    data_in_cols_missing = missing_val_count_by_column[missing_val_count_by_column > 0].values.tolist()
    print("Columns with number of missing data")
    print(missing_val_count_by_column[missing_val_count_by_column > 0])

    # Removing missing values
    # data_clean = data.drop(cols_with_missing, axis=1)
    # print(data_clean.columns)
    data_clean = data

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

    # Select target
    y = train_data.TX_FRAUD
    # print(y)

    # Removing the target
    predictors = train_data.drop(['TX_FRAUD'], axis=1)
    # print(predictors)
    print(predictors.columns)


def load_and_clean_transactions_test_data(test_filename):
    """
    Loads and cleans the transaction test data
    :param test_filename: the filename with the transaction test data
    :return: the dataset
    """
    test_data = clean_transactions_data(test_filename)
    print(test_data.columns)

    # TODO: MERCHANT_ID??


def load_and_clean_customers_data(customers_filename):
    """
    Loads and cleans the customer data
    :param customers_filename: the filename with the customer data
    :return: the dataset
    """
    clean_customers_data(customers_filename)


def load_and_clean_merchants_data(merchants_filename):
    """
    Loads and cleans the merchants data
    :param merchants_filename: the filename with the merchants data
    :return: the dataset
    """
    clean_merchants_data(merchants_filename)


def load_and_clean_terminals_data(terminals_filename):
    """
    Loads and cleans the terminals data
    :param terminals_filename: the filename with the terminals data
    :return: the dataset
    """
    clean_terminals_data(terminals_filename)


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
            load_function(filename)
            break
    else:
        raise Exception("Sorry, I do not know how to clean this dataset.")


if __name__ == '__main__':
    data_filename = 'data/terminals.csv'
    load_and_clean_data(data_filename)

    pass



