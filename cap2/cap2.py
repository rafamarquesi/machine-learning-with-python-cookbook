
if __name__ == '__main__':
    # 2.1
    # Loard scikit-learn's dataset
    from sklearn import datasets

    # Load digits dataset
    digits = datasets.load_digits()

    # Create features matrix
    features = digits.data

    # Create target vector
    target = digits.target

    # View first observation
    print(features[0])

    # 2.2
    # Load Library
    from sklearn.datasets import make_regression

    # Generate features matrix, target vector, and the true coefficients
    features, target, coefficients = make_regression(
        n_samples=100, n_features=3, n_informative=3, n_targets=1,noise=0.0, coef=True, random_state=1
    )

    # View feature matrix and target vector
    print('Feature Matrix\n', features[:3])
    print('Target Vector\n', target[:3])

    # Load Library
    from sklearn.datasets import make_classification

    # Generate features matrix and target vector
    features, target = make_classification(
        n_samples=100, n_features=3, n_informative=3, n_redundant=0, n_classes=2, weights=[.25,.75], random_state=1
    )

    # View feature matrix and target vector
    print('Feature Matrix\n', features[:3])
    print('Target Vector\n', target[:3])

    # Load Library
    from sklearn.datasets import make_blobs

    # Generate feature matrix and target vector
    features, target = make_blobs(
        n_samples=100, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=1
    )

    # View feature matrix and target vector
    print('Feature Matrix\n', features[:3])
    print('Target Vector\n', target[:3])

    # Load library
    import matplotlib.pyplot as plt

    # View scatterplot
    plt.scatter(features[:,0], features[:,1], c=target)
    plt.show()

    # 2.3
    # Load library
    import pandas as pd

    # Create URL
    url = 'https://www1.ncdc.noaa.gov/pub/data/cdo/samples/PRECIP_HLY_sample_csv.csv'

    # Load dataset
    dataframe = pd.read_csv(url)

    # View first two rows
    print(dataframe.head(2))

    # 2.4
    # Load library
    import pandas as pd

    # Create URL
    url = 'https://www.sample-videos.com/xls/Sample-Spreadsheet-10-rows.xls'

    # Load data
    dataframe = pd.read_excel(url, sheet_name=0, header=None)

    # View the first two rows
    print(dataframe.head(2))

    # 2.5
    # Load library
    import pandas as pd

    # Create URL
    url = 'https://api.github.com/repositories'

    # Load data
    dataframe = pd.read_json(url, orient='columns')

    # View the first two rows
    print(dataframe.head(2))

    # 2.6
    # Load libraries
    import pandas as pd
    import sqlite3
    # from sqlalchemy import create_engine

    # Create a connection to the database
    # database_connection = create_engine('sqlite:///sample.db')
    # Create DB with examples -> https://sparkbyexamples.com/pandas/pandas-read-sql-query-or-table/
    database_connection = sqlite3.connect('courses_database')

    # Load data
    dataframe = pd.read_sql_query('SELECT * FROM COURSES', database_connection)

    # View first two rows
    print(dataframe.head(2))