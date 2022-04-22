if __name__ == '__main__':
    # 3.0
    # Load library
    import pandas as pd

    # Create URL
    url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

    # Load data as a dataframe
    dataframe = pd.read_csv(url)

    # Show first 5 rows
    print(dataframe.head(5))

    # 3.1
    # Load library
    import pandas as pd

    # Create DataFrame
    dataframe = pd.DataFrame()

    # Add columns
    dataframe['Name'] = ['Jacky Jackson', 'Steven Stevenson']
    dataframe['Age'] = [38, 25]
    dataframe['Driver'] = [True, False]

    # Show DataFrame
    print(dataframe)

    # Create row
    new_person = pd.Series(['Molly Mooney', 40, True], index=['Name', 'Age', 'Driver'])

    # Append row
    print(dataframe.append(new_person, ignore_index=True))

    # 3.2
    # Load library
    import pandas as pd

    # Create URL
    url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

    # Load data
    dataframe = pd.read_csv(url)

    # Show two rows
    print(dataframe.head(2))

    # Show dimensions
    print(dataframe.shape)

    # Show statistics
    print(dataframe.describe())

    # 3.3
    # Load library
    import pandas as pd

    # Create URL
    url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

    # Load data
    dataframe = pd.read_csv(url)

    # Select first row
    print(dataframe.iloc[0])

    # Select three rows
    print(dataframe.iloc[1:4])

    # Select three rows
    print(dataframe.iloc[:4])

    # Set index
    dataframe = dataframe.set_index(dataframe['Name'])

    # Show row
    print(dataframe.loc['Allen, Miss Elisabeth Walton'])

    # 3.4
    # Load library
    import pandas as pd

    # Create URL
    url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

    # Load data
    dataframe = pd.read_csv(url)

    # Show top two rows where column 'sex' is 'female'
    print(dataframe[dataframe['Sex'] == 'female'].head(2))

    # Filter rows
    print(dataframe[(dataframe['Sex'] == 'female') & (dataframe['Age'] >= 65)])

    # 3.5
    # Load library
    import pandas as pd

    # Create URL
    url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

    # Load data
    dataframe = pd.read_csv(url)

    # Replace values, show two rows
    print(dataframe['Sex'].replace('female', 'Woman').head(2))

    # Replace "female" and "male" with "Woman" and "Man"
    print(dataframe['Sex'].replace(['female', 'male'], ['Woman', 'Man']).head(5))

    # Replace values, show two rows
    print(dataframe.replace(1, 'One').head(2))

    # Replace values, show two rows
    print(dataframe.replace(r'1st', 'First', regex=True).head(2))

    # 3.6
    # Load library
    import pandas as pd

    # Create URL
    url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

    # Load data
    dataframe = pd.read_csv(url)

    # Rename column, show two rows
    print(dataframe.rename(columns={'PClass': 'Passenger Class'}).head(2))

    # Rename columns, show two rows
    print(dataframe.rename(columns={'PClass': 'Passenger Class', 'Sex': 'Gender'}).head(2))

    # Load library
    import collections

    # Create dictionary
    column_names = collections.defaultdict(str)

    # Create keys
    for name in dataframe.columns:
        column_names[name]

    # Show dictionary
    print(column_names)

    # 3.7
    # Load library
    import pandas as pd

    # Create URL
    url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

    # Load data
    dataframe = pd.read_csv(url)

    # Calculate statistics
    print('Maximum:', dataframe['Age'].max())
    print('Minimum:', dataframe['Age'].min())
    print('Mean:', dataframe['Age'].mean())
    print('Sum:', dataframe['Age'].sum())
    print('Count:', dataframe['Age'].count())

    # Show counts
    print(dataframe.count())

    # 3.8
    # Load library
    import pandas as pd

    # Create URL
    url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

    # Load data
    dataframe = pd.read_csv(url)

    # Select unique values
    print(dataframe['Sex'].unique())

    # Show counts
    print(dataframe['Sex'].value_counts())

    # Show counts
    print(dataframe['PClass'].value_counts())

    # Show number of unique values
    print(dataframe['PClass'].nunique())

    # 3.9
    # Load library
    import pandas as pd

    # Create URL
    url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

    # Load data
    dataframe = pd.read_csv(url)

    # Select missing values, show two rows
    print(dataframe[dataframe['Age'].isnull()].head(2))

    # Attempt to replace values with NaN
    # dataframe['Sex'] = dataframe['Sex'].replace('male', NaN)

    # Load library
    import numpy as np

    # Replace values with NaN
    dataframe['Sex'] = dataframe['Sex'].replace('male', np.nan)

    # Load data, set missing values
    dataframe = pd.read_csv(url, na_values=[np.nan, 'NONE', -999])

    # 3.10
    # Load library
    import pandas as pd

    # Create URL
    url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

    # Load data
    dataframe = pd.read_csv(url)

    # Delete column
    print(dataframe.drop('Age', axis=1).head(2))

    # Drop columns
    print(dataframe.drop(['Age', 'Sex'], axis=1).head(2))

    # Drop columns
    print(dataframe.drop(dataframe.columns[1], axis=1).head(2))

    # Create a new DataFrame
    dataframe_name_dropped = dataframe.drop(dataframe.columns[0], axis=1)

    # 3.11
    # Load library
    import pandas as pd

    # Create URL
    url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

    # Load data
    dataframe = pd.read_csv(url)

    # Delete rows, show first two rows of output
    print(dataframe[dataframe['Sex'] != 'male'].head(2))

    # Delete row, show first two rows of output
    print(dataframe[dataframe['Name'] != 'Allison, Miss Helen Loraine'].head(2))

    # Delete row, show first two rows of output
    print(dataframe[dataframe.index != 0].head(2))

    # 3.12
    # Load library
    import pandas as pd

    # Create URL
    url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

    # Load data
    dataframe = pd.read_csv(url)

    # Drop duplicates, show first two rows of output
    print(dataframe.drop_duplicates().head(2))

    # Show number of rows
    print('Number Of Rows In The Original DataFrame:', len(dataframe))
    print('Number Of Rows After Deduping:', len(dataframe.drop_duplicates()))

    # Drop duplicates
    print(dataframe.drop_duplicates(subset=['Sex']))

    # Drop duplicates
    print(dataframe.drop_duplicates(subset=['Sex'], keep='last'))

    # 3.13
    # Load library
    import pandas as pd

    # Create URL
    url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

    # Load data
    dataframe = pd.read_csv(url)

    # Group rows by the values of the column 'Sex', calculate mean of each group
    print(dataframe.groupby('Sex').mean())

    # Group rows
    print(dataframe.groupby('Sex'))

    # Group rows, count rows
    print(dataframe.groupby('Survived')['Name'].count())

    # Group rows, calculate mean
    print(dataframe.groupby(['Sex', 'Survived'])['Age'].mean())

    # 3.14
    # Loard libraries
    import pandas as pd
    import numpy as np

    # Create date range
    time_index = pd.date_range('06/06/2017', periods=100000, freq='30S')

    # Create DataFrame
    dataframe = pd.DataFrame(index=time_index)

    # Create column of random values
    dataframe['Sale_Amount'] = np.random.randint(1, 10, 100000)

    # Group row by week, calculate sum per week
    print(dataframe.resample('W').sum())

    # Show three rows
    print(dataframe.head(3))

    # Group by two weeks, calculate mean
    print(dataframe.resample('2W').mean())

    # Group by month, count rows
    print(dataframe.resample('M').count())

    # Group by month, count rows
    print(dataframe.resample('M', label='left').count())

    # 3.15
    # Load library
    import pandas as pd

    # Create URL
    url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

    # Load data
    dataframe = pd.read_csv(url)

    # Print first two names uppercased
    for name in dataframe['Name'][0:2]:
        print(name.upper())

    # Show first two names uppercased
    print([name.upper() for name in dataframe['Name'][0:2]])

    # 3.16
    # Load library
    import pandas as pd

    # Create URL
    url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

    # Load data
    dataframe = pd.read_csv(url)


    # Create function
    def uppercase(x):
        return x.upper()


    # Apply function, show two rows
    print(dataframe['Name'].apply(uppercase)[0:2])

    # 3.17
    # Load library
    import pandas as pd

    # Create URL
    url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

    # Load data
    dataframe = pd.read_csv(url)

    # Group rows, apply function to groups
    print(dataframe.groupby('Sex').apply(lambda x: x.count()))

    # 3.18
    # Load library
    import pandas as pd

    # Create DataFrame
    data_a = {
        'id': ['1', '2', '3'],
        'first': ['Alex', 'Amy', 'Allen'],
        'last': ['Anderson', 'Ackerman', 'Ali']
    }
    dataframe_a = pd.DataFrame(data_a, columns=['id', 'first', 'last'])

    # Create DataFrame
    data_b = {
        'id': ['4', '5', '6'],
        'first': ['Billy', 'Brian', 'Bran'],
        'last': ['Bonder', 'Black', 'Balwner']
    }
    dataframe_b = pd.DataFrame(data_b, columns=['id', 'first', 'last'])

    # Concatenate DataFrames by rows
    print(pd.concat([dataframe_a, dataframe_b], axis=0))

    # Concatenate DataFrames by columns
    print(pd.concat([dataframe_a, dataframe_b], axis=1))

    # Create row
    row = pd.Series([10, 'Chris', 'Chillon'], index=['id', 'first', 'last'])

    # Append row
    print(dataframe_a.append(row, ignore_index=True))

    # 3.19
    # Load library
    import pandas as pd

    # Create DataFrame
    employee_data = {
        'employee_id': ['1', '2', '3', '4'],
        'name': ['Amy Jones', 'Allen Keys', 'Alice Bees', 'Tim Horton']
    }
    dataframe_employees = pd.DataFrame(employee_data, columns= ['employee_id', 'name'])

    # Create DataFrame
    sales_data = {
        'employee_id': ['3', '4', '5', '6'],
        'total_sales': [23456, 2512, 2345, 1455]
    }
    dataframe_sales = pd.DataFrame(sales_data, columns=['employee_id', 'total_sales'])

    # Merge DataFrames
    print(pd.merge(dataframe_employees, dataframe_sales, on='employee_id'))

    # Merge DataFrames
    print(pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='outer'))

    # Merge DataFrames
    print(pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='left'))

    # Merge DataFrames
    print(pd.merge(dataframe_employees, dataframe_sales, left_on='employee_id', right_on='employee_id'))