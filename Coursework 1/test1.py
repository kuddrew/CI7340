# Daniel Drew - k2059915
import os  # change working diredctory to current directory
import os.path as path # check file exists
import pandas as pd # import csv, create and manipulate dataframes
import matplotlib.pyplot as plt  # so we can plot scatter of the data & do a bit of visualisation
from pandas import option_context # for prettying up output
`import seaborn as sns  # so we can plot the correlation matrix`
import numpy as np  # because arrays


def set_current_path():
    working_dir = "Coursework 1"
    os.chdir(working_dir)
    return 1


def load_data(data_set):
    # Import datasets
    dataPath = "./data/"
    dataFilename = data_set + ".csv"
    if path.isfile(dataPath + dataFilename):
        # file exists - let's import it
        imported_data = pd.read_csv(dataPath + dataFilename)  # imports as dataframe
        # imported_data = np.genfromtxt (dataPath + dataFilename, delimiter=",")  # imports as numpy array
        print(dataPath + dataFilename + " loaded\n")
    else:
        # file does not exist - throw an error
        print("File does not exist")
    return imported_data


def display_features(df):
    print(type(df))
    num_rows = len(df)
    num_cols = len(df.columns)
    # col_names = data_set.keys()
    print("Number of rows / records: {}\nNumber of columns / attributes / dimensionality: {}\n"
          .format(num_rows, num_cols))
    # print("Column names {} \n".format(df.columns))
    print("Column names (and any number of any null values):\n{}\n".format(df.isnull().any()))
    #print("Column names")
    #for col_name in df.columns:
    #    print(col_name)
    return 1


def show_covariance(df):
    plt.figure(figsize=(25, 25))
    plt.suptitle("Covariance of data attributes", fontsize=20)
    plt.tight_layout()
    # drop 'target' column from analysis / correlation
    df2 = df  #.drop(['target'], axis=1)

    # Summary key aspects of all attributes
    # set the parameters for prettier output of the dataframe
    my_option_context = option_context('display.max_rows', None,
                        'display.max_columns', None,
                        'display.width', 1000,
                        'display.precision', 2,
                        'display.colheader_justify', 'left')
    with my_option_context:
        my_desc = df2.describe()
        print("Key aspects of all attributes:\n{}\n".format(my_desc))

    ax = plt.subplot(2, 2, 1)
    ax.set_title("Covariance of All Attributes", fontsize=15)
    plt.ylabel('Attributes')
    plt.xlabel('Attributes')
    plt.show()

    # generate correlation matrix
    corr = df2.corr()
    # mask the matrix to show just bottom left
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # plot the correlation matrix
    sns.heatmap(corr, square=True, annot=True, mask=mask, cmap = 'coolwarm')

    # Summary key aspects of attributes by class
    for my_class in range(len(df.columns)):
        # filter on class
        # df2 = df[df['target'] == my_class]
        # drop 'target' column from correlation matrix
        # df3 = df2.drop(['target'], axis=1)
        my_desc = df.describe()

        with my_option_context:
            print("Key Aspects of Attributes in {}:\n{}\n".format(df.columns[my_class], my_desc))

        # plot a heatmap to visualise the correlation between the different attributes
        ax = plt.subplot(2, 2, my_class + 2)  # +1 because my_class is an index and starts at 0
        #                                        and +2 because we did the plot of all earlier
        ax.set_title("Covariance of Attributes in {}\n".format(df.columns[my_class]), fontsize=15)
        plt.ylabel('Attributes')
        plt.xlabel('Attributes')

        # generate correlation matrix
        corr = df.corr()
        # mask the matrix to show just bottom left
        mask = np.triu(np.ones_like(corr, dtype=bool))
        # plot the correlation matrix
        sns.heatmap(corr, square=True, annot=True, mask=mask, cmap = 'coolwarm')

    return 1


def plot_bar_charts(df):
    print(type(df))
    x_values = df.columns
    ax = df.plot.bar()
    print(x_values)

    return 1

def main():
    set_current_path()
    if_data = load_data("input_features")
    print(if_data.shape)
    # print(if_data)
    tv_data = load_data("target_values")

    #display_features(if_data)
    #show_covariance(if_data)

    plot_bar_charts(if_data.iloc[:,0:7])

if __name__ == "__main__":
    main()
