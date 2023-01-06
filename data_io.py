import pandas as pd


def read_data(path):
    """
    A function to read data from a file under given path
    and split it to other feature values and target feature values.
    Skips all lines in file with too many or too few values.
    Returns feature values, target feature values and list of feature names
    Returns None if an exception occurs and prints a proper information
    """
    # @TODO permission error and other errors while opening a file handling
    try:
        data = pd.read_csv(path, on_bad_lines='skip')
    except PermissionError:
        print(f"You do not have permission to open file under path:\n{path}")
        print('Please specify another file.')
        return None
    except IsADirectoryError:
        print(f"Given path: '{path}' doesn't lead to a file.")
        print("Please give the correct path with the file name and its extension.")
        return None
    except FileNotFoundError:
        print(f"No file found under path:\n{path}")
        print('Please check if file exists and make sure to specify its extension')
        print('or specify another file.')
        return None
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def main():
    read_data('./drzewo decyzyjne/datasets/test_file.csv')


if __name__ == '__main__':
    main()
