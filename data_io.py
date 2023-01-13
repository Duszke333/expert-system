import pandas as pd


def read_data(path):
    """
    A function that reads data from a file under given path.
    Skips all lines in file with too many or too few values.
    Prints a proper message and returns None if an exception occurs.
    Returns a pandas DataFrame object with data collected from file.
    """
    try:
        data = pd.read_csv(path, on_bad_lines='skip')
    except PermissionError:
        print(f"You do not have permission to open file under path:\n{path}")
        print('Please specify path to another file.')
        return None
    except IsADirectoryError:
        print(f"Given path: '{path}' doesn't lead to a file.")
        print("Please give the correct path with the file name and its extension.")
        return None
    except FileNotFoundError:
        print(f"No file found under path:\n{path}")
        print('Please check if file exists and make sure to specify its extension ', end='')
        print('or specify another file.')
        return None
    except OSError:
        print(f"Given path: '{path}' leads to unexisting directory.")
        print('Please input correct path.')
        return None
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def write_data(data, path):
    """
    A function that saves given pandas DataFrame object to a .csv file.
    Prints a proper message and returns False if an exception occurs.
    Returns True if saving to a file was successful.
    """
    try:
        data.to_csv(path, index=False)
    except IsADirectoryError:
        print(f"Given path: '{path}' doesn't lead to a file.")
        print("Please give the correct path with the file name and its extension.")
        return False
    except PermissionError:
        print(f"You do not have permission to open file under path:\n{path}")
        print('Please specify path to another file.')
        return False
    except OSError:
        print(f"Given path: '{path}' leads to unexisting directory.")
        print('Please input correct path.')
        return False
    return True
