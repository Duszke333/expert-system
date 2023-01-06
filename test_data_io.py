import pandas as pd
from io import StringIO
from data_io import read_data


def test_read_data():
    file_data = """Outlook,Temperature,Humidity,Wind,PlayGolf
Sunny,Hot,High,Weak,No
Sunny,Hot,High,Strong,No,Bike
Overcast,Hot,High,Weak,Yes
Rainy,Mild,High,,Yes
Rainy,Cool,Normal,Weak,Yes"""
    file = StringIO(file_data)
    expected_data = {
        'Outlook': ['Sunny', 'Overcast', 'Rainy'],
        'Temperature': ['Hot', 'Hot', 'Cool'],
        'Humidity': ['High', 'High', 'Normal'],
        'Wind': ['Weak', 'Weak', 'Weak'],
        'PlayGolf': ['No', 'Yes', 'Yes']
    }
    expected_function_result = pd.DataFrame(expected_data)
    function_result = read_data(file)
    assert all(function_result == expected_function_result) is True


def test_read_data_errors():
    assert read_data('./datasets/') is None
    assert read_data('./datasets/non_existing_file.txt') is None
    assert read_data('./datasets/file_without_extension') is None
    # Cannot test PermissionError
