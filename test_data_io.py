import pandas as pd
from io import StringIO
from data_io import read_data, write_data


def test_read_data_simple():
    file_data = """Outlook,Temperature,Humidity,Wind,PlayGolf
Sunny,Hot,High,Weak,No
Overcast,Hot,High,Weak,Yes
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
    assert all(function_result.eq(expected_function_result)) is True


def test_read_data_with_bad_rows():
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
    assert all(function_result.eq(expected_function_result)) is True


def test_read_data_with_duplicates():
    file_data = """Outlook,Temperature,Humidity,Wind,PlayGolf
Sunny,Hot,High,Weak,No
Overcast,Hot,High,Weak,Yes
Rainy,Cool,Normal,Weak,Yes
Sunny,Hot,High,Weak,No
Overcast,Hot,High,Weak,Yes"""
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
    assert all(function_result.eq(expected_function_result)) is True


def test_read_data_errors():
    assert read_data('./datasets/') is None
    assert read_data('./datasets/non_existing_file.txt') is None
    assert read_data('./datasets/file_without_extension') is None
    assert read_data('./datasets/nonexisting directory/file.csv') is None
    # Cannot test PermissionError


def test_write_data():
    raw_data = {
        'Outlook': ['Sunny', 'Overcast', 'Rainy'],
        'Temperature': ['Hot', 'Hot', 'Cool'],
        'Humidity': ['High', 'High', 'Normal'],
        'Wind': ['Weak', 'Weak', 'Weak'],
        'PlayGolf': ['No', 'Yes', 'Yes']
    }
    data = pd.DataFrame(raw_data)
    file_handle = StringIO()
    assert write_data(data, file_handle) is True


def test_write_data_error():
    expected_data = {
        'Outlook': ['Sunny', 'Overcast', 'Rainy'],
        'Temperature': ['Hot', 'Hot', 'Cool'],
        'Humidity': ['High', 'High', 'Normal'],
        'Wind': ['Weak', 'Weak', 'Weak'],
        'PlayGolf': ['No', 'Yes', 'Yes']
    }
    data = pd.DataFrame(expected_data)
    assert write_data(data, './datasets/') is False
    assert write_data(data, './datasets/nonexisting directory/file.csv') is False
    # Cannot test PermissionError
