import pandas as pd
from tree import DecisionTree
from tree import validate_choice, validate_yes_no
from data_io import read_data, write_data


def import_data_choice():
    """
    A function that lets the user choose where to collect data from.
    Returns the result of a function chosen by the user.
    """
    text_data_input_choices = """
    1: Import from csv file
    2: Input from keyboard
    3: Import from csv file and input from keyboard
    4: Exit program"""
    data_input_choices = {
        '1': import_from_file,
        '2': input_data_from_keyboard,
        '3': import_and_input,
        '4': exit
    }
    print("Choose how do you want to input rules or if you want to exit:", end='')
    print(text_data_input_choices)
    data_input_choice = input('>>').strip()
    data_input_choice = validate_choice(data_input_choice, data_input_choices.keys())
    return data_input_choices[data_input_choice]()


def import_from_file():
    """
    A function that asks user for path to file then reads data from it.
    Returns collected data and given path to file.
    """
    print('Please input path to the desired .csv file (with file extension): ')
    file_path = input('>>').strip()
    data_from_file = read_data(file_path)
    while data_from_file is None:
        file_path = input('>>').strip()
        data_from_file = read_data(file_path)
    return data_from_file, file_path


def add_new_rule_to_data(data):
    """
    A function that collects a value of every feature for given dataset
    and adds the array of input values as a new rule to the data.
    """
    data_object_values = []
    for feature in data.columns:
        msg = f'Please input feature "{feature}" value: '
        feature_value = input(msg).strip()
        try:
            data_object_values.append(eval(feature_value))
        except Exception:
            data_object_values.append(feature_value)
    data.loc[len(data)] = data_object_values


def input_data_from_keyboard(data=None, file_path=None):
    """
    A function that lets user create a dataset and fill it with values.
    It then asks the user if he wants to save it to a file.
    Returns the created dataset and file path.
    """
    if data is None:
        feature_names = []
        while True:
            print('Please input the feature name', end='')
            if len(feature_names) >= 2:
                print('or type "quit" to end adding', end='')
            feature_name = input(': ').strip()
            while feature_name in feature_names:
                print('Error - feature name has already been given.', end='')
                feature_name = input('Please input another feature name: ').strip()
            if feature_name.lower() == 'quit':
                if len(feature_names) >= 2:
                    break
                else:
                    print('Feature cannot be named "quit". Try again.')
                    continue
            feature_names.append(feature_name)
        data = pd.DataFrame(columns=feature_names)
    not_done_collecting_information = True
    data_object_index = 1
    print('Now please input values for rules.')
    while not_done_collecting_information:
        print(f'Rule no. {data_object_index}:')
        add_new_rule_to_data(data)
        data_object_index += 1
        done = input('Do you wish to continue? [Yes/No]\n>>').strip()
        not_done_collecting_information = validate_yes_no(done)
    file_path = ask_to_save_data(data, file_path)
    return data, file_path


def import_and_input():
    """
    A function that imports data from file,
    then supplements it with data input by user.
    Returns data and path to file
    """
    data, file_path = import_from_file()
    data, file_path = input_data_from_keyboard(data, file_path)
    return data, file_path


def get_target_feature_name(list_of_features):
    """
    A function that asks user to indicate which feature will be searched for,
    then returns it.
    """
    print('Please choose the name of the target feature from names listed below:')
    print(list_of_features)
    target_feature = validate_choice(input('>>').strip(), list_of_features)
    return target_feature


def build_tree(data, target_feature):
    """
    A function that creates the decision tree, then returns it.
    """
    print('Creating the tree... ', end='')
    tree = DecisionTree(data, target_feature)
    print('Done!')
    print(tree.coverage())
    return tree


def learn(data, file_path, target_feature):
    """
    A function that extends the database with a new rule
    and asks to save it to a file,
    then builds a new tree from the updated database.
    Returns the tree and path to file.
    """
    print('Please input new information so I can work properly this time.')
    add_new_rule_to_data(data)
    file_path = ask_to_save_data(data, file_path)
    print('Tree will now be rebuilt.')
    tree = build_tree(data, target_feature)
    return tree, file_path


def ask_to_save_data(data, file_path):
    """
    A function that asks the user if he wants to save database to a file.
    If he doesn't, it returns file path passed to it as argument.
    If he does, It calls upload_data_to_file() function
    and returns the result of it, which also is a file path.
    """
    print('Would you like to save updated data to a file? [Yes/No]')
    save_choice = validate_yes_no(input('>>').strip())
    return upload_data_to_file(data, file_path) if save_choice else file_path


def upload_data_to_file(data, file_path):
    """
    A function that saves data to the file under given path or asks for it
    if it has not been given before.
    Returns the file path.
    """
    if not file_path:
        file_path = input('Please input path to file where data will be saved: ').strip()
    else:
        print('Overwriting file under path given earlier.')
    done_writing = write_data(data, file_path)
    while not done_writing:
        file_path = input('>>').strip()
        done_writing = write_data(data, file_path)
    print('Done!')
    return file_path


def make_decision(tree, data, file_path):
    """
    A function that asks the user questions and tries to guess the correct answer.
    If the answer is not correct, it calls the learn() function to extend the database.
    """
    print('You will now be asked a series of questions.')
    print('Provided information will help determine the correct answer.')
    print("Let's begin.")
    the_guess = tree.begin_guessing()
    print(f'My guess is: {the_guess}.')
    correct = input('Is it correct? [Yes/No]\n>>').strip()
    correct = validate_yes_no(correct)
    if not correct:
        tree, file_path = learn(data, file_path, tree.target_name)
    print('Do you wish to guess again? [Yes/no]')
    if validate_yes_no(input('>>').strip()):
        make_decision(tree, data, file_path)


def main():
    """
    The main function of the program. It calls other functions in specific order.
    """
    print("Welcome to the expert system.")
    data, file_path = import_data_choice()
    target_feature = get_target_feature_name(list(data.columns))
    tree = build_tree(data, target_feature)
    make_decision(tree, data, file_path)
    print('Goodbye.')


if __name__ == '__main__':
    main()
