import pandas as pd
from tree import DecisionTree
from tree import validate_choice, validate_yes_no
from data_io import read_data, write_data
from os import system


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
    input_choice = input('>>').strip()
    input_choice = validate_choice(input_choice, data_input_choices.keys())
    system('cls||clear')
    return data_input_choices[input_choice]()


def import_from_file():
    """
    A function that asks user for path to file, then reads data from it,
    then calls the get_resultant_feature_name() function.
    Returns collected data, given path to file and the resultant feature label.
    """
    print('Please input path to the desired csv file (with file extension): ')
    data_from_file = None
    while data_from_file is None:
        file_path = input('>>').strip()
        system('cls||clear')
        data_from_file = read_data(file_path)
    resultant_feature = get_resultant_feature_name(list(data_from_file.columns))
    return data_from_file, file_path, resultant_feature


def add_new_rule_to_data(data, resultant_feature):
    """
    A function that collects a value of every feature for given dataset
    and adds the list of input values as a new rule to the data.
    It prevents the user from repeating rules and, if the input rule is conflicting with
    already existing data (same values, different outcomes), it informs the user
    about this and asks him if he wants to replace the old rule with the new one.
    """
    data_object_values = []
    for feature in data.columns:
        print(f'Please input feature "{feature}" value')
        feature_value = input('>>').strip()
        try:
            data_object_values.append(eval(feature_value))
        except Exception:
            data_object_values.append(feature_value)
    data.loc[len(data)] = data_object_values
    value_columns = list(data.columns)
    value_columns.remove(resultant_feature)
    check_for_conflicts = data.duplicated(subset=value_columns, keep=False)
    if any(check_for_conflicts):
        old_index = check_for_conflicts[check_for_conflicts].index[0]
        old_outcome = data.iloc[old_index][resultant_feature]
        new_outcome = data.iloc[-1][resultant_feature]
        if old_outcome == new_outcome:
            system('cls||clear')
            print('Error - rule already in database!')
            print('Please input another rule.')
            data.drop_duplicates(inplace=True)
            add_new_rule_to_data(data, resultant_feature)
        else:
            print('Warning - rule with the same values but different outcome found!')
            print('The rules are:')
            print(data.iloc[[old_index, -1]])
            print('Would you like to replace the old rule (the higher one)? [Yes/No]')
            replace = validate_yes_no(input('>>').strip())
            if replace:
                data.drop_duplicates(subset=value_columns, inplace=True, keep='last')
                data.reset_index(drop=True, inplace=True)
            else:
                system('cls||clear')
                print('Please input a new rule then.')
                data.drop_duplicates(subset=value_columns, inplace=True)
                add_new_rule_to_data(data, resultant_feature)
    else:
        print('Rule added successfully!')
        input('Press ENTER to continue.\n')
        system('cls||clear')


def input_data_from_keyboard():
    """
    A function that lets the user create an empty dataset,
    then passes it to the input_data_from_keyboard_function(), where
    the created dataset can be filled with rules.
    Returns the filled dataset.
    """
    feature_names = []
    print('Please input at least 2 different feature names.')
    while True:
        print('Please input the feature name', end='')
        if len(feature_names) >= 2:
            print(' or type "quit" to end adding', end='')
        feature_name = input(':\n>>').strip()
        while not feature_name:
            print('Error - feature name must be given.')
            feature_name = input('>>').strip()
        while feature_name in feature_names:
            print('Error - feature name has already been given.', end='')
            feature_name = input('Please input another feature name\n>>').strip()
        if feature_name.lower() == 'quit':
            if len(feature_names) >= 2:
                break
            else:
                print('Feature cannot be named "quit". Try again.')
                continue
        feature_names.append(feature_name)
    data = pd.DataFrame(columns=feature_names)
    system('cls||clear')
    resultant_feature = get_resultant_feature_name(feature_names)
    return fill_dataset_with_rules(data, None, resultant_feature)


def fill_dataset_with_rules(data, file_path, resultant_feature):
    """
    A function that lets the user add more rules to the dataset.
    It then asks the user if he wants to save it to a file.
    Returns the updated dataset, file path and the resultant feature label.
    """
    not_done_collecting_information = True
    data_object_index = 1
    print('Now please input at least 2 rules.')
    while not_done_collecting_information:
        print(f'Rule no. {data_object_index}:')
        add_new_rule_to_data(data, resultant_feature)
        data_object_index += 1
        done = input('Do you wish to add more rules? [Yes/No]\n>>').strip()
        not_done_collecting_information = validate_yes_no(done)
        system('cls||clear')
    file_path = ask_to_save_data(data, file_path)
    return data, file_path, resultant_feature


def import_and_input():
    """
    A function that imports data from file,
    then supplements it with data input by user.
    Returns data and path to file
    """
    data, file_path, resultant_feature = import_from_file()
    data, file_path, _ = fill_dataset_with_rules(data, file_path, resultant_feature)
    return data, file_path, resultant_feature


def get_resultant_feature_name(list_of_features):
    """
    A function that asks user to indicate which label is the resultant feature,
    then returns it.
    """
    print('Please choose the resultant feature label from feature labels listed below:')
    print(list_of_features)
    resultant_feature = validate_choice(input('>>').strip(), list_of_features)
    system('cls||clear')
    return resultant_feature


def build_tree(data, resultant_feature):
    """
    A function that creates the decision tree, then returns it.
    """
    print('Creating the tree... ', end='')
    tree = DecisionTree(data, resultant_feature)
    print('Done!')
    print(tree.coverage())
    tree.printer()
    input('Press ENTER to continue.\n')
    system('cls||clear')
    return tree


def learn(data, file_path, resultant_feature):
    """
    A function that extends the database with a new rule
    and asks to save it to a file,
    then builds a new tree from the updated database.
    Returns the tree, path to the file and the updated dataset.
    """
    system('cls||clear')
    print('Please input a new rule so I can work properly next time.')
    add_new_rule_to_data(data, resultant_feature)
    file_path = ask_to_save_data(data, file_path)
    print('Tree will now be rebuilt.')
    tree = build_tree(data, resultant_feature)
    return tree, file_path, data


def ask_to_save_data(data, file_path):
    """
    A function that asks the user if he wants to save the updated database to a file.
    If he doesn't, it returns file path passed to it as argument.
    If he does, It calls upload_data_to_file() function
    and returns the result of it, which also is a file path.
    """
    print('Would you like to save updated data to a file? [Yes/No]')
    save_choice = validate_yes_no(input('>>').strip())
    system('cls||clear')
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
    print('File saved successfully!')
    input('Press ENTER to continue.\n')
    system('cls||clear')
    return file_path


def make_decision(tree, data, file_path):
    """
    A function that asks the user questions and decides what the correct answer is.
    If the answer is not correct, it calls the learn() function to extend the database.
    """
    print('You will now be asked a series of questions.')
    print('Provided information will help determine the correct answer.')
    input('Press ENTER to continue.\n')
    the_decision = tree.decide()
    print(f'The suggested decision is: {the_decision}.')
    correct = input('Is it correct? [Yes/No]\n>>').strip()
    correct = validate_yes_no(correct)
    system('cls||clear')
    if not correct:
        tree, file_path, data = learn(data, file_path, tree.outcome_header)
    print('Do you wish for me to make a decision again? [Yes/no]')
    if validate_yes_no(input('>>').strip()):
        system('cls||clear')
        print('Making another decision...')
        make_decision(tree, data, file_path)
    system('cls||clear')


def main():
    """
    The main function of the program. It calls other functions in specific order.
    """
    try:
        print("Welcome to the expert system.")
        data, file_path, resultant_feature = import_data_choice()
        tree = build_tree(data, resultant_feature)
        make_decision(tree, data, file_path)
        print('Thank you for using the program.')
    except (KeyboardInterrupt, EOFError):
        system('cls||clear')
        print('Thank you for using the program.')


if __name__ == '__main__':
    main()
