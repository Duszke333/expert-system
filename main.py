import pandas as pd
from tree import DecisionTree
from data_io import read_data, write_data
from os import system


def validate_choice(possible_choices):
    """
    A function that checks whether the user input answer is present in given choice list.
    If not, user will be asked again to answer until the choice is correct.
    Function returns the valid user input choice.
    """
    choice = input('>>').strip()
    while choice not in possible_choices:
        choice = input('Unrecognized choice. Please choose again: ').strip()
    return choice


def validate_yes_no():
    """
    A function that checks if a yes/no answer is in fact a yes/no.
    If not, user will be asked again to answer until the choice is correct.
    Returns a True / False depending on the answer (Yes / No).
    """
    choice = input('>>').strip()
    while choice.lower() not in ['yes', 'no']:
        choice = input('Unrecognized choice. Please choose again [Yes/No]: ').strip()
    return True if choice.lower() == 'yes' else False


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
    input_choice = validate_choice(data_input_choices.keys())
    system('cls||clear')
    return data_input_choices[input_choice]()


def import_from_file():
    """
    A function that asks user for path to file, then reads data from it,
    then calls the get_outcome_variable_name() function.
    Returns collected data, given path to file and the outcome variable header.
    """
    print('Please input path to the desired csv file (with file extension): ')
    print('*example: ./datasets/(name_of_chosen_file).csv')
    data_from_file = None
    while data_from_file is None:
        file_path = input('>>').strip()
        system('cls||clear')
        data_from_file = read_data(file_path)
    outcome_variable = get_outcome_variable_name(list(data_from_file.columns))
    return data_from_file, file_path, outcome_variable


def add_new_rule_to_data(data, outcome_variable):
    """
    A function that collects a value of every variable for given dataset
    and adds the list of input values as a new rule to the data.
    It prevents the user from repeating rules and, if the input rule is conflicting with
    already existing data (same values, different outcomes), it informs the user
    about this and asks him if he wants to replace the old rule with the new one.
    """
    data_object_values = []
    for variable in data.columns:
        print(f'Please input variable "{variable}" value:')
        variable_value = input('>>').strip()
        while not variable_value:
            print('Variable value cannot be empty! Please input again:')
            variable_value = input('>>').strip()
        try:
            data_object_values.append(eval(variable_value))
        except Exception:
            data_object_values.append(variable_value)
    data.loc[len(data)] = data_object_values
    value_columns = list(data.columns)
    value_columns.remove(outcome_variable)
    check_for_conflicts = data.duplicated(subset=value_columns, keep=False)
    if any(check_for_conflicts):
        old_index = check_for_conflicts[check_for_conflicts].index[0]
        old_outcome = data.iloc[old_index][outcome_variable]
        new_outcome = data.iloc[-1][outcome_variable]
        if old_outcome == new_outcome:
            system('cls||clear')
            print('Error - rule already in database!: ')
            print(data.iloc[[-1]], '\n')
            print('Please input another rule.')
            data.drop_duplicates(inplace=True)
            add_new_rule_to_data(data, outcome_variable)
        else:
            print('Warning - rule with the same values but different outcome found!')
            print('The rules are:')
            print(data.iloc[[old_index, -1]])
            print('Would you like to replace the old rule (the higher one)? [Yes/No]')
            replace = validate_yes_no()
            if replace:
                data.drop_duplicates(subset=value_columns, inplace=True, keep='last')
                data.reset_index(drop=True, inplace=True)
            else:
                system('cls||clear')
                print('Please input a new rule then.')
                data.drop_duplicates(subset=value_columns, inplace=True)
                add_new_rule_to_data(data, outcome_variable)
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
    variable_names = []
    print('Please input at least 2 different variable names.')
    variable_name = ''
    while not (variable_name.lower() == 'quit' and len(variable_names) >= 2):
        print('Please input the variable name', end='')
        if len(variable_names) >= 2:
            print(' or type "Quit" to end adding', end='')
        variable_name = input(':\n>>').strip()
        while not variable_name:
            print('Error - variable name cannot be empty.')
            variable_name = input('>>').strip()
        while variable_name in variable_names:
            print('Error - this variable name has already been given. ', end='')
            variable_name = input('Please input another variable name:\n>>').strip()
        if variable_name.lower() != 'quit':
            variable_names.append(variable_name)
        elif len(variable_names) < 2:
            print('Variable cannot be named "Quit". Try again.')
    data = pd.DataFrame(columns=variable_names)
    system('cls||clear')
    outcome_variable = get_outcome_variable_name(variable_names)
    return fill_dataset_with_rules(data, None, outcome_variable)


def fill_dataset_with_rules(data, file_path, outcome_variable):
    """
    A function that lets the user add more rules to the dataset.
    It then asks the user if he wants to save it to a file.
    Returns the updated dataset, file path and the outcome variable header.
    """
    not_done_collecting_information = True
    data_object_index = 1
    print('Now please input at least 2 rules.')
    while not_done_collecting_information:
        print(f'Rule no. {data_object_index}:')
        add_new_rule_to_data(data, outcome_variable)
        data_object_index += 1
        if data_object_index > 2:
            print('Do you wish to add more rules? [Yes/No]')
            not_done_collecting_information = validate_yes_no()
        system('cls||clear')
    file_path = ask_to_save_data(data, file_path)
    return data, file_path, outcome_variable


def import_and_input():
    """
    A function that imports data from file,
    then supplements it with data input by user.
    Returns data, path to file and outcome variable header.
    """
    data, file_path, outcome_variable = import_from_file()
    data, file_path, _ = fill_dataset_with_rules(data, file_path, outcome_variable)
    return data, file_path, outcome_variable


def get_outcome_variable_name(list_of_variables):
    """
    A function that asks user to indicate which variable is the outcome variable,
    then returns it.
    """
    print('Please choose the outcome variable from variables listed below:')
    print(list_of_variables)
    outcome_variable = validate_choice(list_of_variables)
    system('cls||clear')
    return outcome_variable


def build_tree(data, outcome_variable):
    """
    A function that creates the decision tree, then returns it.
    """
    print('Creating the tree... ', end='')
    tree = DecisionTree(data, outcome_variable)
    print('Done!')
    print(tree.accuracy())
    input('Press ENTER to continue.\n')
    system('cls||clear')
    return tree


def learn(data, file_path, outcome_variable):
    """
    A function that extends the database with a new rule
    and asks to save it to a file,
    then builds a new tree from the updated database.
    Returns the tree, path to the file and the updated dataset.
    """
    system('cls||clear')
    print('Please input a new rule so I can work properly next time.')
    add_new_rule_to_data(data, outcome_variable)
    file_path = ask_to_save_data(data, file_path)
    print('Tree will now be rebuilt.')
    tree = build_tree(data, outcome_variable)
    return tree, file_path, data


def ask_to_save_data(data, file_path):
    """
    A function that asks the user if he wants to save the updated database to a file.
    If he doesn't, it returns file path passed to it as argument.
    If he does, It calls upload_data_to_file() function
    and returns the result of it, which also is a file path.
    """
    print('Would you like to save updated data to a file? [Yes/No]')
    save_choice = validate_yes_no()
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
        print('Overwriting file under path given earlier...')
    done_writing = write_data(data, file_path)
    while not done_writing:
        file_path = input('>>').strip()
        done_writing = write_data(data, file_path)
    print('File saved successfully!')
    input('Press ENTER to continue.\n')
    system('cls||clear')
    return file_path


def determine_decision(node, question_number=1):
    """
    A function that asks the user questions, determines the
    outcome based on his answers and returns it.
    """
    system('cls||clear')
    if node.decision is not None:
        return node.decision
    print(f'Question no. {question_number}:')
    if node.threshold:
        print('Is the following inequality satisfied: ', end='')
        print(f'{node.variable} <= {node.threshold}? [Yes/No]')
        choice = validate_yes_no()
        if choice:
            return determine_decision(node.subnodes['<='], question_number + 1)
        else:
            return determine_decision(node.subnodes['>'], question_number + 1)
    else:
        possible_answers = list(node.subnodes.keys())
        if sorted(possible_answers) == [False, True]:
            print(f'Does the "{node.variable}" variable apply to your data? [Yes/No]')
            choice = validate_yes_no()
        else:
            print('Please choose the value that matches the ', end='')
            print(f'"{node.variable}" variable of your data from the list below:')
            print(possible_answers)
            choice = validate_choice(node.subnodes.keys())
        return determine_decision(node.subnodes[choice], question_number + 1)


def make_decision(tree, data, file_path, outcome_variable):
    """
    A function that asks the user questions and decides what the correct answer is.
    If the answer is not correct, it calls the learn() function to extend the database.
    """
    # tree.printer()
    print(f'Looking for decision regarding "{outcome_variable}"')
    print('The decision making process will now start.')
    print('You will be asked a series of questions.')
    print('Provided information will help determine the correct answer.')
    input('Press ENTER to continue.\n')
    the_decision = determine_decision(tree.root)
    print(f'The suggested decision is: {the_decision}.')
    print('Is it correct? [Yes/No]')
    correct = validate_yes_no()
    system('cls||clear')
    if not correct:
        tree, file_path, data = learn(data, file_path, tree.outcome_header)
    print('Do you wish for me to make a decision again? [Yes/no]')
    if validate_yes_no():
        system('cls||clear')
        print('Making another decision...')
        make_decision(tree, data, file_path, outcome_variable)
    system('cls||clear')


def main():
    """
    The main function of the program. It calls other functions in specific order.
    """
    try:
        system('cls||clear')
        print("Welcome to the expert system.")
        data, file_path, outcome_variable = import_data_choice()
        tree = build_tree(data, outcome_variable)
        make_decision(tree, data, file_path, outcome_variable)
    except (KeyboardInterrupt, EOFError):
        system('cls||clear')
    print('Thank you for using the program.')


if __name__ == '__main__':
    main()
