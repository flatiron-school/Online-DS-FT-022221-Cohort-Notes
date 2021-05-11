import numpy as np
import pandas as pd

def is_female(sex_cell):
    """
    :param sex_series: an element of the sex_upon_outcome series with 5 sex types
        Neutered Male
        Spayed Female
        Intact Male
        Intact Female
    :return: a value from  3 sex types
        unknown, male, female
    """

    if 'female' in sex_cell.lower():
        return 'female'
    elif 'unknown' in sex_cell.lower():
        return 'unknown'
    else:
        return 'male'

def female_binary(sex_cell):
    """
    :return: 1 for female, 0 for male
    """

    if 'female' in sex_cell.lower():
        return 1
    else:
        return 0

def outcome_binary(outcome_element):
    """
    :param outcome_element: an element from the outcome series in the shelter dataset
        Adoption
        Transfer
        Return to Owner
        Euthanasia
        Died
        Disposal
        Rto-Adopt
        Relocate
    :return:
        1 for adoption
        0 for any other outcome
    """
    if 'adoption' in outcome_element.lower():
        return 1
    else:
        return 0

def animal_type(animal_element):
    """
    :param animal_element:
    an element from the animal type column: one of four animal types
        Dog
        Cat
        Other
        Bird
    :return: 1 for dog, 0 for cat
    """

    if 'dog' in animal_element.lower():
        return 1

    else:
        return 0

def age_to_days(age_element):

    '''
    params: age upon outcome of shelter animal.
    A number followed by a unit of time
    'NULL', 'days', 'month', 'months', 'week', 'weeks', 'year', 'years'

    returns: days old at outcome
    '''

    age_split = age_element.split(' ')

    if len(age_split) == 1:
        return np.nan

    elif age_split[1] == 'days':
        return int(age_split[0])

    elif age_split[1] in (['month' or 'months']):
        return int(age_split[0]) * 30

    elif age_split[1] in ['week' or 'weeks']:
        return int(age_split[0]) * 7

    else:
        return int(age_split[0]) * 365



def preprocess_df(df):
    """
    :param df:
    :return:
    """

    df['age_in_days'] = pd.to_datetime(df['datetime']) - pd.to_datetime(df['date_of_birth'])
    df['age_in_days'] = df['age_in_days'].apply(lambda x: x.days)
    df = df[(df.animal_type == 'Dog') | (df.animal_type == 'Cat')]
    df['is_dog'] = df['animal_type'].apply(animal_type)


    df['is_female'] = df['sex_upon_outcome'].apply(is_female)
    df = df[df.is_female != 'unknown']
    df['is_female'] = df.is_female.apply(female_binary)

    df['adoption'] = df['outcome_type'].apply(outcome_binary)
    df = df[['is_dog', 'age_in_days', 'is_female', 'adoption']]

    return df



