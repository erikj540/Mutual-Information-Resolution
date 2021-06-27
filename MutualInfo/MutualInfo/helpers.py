import os, sys, argparse, shutil, pickle
import numpy as np 
import pandas as pd

def swap_dict_key_and_values(d):
    """
    Given a dictionary, return a dictionary with the keys and values of the original dictionary swapped
    """
    d2 = {}
    for key, value in d.items():
        if type(value)==int:
            d2[int(value)] = key
        else:
            d2['{}'.format(value)] = key
        
    return d2

def flatten_list_of_lists(list_of_lists):
    """
    Given a list of lists, e.g., [[a,b], [c,d]], returns [a,b,c,d]
    """
    return [element for sublist in list_of_lists for element in sublist]

def pickle_object(save_path, obj):
    """
    Pickle object to save_path location

    Parameters:
        -- save_path (str) : path to pickled object
        -- obj : whatever object you want pickled

    Returns:
        -- None
    """
    outfile = open(save_path, 'wb')
    
    
    pickle.dump(obj, outfile)
    outfile.close()

    return None

def unpickle_object(save_path):
    """
    Unpickle object saved to save_path location

    Parameters:
        -- save_path (str) : path to pickled object

    Returns:
        -- obj : pickled object
    """
    infile = open(save_path, 'rb')
    obj = pickle.load(infile)
    infile.close()

    return obj

def pickle_objects_in_list(object_names, folder_path):
    """
    Pickle a bunch of objects. Provide list of object names as strings (e.g.,
    ['x', 'y', ...] and path to folder to save in
    """
    for name in object_names:
        pickle_object(os.path.join(folder_path, '{}.pkl'.format(name)), globals()[name])
    return None

def find_set_difference(list0, list1):
    """
    Parameters:
        -- list0 (list) : some list
        -- list1 (list) : some list

    Returns:
        -- stuff_in_list0_not_list1 : self-explanatory
        -- stuff_in_list1_not_list0 : self-explanatory
    """
    stuff_in_list0_not_list1 = []
    stuff_in_list1_not_list0 = []
    
    for item in list0:
        if item not in list1:
            stuff_in_list0_not_list1.append(item)
    
    for item in list1:
        if item not in list0:
            stuff_in_list1_not_list0.append(item)
    
    return (stuff_in_list0_not_list1, stuff_in_list1_not_list0)

def determine_if_sets_have_the_same_contents(list0, list1):

    if len(list0)!=len(list1):
        print('lists have different sizes')
        return False

    else:
        for item in list0:
            if item not in list1:
                return False

        for item in list1:
            if item not in list0:
                return False

        return True

def sort_list(lst, index2sorton, type='asc'):
    """

    """
    if type=='asc':
        return sorted(lst, key=lambda x: x[index2sorton])
    elif type=='des':
        return sorted(lst, key=lambda x: x[index2sorton], reverse=True)

def print_dict_elements(d, num_elements):
    print(dict(list(d.items())[0:num_elements]))
    return None

def remove_and_make_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    return None

def get_elements_from_list(list1, idx):
    return list(map(lambda x: x[idx], list1))