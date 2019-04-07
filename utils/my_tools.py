#!/usr/local/bin/python3

import inspect

def custom_inspect_class(This_Class):
    return inspect.getmembers(This_Class, lambda a:not(inspect.isroutine(a)))

def unpack_list(must_unpack,uniq = False):
    '''Function to unpack lists
    Example : [[], [], [], ['bootstrap', 'max_depth', 'max_features', 'min_samples_leaf', 'n_estimators'], []] 
            turns into
            ['bootstrap', 'max_depth', 'max_features', 'min_samples_leaf', 'n_estimators']
            
    '''
    
    simple_list  = [item for sublist in must_unpack for item in sublist]
    
    if(uniq == True):
        simple_list = set(simple_list)
    else:
        pass
    
    return list(simple_list)
