#!/usr/local/bin/python3

import inspect

def custom_inspect_class(This_Class):
    return inspect.getmembers(This_Class, lambda a:not(inspect.isroutine(a)))