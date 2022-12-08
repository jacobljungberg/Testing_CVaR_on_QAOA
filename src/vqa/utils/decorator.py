from functools import wraps
import datetime
import time
import logging
import psutil
import os
import sys

""" Decorators provide a simple syntax for higher order functions. 
  
  A higher order function does at least two things. 
  1. takes one or more functions as arguments 
  2. returns a function as its result 

  https://en.wikipedia.org/wiki/Higher-order_function 
"""


def timer(f):
    """Decorator that measures execution time of the wrapped function

    https://towardsdatascience.com/function-wrappers-in-python-5146f3ad0601

    Args:
        f: function to be wrapped

    Returns:

    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        logging.info(
            f"time -> {args[0].__class__.__name__}:{f.__name__} -> {round(end - start, 6)}"
        )
        return result

    return wrapper


def obj_memory(f):
    """Decorator to read memory usage

    If you apply this decorator on a class function, it will read memory usage of all class attributes
    before and after execution of the wrapped function. This very accurate, but works only for class functions
    and respective attributes.

    Args:
        f: wrapped function

    Returns:

    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        try:
            self = args[0]
            mem = 0
            for variable in self.__dict__.keys():
                mem += _get_memory_size(getattr(self, variable))
            logging.info(
                f"object memory usage -> {self.__class__.__name__}:{f.__name__} -> {mem}"
            )
        except AttributeError:
            pass
        return result

    return wrapper


def memory(f):
    """Decorator to read memory usage

    The approach of this decorator is to read RAM usage of the python thread before and after execution.
    Results may vary due to garbage collector and other applications running on the same computer, but with
    this method it is also possible to monitor body variables that are no class variables.

    Args:
        f: wrapped function

    Returns:

    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        start = process.memory_info().rss
        result = f(*args, **kwargs)
        end = process.memory_info().rss
        logging.info(
            f"memory usage -> {args[0].__class__.__name__}:{f.__name__} -> {end - start}"
        )
        return result

    return wrapper


def eval_logger(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        logging.debug(
            f"evaluates to -> {args[0].__class__.__name__}:{f.__name__} -> {result}"
        )
        return result

    return wrapper


def _get_memory_size(obj: object, max_depth=5, size=0) -> int:
    """Recursive function to determine the memory size of an object

    Args:
        obj: object to be measured
        max_depth: how many recursion steps to perform before break
        size: current size of the object

    Returns:
        size of the object

    """
    if max_depth != 0:
        try:
            for attribute_name in obj.__dict__.keys():
                size += _get_memory_size(
                    getattr(obj, attribute_name), max_depth=max_depth - 1, size=size
                )
        except AttributeError:
            pass
        size += sys.getsizeof(obj)
    return size
