import numpy as np
from io import StringIO


def tag_parser(file_path):
    """parses .tag files by taking the file path. 
    Functionality is currently limited to only certain tag files and is not guaranteeded 
    to work everywhere"""
    with open(file_path) as f:
        t = f.read()
        t = t.split("Points =\n")[1]
        t = t.replace(" 0.1 1 1 \"Marker\"", "")
        t = t.replace(";", "")
        t = t.replace(" \n", "\n")

        t = t[1:]

        t = StringIO(t)
        return np.genfromtxt(t, delimiter=' ')

