#!/usr/bin/env python

"""
loads in the data from the configuration file
"""

import os

class Options(object):
    """
    Options class that allows the use to import
    """

    def __init__(self, input=None):
        """Class initialization"""
        self.job_dir      = os.getcwd()
        self.script_dir   = os.path.dirname(os.path.realpath(__file__))
        self.default_name = 'defaults.ini'
        self.input        = input

    def get(self, item):
        """Searches for the parameter first in input file
        then in the defaults"""
        if not self.input is None and os.path.exists(self.input):
            data = open(self.input, 'r').readlines()
            for line in data:
                if '#' in line[0] or '@' in line[0]:
                    continue
                if item in line and '=' not in line:
                    if item in line.split()[-1].strip():
                        continue
                    if len(item) == 1 and item != line.split('=')[0].strip():
                        continue
                    return line.split()[-1].strip()
                if item in line:
                    if item in line.split('=')[1]:
                        continue
                    if len(item) == 1 and item != line.split('=')[0].strip():
                        continue
                    return line.split()[-1].strip()
        data = open(self.script_dir + '/' + self.default_name, 'r').readlines()
        for line in data:
            if line[0] == '#' or '@' in line[0]:
                continue
            if item in line:
                if len(item) == 1 and item != line.split('=')[0].strip():
                    continue
                return line.split('=')[-1].strip()
        return 'FAILED000'

    def get_float(self, item):
        """Pulls a float out of the file"""
        val = self.get(item)
        if 'FAILED000' in val:
            print("Invalid Parameter:", item)
            exit()
        try:
            return float(val)
        except ValueError:
            print("Error - Invalid Data Type:", item, "\nExpected Float, given:", val)
            exit()

    def get_int(self, item):
        val = self.get(item)
        if 'FAILED000' in val:
            print("Invalid Parameter:", item)
            exit()
        try:
            return int(val)
        except ValueError:
            print("Error - Invalid Data Type:", item, "\nExpected Int, given:", val)
            exit()

    def get_bool(self, item):
        val = self.get(item)
        if 'FAILED000' in val:
            print("Invalid Parameter:", item)
            exit()
        if 'TRUE' in val.upper():
            val = True
        elif 'FALSE' in val.upper():
            val = False
        else:
            print("Error - Invalid Data Type:", item, "\nExpected Bool, given:", val)
            exit()
        return val

    def get_str(self, item):
        val = self.get(item)
        if 'FAILED000' in val:
            print("Invalid Parameter:", item)
            exit()
        return val

    def get_strlist(self, item):
        val = self.get(item)
        if 'FAILED000' in val:
            print("Invalid Parameter:", item)
            exit()
        return [val.strip() for val in val.split(',')]
