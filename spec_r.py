# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:17:46 2019

@author: Admin
"""

import ffmpeg
import os
import subprocess
import csv
import pandas as pd

from os import listdir
from os.path import isfile, join


class Spec:
    # calling R from Python
    # Define command and arguments
    command = 'Rscript'

    def __init__(self, path, temp=os.getcwd()+'/data/tmpcsv/', path2script='C:/R/spectr.R'):
        self.path = path
        self.temp = os.getcwd() + temp
        self.path2script = path2script
        self.ff = ffmpeg.FFmpeg(self.path)

    def find_voice(self):
        '''find actual current list of voice file in directory -> []'''
        return [
            _ for _ in listdir(self.path) if isfile(join(
                self.path, _))and (_.endswith('mp3') or _.endswith('wav'))]

    def spec(self):
        file_list = self.find_voice()
        # Variable number of args in a list
        for name in file_list:
            duration = self.ff.duration(self.path + name)
            # temp = os.getcwd() + '/data/tmpcsv/'
            args = [name, duration, self.path, self.temp]
            # Build subprocess command
            cmd = [Spec.command, self.path2script] + args
            # check_output will run the command and store to result
            p = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, shell=True)
            output, _ = p.communicate()
            spec = output.decode('ascii').split()
            print(name)


if __name__ == '__main__':
    path = os.getcwd() + '\\data\\voice\\clips\\'
    spec = Spec(path)
    spec.spec()
