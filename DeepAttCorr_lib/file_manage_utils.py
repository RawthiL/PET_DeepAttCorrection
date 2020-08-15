#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

import shutil, os


def check_create_path(path_name, path_loc, clear_folder=False):
    if not os.path.exists(path_loc):
        print('Creating %s: %s'%(path_name,path_loc))
        os.makedirs(path_loc)
    else:
        print('%s: %s -- Exists'%(path_name,path_loc))
        if clear_folder:
            print('\t\t Clearing...')
            shutil.rmtree(path_loc)
            check_create_path(path_name, path_loc, clear_folder=False)
            
            
