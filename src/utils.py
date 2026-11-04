#! /usr/bin/python3
# Author : Kevin Feghoul

import os
import pickle as pkl
from typing import NoReturn


# ======================================================================================================================================
# Gestion de fichier pickle
# ======================================================================================================================================

# Récupérer les données d'un fichier pkl
def load_pickle(filename: str):
    '''Save the data in the pickle (pkl) format.'''
    with open(filename, 'rb') as handle:
        data = pkl.load(handle)
    return data

# Sauvegarder les données dans un fichier pickle
def save_pickle(filename: str, data: dict) -> None:
    '''Save the data in the pickle (pkl) format.'''
    with open(filename, 'wb') as handle:
        pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)


# ======================================================================================================================================
# Gestion des répertoires
# ======================================================================================================================================

# Création d'un répertoire
def make_dirs(path: str):
    '''Make directory if not exists.'''
    if not os.path.exists(path):
        os.makedirs(path)


# ======================================================================================================================================
# Gestion des fichiers de résultats (de tous les modèles pour une catégorie)
# ======================================================================================================================================

# Ajout d'une nouvelle ligne à la fin d'un fichier
def append_new_line(filename :str, text_to_append) -> NoReturn:
    '''Append given text as a new line at the end of the file.'''
    # Open the file in append & read mode ('a+')
    with open(filename, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)











































