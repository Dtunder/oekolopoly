import sys
import os
from cx_Freeze import setup, Executable

# File for creating standalone executable for oekolopoly 2.0
# invoke: python setup.py build

files_to_include = ['oekolopoly/my_oekolopoly_gui/assets', 'oekolopoly/my_oekolopoly_gui/trained_agents',
                    'oekolopoly/my_oekolopoly_gui/bin', 'oekolopoly/my_oekolopoly_gui/Spielanleitung.pdf',
                    'oekolopoly/my_oekolopoly_gui/feedback.txt','oekolopoly/my_oekolopoly_gui/current_game_history.txt']

packages_to_include = ['oekolopoly']

executable = Executable(
    script="oekolopoly/my_oekolopoly_gui/oekolopoly_gui.py",
    base="gui",
    icon="oekolopoly/my_oekolopoly_gui/assets/python_icon.ico"
)

setup(
    name="Oekolopoly_v1",
    version="1.0",
    description="Oekolopoly",
    author="Me",
    options={'build_exe': {'include_files': files_to_include, 'packages': packages_to_include}},
    executables=[executable]
)
