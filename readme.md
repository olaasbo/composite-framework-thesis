# Grasshopper Plug-ins
The **Grasshopper** definition requires the use of the following plug-ins (requested when opening the `.gh-file`):
- Karamba3D     > https://www.karamba3d.com/
- GOAT          > https://www.food4rhino.com/en/app/goat

# Installing required Python site-packages
The ghPython components makes use of external modules that are not natively available. Some of these modules, such as numpy, requires the use of 
"ghpythonremote", see: https://github.com/pilcru/ghpythonremote. However, numpy and similar packages are extremely slow in cPython, thus the use is limited to plotting and Post-processing. 

To make the folder with `site-packages` available to Rhino, save it to a directory such as `"C:\RhinoPython"`. 
Then, In Rhino: 
- Tools > PythonScript > Edit 
In the Rhino Python Editor:
- Tools > Options... > Copy path to the "Module Search Paths"

Additional packages can be installed with Python 2.7 and pip, before adding these to the directory

# Python scripts for Post-processing
The Grasshopper definition generates an Excel file -- both for global and local optimisation -- which are handled with different 
Pythonscripts to create plots and visualise the results. 
