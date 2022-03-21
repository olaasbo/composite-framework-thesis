The **Grasshopper** definition requires the use of the following plug-ins (requested when opening the .gh-file):
      >>> Karamba3D     > https://www.karamba3d.com/
      >>> GOAT          > https://www.food4rhino.com/en/app/goat

The ghPython components makes use of external modules that are not natively available. Some of these modules, such as numpy, requires the use of 
"ghpythonremote", see: https://github.com/pilcru/ghpythonremote. However, numpy and similar packages are extremely slow in cPython, thus the use is limited to plotting and Post-processing. 

Therefore, the folder with site-packages should be 
made available to Rhino. Save the folder to a path, e.g. "C:\RhinoPython". 
In order for Rhino to access these site-packages,
>>> In Rhino: 
      Tools > PythonScript > Edit 
      >>> In the Rhino Python Editor:
            Tools > Options... > Copy path to the "Module Search Paths"
      Additional packages must can be installed with Python 2.7 and pip, before adding these to the directory
