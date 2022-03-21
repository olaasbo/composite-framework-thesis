The Grasshopper definition requires the use of the following plug-ins (also notified when opening the .gh-file):
>>> Karamba3D

The ghPython components makes use of external modules that are not natively available. Therefore, the folder with site-packages should be 
made available to Rhino. Save the folder to a path, e.g. "C:\RhinoPython". 
In order for Rhino to access these site-packages,
>>> In Rhino: 
      Tools > PythonScript > Edit 
      >>> In the Rhino Python Editor:
            Tools > Options... > Copy path to the "Module Search Paths"
