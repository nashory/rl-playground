#!/usr/bin/python
#
# myobjimport

import os.path, sys
from cgkit.cgtypes import *
from cgkit.objimport import *
from cgkit.objimport import _OBJReader
from cgkit.pluginmanager import *

class MyOBJImporter(OBJImporter):

    _protocols = ["Import"]
    _reader = None

    # extension
    def extension():
        """
        Return the file extensions for this format.
        """
        return ["obj"]
    extension = staticmethod(extension)

    # description
    def description(self):
        """
        Return a short description for the file dialog.
        """
        return "Wavefront object file"
    description = staticmethod(description)

    # importFile
    def importFile(self, filename, parent=None):
        """
        Import an OBJ file.
        """
        f = file(filename)
        self._reader = _OBJReader(root=None)
        self._reader.read(f)

################################################################################

# Register the MyObjImporter class as a plugin class
pluginmanager.register(MyOBJImporter)

