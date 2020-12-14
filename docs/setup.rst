#####
Setup
#####

This is some setup hints for a better experience !

Global variables
================

If you use anaconda and work in an environment, then before launching any code,
this is a simple solution:

* create a file named `odeon_setup.sh`
* make it executable : `chmod 755 odeon_setup.sh`
* check the path for `proj.db` and `gdal` in your conda environment (`find . -name proj.db`)

Then put these lines in your script (update the path):

.. code-block:: bash
   
   export GDAL_DATA="/home/nil/anaconda3/envs/dlxx/share/gdal"
   export PROJ_LIB="/home/nil//anaconda3/envs/dlxx/share/proj/proj.db"

Before playing with ODEON, launch the script.
