Quick Start
===========

This page is intended to be a user guide for the developers to quickly
be able to use/re-use the codebase for their own project 
or, extending/enhancing the codebase for better performing models.

Pre-requisities
---------------

Knowledge pre-requisites

* Python: The code is written in Python and thus, some prior knowledge of Python programming is useful.

* PostGres: The database format of MSR2020 is postgres and hence, familiarity of postgresql is essential for getting started with this project.

Library pre-requisites

* psycopg2 - postgres and python connection adapter library to access the database.

* sklearn - machine learning model implementations like linear regression.

* matplotlib - visualizations of the plots.

* sphinx and sphinx-rtd-theme - Python auto-documentation generator.

* PyCharm - or, any other IDE to edit the code.

Getting Started
---------------

In the main repository, there exists a python file called `main.py`.

Create a `database.ini` file and store your config there. The config must include the following:

.. code-block::

    [postgresql]
    host=<host-name>
    port=<port-number>
    database=<database-name>
    user=<user-name>
    password=<password>


Go to terminal or any IDE and run that file using the following command:

.. code-block::

    python main.py


Reference
---------

In case you use this code, please cite our repository in your work.
