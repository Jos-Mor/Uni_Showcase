k-Means clustering algorithm

EduHPC 2023: Peachy assignment

(c) 2022 Diego García-Álvarez, Arturo Gonzalez-Escribano
Group Trasgo, Universidad de Valladolid (Spain)

--------------------------------------------------------------

Group Members:

Guilherme Antunes nº 70231
José Morgado nº 59457
Pedro Mascarenhas nº 63240

--------------------------------------------------------------

To compile our code simply use the command "make all".

To run the compiled code just use the default arguments:
input_file number_of_cluster maximum_number_of_iterations number_of_data_point_changes maximum_centroid_movement output_file

We provide a python script to automatically run tests such as running several params inputs, verify correctness, compare results and plot a graph.
There is a more detailed explanation on what the script does in the report and in the start of test_script.py itself.

Before running our python script you will need to install matplotlib, we use this library to draw a simple bar graph that shows the difference in execution times between the sequential and parallelized versions of our code.



--------------------------------------------------------------

Read the handout and use the sequential code as reference to study.
Use the other source files to parallelize with the proper programming model.

Edit the first lines in the Makefile to set your preferred compilers and flags
for both the sequential code and for each parallel programming model: 
OpenMP, MPI, and CUDA.

To see a description of the Makefile options execute:
$ make help 

Use the input files in the test_files directory for your first tests.
Students are encouraged to manually write or automatically generate
their own input files for more complete tests. See a description of
the input files format in the handout.

