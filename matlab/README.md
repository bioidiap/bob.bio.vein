# Maximum Curvature and Miura Matching (Cross-correlation)

This directory contains scripts to generate references from the Matlab library
for Repeated Line Tracking (RLT) and Maximum Curvature extraction from B.Ton.
The original source code download link is here: http://ch.mathworks.com/matlabcentral/fileexchange/35716-miura-et-al-vein-extraction-methods


## Usage Instructions

Make sure you have the `matlab` binary available on your path. At Idiap, this
can be done by executing:

```sh
$ SETSHELL matlab
$ which matlab
/idiap/resource/software/matlab/stable/bin/matlab
```

Call `run.sh`, that will perform the maximum curvature on the provided image,
and output various control variables in various HDF5 files:

```sh
$ run.sh ../bob/bio/vein/tests/extractors/image.hdf5 ../bob/bio/vein/tests/extractors/mask.hdf5 mc
...
```

Or, a quicker way, without contaminating your environment:

```sh
$ setshell.py matlab ./run.sh ../bob/bio/vein/tests/extractors/image.hdf5 ../bob/bio/vein/tests/extractors/mask.hdf5 mc
...
```

The program `run.sh` calls the function `lib/maximum_curvature.m`, which
processes and dumps results to output files.

After running, this program produces 5 files:

* `*_kappa_matlab.hdf5`: contains the kappa variable
* `*_v_matlab.hdf5`: contains the V variable
* `*_vt_matlab.hdf5`: contains the accumulated Vt variable
* `*_cd_matlab.hdf5`: contains the Cd variable
* `*_g_matlab.hdf5`: contains the accumulated Cd variable called G (paper)
* `*_bin_matlab.hdf5:`: the final binarised results (G thresholded)

Once you have generated the references, you can compare the Maximum Curvature
algorithm implemented in Python against the one in Matlab with the following
command:

```sh
$ ../bin/python compare.py
Comparing kappa[0]: 2.51624726501e-14
Comparing kappa[1]: 2.10662186414e-13
Comparing kappa[2]: 1.09901515494e-13
Comparing kappa[3]: 1.0902340027e-13
Comparing V[0]: 1.04325752096e-14
Comparing V[1]: 8.5519523202e-14
Comparing V[2]: 9.20009110336e-05
Comparing V[3]: 4.02339072347e-14
Comparing Vt: 9.20009111675e-05
Comparing Cd[0]: 1.08636347658e-13
Comparing Cd[1]: 2.8698038577e-14
Comparing Cd[2]: 3.40774680626e-14
Comparing Cd[3]: 3.2892922132e-14
Comparing G: 1.57966982699e-13
```

The program displays the sum of absolute differences between the Matlab
reference and Python's.
