#!/usr/bin/env bash

if [ $# == 0 ]; then
  echo "usage: $0 input_image.mat input_region.mat output_stem"
  exit 1
fi

_matlab=`which matlab`;

if [ -z "${_matlab}" ]; then
  echo "I cannot find a matlab binary to execute, please setup first"
  exit 1
fi

# Does some environment manipulation to get paths right before we start
_basedir="$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
_matlabpath=${_basedir}/lib
unset _basedir;
if [ -n "${MATLABPATH}" ]; then
  _matlabpath=${_matlabpath}:${MATLABPATH}
fi
export MATLABPATH=${_matlabpath};
unset _matlabpath;

# Calls matlab with our inputs
${_matlab} -nodisplay -nosplash -nodesktop -r "image = im2double(hdf5read('${1}','/array')); region = double(hdf5read('${2}','/array')); miura_max_curvature(image, region, 3, '${3}'); quit;"
