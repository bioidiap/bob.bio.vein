# author: Yannick Dayer <yannick.dayer@idiap.ch>
# Fri 16 Oct 2020 12:14:56 UTC+02

# This is a config file for bob.bio.vein
# It defines the database protocol to use with the Database Interface for the
# 3D Fingervein dataset, defined ar bob.db.fv3d.

# It is defined as a resource in the setup file of this package.

# Usage:
# $ bob bio pipelines vanilla-biometrics <pipeline> right fv3d
# or:
# $ bob bio pipelines vanilla-biometrics -p <pipeline> right fv3d

# The protocol resource must be specified before the database resource.

protocol = 'right'
