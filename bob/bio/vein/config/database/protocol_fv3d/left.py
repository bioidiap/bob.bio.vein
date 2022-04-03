# author: Yannick Dayer <yannick.dayer@idiap.ch>
# Fri 16 Oct 2020 12:14:05 UTC+02

# This is a config file for bob.bio.vein
# It defines the database protocol to use with the Database Interface for the
# 3D Fingervein dataset, defined at bob.db.fv3d.

# It is defined as a resource in the setup file of this package.

# Usage:
# $ bob bio pipeline simple<pipeline> left fv3d
# or:
# $ bob bio pipeline simple -p <pipeline> left fv3d

# The protocol resource must be specified before the database resource.


# Available protocols are:
# 'central', 'left', 'right', 'stitched'

protocol = "left"
