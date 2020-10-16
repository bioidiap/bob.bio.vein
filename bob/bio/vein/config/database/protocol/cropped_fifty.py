# author: Yannick Dayer <yannick.dayer@idiap.ch>
# Fri 16 Oct 2020 14:50:54 UTC+02

# This is a config file for bob.bio.vein
# It defines the database protocol to use with the Database Interface for the
# VeraFinger dataset, defined ar bob.db.verafinger.

# It is defined as a resource in the setup file of this package.

# Usage:
# $ bob bio pipelines vanilla-biometrics <pipeline> Cropped-Fifty verafinger
# or:
# $ bob bio pipelines vanilla-biometrics -p <pipeline> Cropped-Fifty verafinger

# The protocol resource must be specified before the database resource.

protocol = 'Cropped-Fifty'
