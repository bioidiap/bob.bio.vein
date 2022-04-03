# author: Yannick Dayer <yannick.dayer@idiap.ch>
# Fri 16 Oct 2020 14:51:16 UTC+02

# This is a config file for bob.bio.vein
# It defines the database protocol to use with the Database Interface for the
# VeraFinger dataset, defined at bob.db.verafinger.

# It is defined as a resource in the setup file of this package.

# Usage:
# $ bob bio pipeline simple <pipeline> Nom verafinger
# or:
# $ bob bio pipeline simple -p <pipeline> Nom verafinger

# The protocol resource must be specified before the database resource.


# Available protocols are (some require the creation of your own config file):
# 'B', 'Cropped-B', 'Cropped-Fifty', 'Cropped-Full', 'Cropped-Nom', 'Fifty',
# 'Full', 'Nom'

# This will be the default protocol if none is specified.

protocol = "Nom"
