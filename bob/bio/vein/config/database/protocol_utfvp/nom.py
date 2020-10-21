# author: Yannick Dayer <yannick.dayer@idiap.ch>
# Wed 21 Oct 2020 10:30:22 UTC+02

# This is a config file for bob.bio.vein
# It defines the database protocol to use with the Database Interface for the
# utfvp dataset, defined at bob.db.utfvp.

# It is defined as a resource in the setup file of this package.

# Usage:
# $ bob bio pipelines vanilla-biometrics <pipeline> nom utfvp
# or:
# $ bob bio pipelines vanilla-biometrics -p <pipeline> nom utfvp

# The protocol resource must be specified before the database resource.


# Available protocols are (some require the creation of your own config file):
# '1vsall', 'full', 'fullLeftIndex', 'fullLeftMiddle', 'fullLeftRing',
# 'fullRightIndex', 'fullRightMiddle', 'fullRightRing', 'nom', 'nomLeftIndex',
# 'nomLeftMiddle', 'nomLeftRing', 'nomRightIndex', 'nomRightMiddle',
# 'nomRightRing'


# This will be the default protocol if none is specified.

protocol = 'nom'

