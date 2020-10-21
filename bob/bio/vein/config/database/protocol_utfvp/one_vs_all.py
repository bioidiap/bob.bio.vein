# author: Yannick Dayer <yannick.dayer@idiap.ch>
# Wed 21 Oct 2020 10:34:41 UTC+02

# This is a config file for bob.bio.vein
# It defines the database protocol to use with the Database Interface for the
# utfvp dataset, defined at bob.db.utfvp.

# It is defined as a resource in the setup file of this package.

# Usage:
# $ bob bio pipelines vanilla-biometrics <pipeline> 1vsall utfvp
# or:
# $ bob bio pipelines vanilla-biometrics -p <pipeline> 1vsall utfvp

# The protocol resource must be specified before the database resource.


# Available protocols are (some require the creation of your own config file):
# '1vsall', 'full', 'fullLeftIndex', 'fullLeftMiddle', 'fullLeftRing',
# 'fullRightIndex', 'fullRightMiddle', 'fullRightRing', 'nom', 'nomLeftIndex',
# 'nomLeftMiddle', 'nomLeftRing', 'nomRightIndex', 'nomRightMiddle',
# 'nomRightRing'


protocol = 'full'

