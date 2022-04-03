# author: Yannick Dayer <yannick.dayer@idiap.ch>
# Wed 21 Oct 2020 11:27:08 UTC+02

# This is a config file for bob.bio.vein
# It defines the database protocol to use with the Database Interface for the
# Put Vein dataset, defined at bob.db.putvein.

# It is defined as a resource in the setup file of this package.

# Usage:
# $ bob bio pipeline simple <pipeline> wrist-LR-1 putvein
# or:
# $ bob bio pipeline simple -p <pipeline> wrist-LR-1 putvein

# The protocol resource must be specified before the database resource.


# Available protocols are (some require the creation of your own config file):
# 'palm-L_1', 'palm-LR_1', 'palm-R_1', 'palm-RL_1', 'palm-R_BEAT_1',
# 'palm-L_4', 'palm-LR_4', 'palm-R_4', 'palm-RL_4', 'palm-R_BEAT_4',
# 'wrist-L_1', 'wrist-LR_1', 'wrist-R_1', 'wrist-RL_1', 'wrist-R_BEAT_1',
# 'wrist-L_4', 'wrist-LR_4', 'wrist-R_4', 'wrist-RL_4', 'wrist-R_BEAT_4'


# This will be the default protocol if none is specified.

protocol = "wrist-LR_1"
