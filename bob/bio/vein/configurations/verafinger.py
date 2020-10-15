#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Mon 26 Sep 2016 17:21:42 CEST

"""`VERA Fingervein`_ is a database for biometric fingervein recognition

It consists of 440 images from 110 clients. It was produced at the Idiap
Research Institute in Martigny and at Haute Ecole Spécialisée de Suisse
Occidentale in Sion, in Switzerland. The reference citation is [TVM14]_.

You can download the raw data of the `VERA Fingervein`_ database by following
the link.
"""

from bob.extension import rc
from bob.bio.vein.database.verafinger import Database
from bob.bio.base.pipelines.vanilla_biometrics import DatabaseConnector

_verafinger_directory = rc["bob.db.verafinger.directory"]
"""Value in ``~/.bobrc`` for this dataset directory"""

protocol = 'Nom' # TODO protocol implementation in bob pipelines?
"""The default protocol to use for tests

We accept any biometric recognition protocol implemented by bob.db.verafinger.
Variants of the biometric recognition protocol ending in ``-va`` can be used to
test for vulnerability analysis. For example, use the protocol ``Nom-va`` to
test the vulnerability of a biometric recognition pipeline using the ``Nom``
protocol for enrollment and probe samples from presentation attacks.
"""

"""Updated with the wrapper for the pipelines package"""
database = DatabaseConnector(Database(
    original_directory=_verafinger_directory,
    original_extension='.png',
    protocol=protocol),

    annotation_type=None,
    fixed_positions=None
)
"""The :py:class:`bob.bio.base.database.BioDatabase` derivative with Verafinger
database settings, wrapped with the vanilla-biometrics database connector.

.. warning::

   This class only provides a programmatic interface to load data in an orderly
   manner, respecting usage protocols. It does **not** contain the raw
   datafiles. You should procure those yourself.

Notice that ``original_directory`` is set to
``rc[bob.db.verafinger.directory]``. You must make sure to set this value with
``bob config set bob.db.verafinger.directory`` to the place where you actually
installed the `vera fingervein`_ dataset, as explained in the section
:ref:`bob.bio.vein.baselines`.
"""
