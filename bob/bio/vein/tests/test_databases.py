#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Thu May 24 10:41:42 CEST 2012

from nose.plugins.skip import SkipTest

import bob.bio.base
from bob.bio.base.test.utils import db_available
from bob.bio.base.test.test_database_implementations import check_database


@db_available('utfvp')
def test_utfvp():
    database = bob.bio.base.load_resource('utfvp', 'database', preferred_package='bob.bio.vein')
    try:
        check_database(database, protocol='nomLeftRing', groups=('dev', 'eval'))
    except IOError as e:
        raise SkipTest(
            "The database could not queried; probably the db.sql3 file is missing. Here is the error: '%s'" % e)


@db_available('verafinger')
def test_verafinger():
    database = bob.bio.base.load_resource('verafinger', 'database', preferred_package='bob.bio.vein')
    try:
        check_database(database, protocol='Fifty', groups=('dev', 'eval'))
    except IOError as e:
        raise SkipTest(
            "The database could not queried; probably the db.sql3 file is missing. Here is the error: '%s'" % e)

@db_available('biowave_test')
def test_biowave_test():
    #database = bob.bio.base.load_resource('biowave_test', 'database', preferred_package='bob.bio.vein')
    module = bob.bio.base.load_resource('biowave_test', 'config', preferred_package='bob.bio.vein')
    try:
        #check_database(database, protocol='all', groups=('dev', 'eval'), skip_train=True)
        check_database(module.database, protocol=module.protocol, groups=('dev', 'eval'), skip_train=True)
    except IOError as e:
        raise SkipTest(
            "The database could not queried; probably the db.sql3 file is missing. Here is the error: '%s'" % e)

@db_available('biowave_v1')
def test_biowave_v1():
    #database = bob.bio.base.load_resource('biowave_v1', 'database', preferred_package='bob.bio.vein')
    module = bob.bio.base.load_resource('biowave_v1', 'config', preferred_package='bob.bio.vein')
    try:
        #check_database(database, protocol='Idiap_1_1_R', groups=('dev', 'eval'))
        check_database(module.database, protocol=module.protocol, groups=('dev', 'eval'))
    except IOError as e:
        raise SkipTest(
            "The database could not queried; probably the db.sql3 file is missing. Here is the error: '%s'" % e)
