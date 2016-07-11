#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import bob.bio.base.grid import Grid


grid = Grid(
    training_queue = '64G',

    number_of_preprocessing_jobs = 100,
    preprocessing_queue = '4G',

    number_of_extraction_jobs = 100,
    extraction_queue = '8G-io-big',

    number_of_projection_jobs = 20,
    projection_queue = '8G-io-big',

    number_of_enrollment_jobs = 2,
    enrollment_queue = '8G-io-big',

    number_of_scoring_jobs = 1,
    scoring_queue = '8G-io-big'
    )
