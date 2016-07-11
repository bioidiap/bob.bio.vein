#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import bob.bio.base.grid import Grid


grid = Grid(
    training_queue = '8G',

    number_of_preprocessing_jobs = 32,
    preprocessing_queue = '4G-io-big',

    number_of_extraction_jobs = 32,
    extraction_queue = '4G-io-big',

    number_of_projection_jobs = 32,
    projection_queue = {},

    number_of_enrollment_jobs = 32,
    enrollment_queue = {},

    number_of_scoring_jobs = 32,
    scoring_queue = '4G-io-big',
    )
