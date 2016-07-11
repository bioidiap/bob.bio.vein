#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import bob.bio.base.grid import Grid


grid = Grid(
    training_queue = '8G',

    number_of_preprocessing_jobs = 1000,
    preprocessing_queue = {},

    number_of_extraction_jobs = 1000,
    extraction_queue = {},

    number_of_projection_jobs = 1000,
    projection_queue = {},

    number_of_enrollment_jobs = 100,
    enrollment_queue = '2G',

    number_of_scoring_jobs = 1500,
    scoring_queue = {'queue': 'q_gpu'},
    )
