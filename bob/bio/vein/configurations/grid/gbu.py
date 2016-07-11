#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import bob.bio.base.grid import Grid


grid = Grid(
    training_queue = '32G',

    number_of_preprocessing_jobs = 1000,
    preprocessing_queue = '8G',

    number_of_extraction_jobs = 100,
    extraction_queue = '8G',

    number_of_projection_jobs = 100,
    projection_queue = '8G',

    number_of_enrollment_jobs = 100,
    enrollment_queue = '8G',

    number_of_scoring_jobs = 10,
    scoring_queue = '8G'
    )
