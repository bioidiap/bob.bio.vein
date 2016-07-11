#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import bob.bio.base.grid import Grid


grid = Grid(
    training_queue='32G',

    number_of_preprocessings_per_job=200,
    preprocessing_queue='4G',

    number_of_extraction_jobs=200,
    extraction_queue='8G',

    number_of_projection_jobs=200,
    projection_queue='8G',

    number_of_enrollment_jobs=10,
    enrollment_queue='8G',

    number_of_scoring_jobs=10,
    scoring_queue='8G',
    )
