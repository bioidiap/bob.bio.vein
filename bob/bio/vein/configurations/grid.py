#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from bob.bio.base.grid import Grid

# our preferred grid setup for Idiap
default = Grid(
    training_queue='32G',

    number_of_preprocessing_jobs=200,
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
