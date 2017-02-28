#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from bob.bio.base.grid import Grid

# our preferred grid setup for Idiap
default = Grid(
    training_queue='32G',

    number_of_preprocessing_jobs=32,
    preprocessing_queue='8G-io-big',

    number_of_extraction_jobs=32,
    extraction_queue='8G-io-big',

    number_of_projection_jobs=32,
    projection_queue='8G-io-big',

    number_of_enrollment_jobs=32,
    enrollment_queue='8G-io-big',

    number_of_scoring_jobs=32,
    scoring_queue='8G-io-big',
    )



# prefered setup if score computation is slow:
idiap_speedup_score = Grid(
    training_queue='32G',

    number_of_preprocessing_jobs=200,
    preprocessing_queue='4G-io-big',

    number_of_extraction_jobs=200,
    extraction_queue='8G-io-big',

    number_of_projection_jobs=200,
    projection_queue='8G-io-big',

    number_of_enrollment_jobs=200,
    enrollment_queue='8G-io-big',

    number_of_scoring_jobs=200,
    scoring_queue='8G-io-big',
    )

# prefered setup if score computation is slow:
idiap_user_machines_speedup_score = Grid(
    training_queue='32G',

    number_of_preprocessing_jobs=32,
    preprocessing_queue='4G',

    number_of_extraction_jobs=32,
    extraction_queue='8G',

    number_of_projection_jobs=32,
    projection_queue='8G',

    number_of_enrollment_jobs=32,
    enrollment_queue='8G',

    number_of_scoring_jobs=50,
    scoring_queue='8G',
    )

# prefered setup if score computation is slow:
idiap_q_all = Grid(
    training_queue='4G',

    number_of_preprocessing_jobs=200,
    preprocessing_queue='4G',

    number_of_extraction_jobs=200,
    extraction_queue='4G',

    number_of_projection_jobs=200,
    projection_queue='4G',

    number_of_enrollment_jobs=200,
    enrollment_queue='4G',

    number_of_scoring_jobs=200,
    scoring_queue='4G',
    )


idiap_q_all_modest = Grid(
    training_queue='4G',

    number_of_preprocessing_jobs=32,
    preprocessing_queue='4G',

    number_of_extraction_jobs=32,
    extraction_queue='4G',

    number_of_projection_jobs=32,
    projection_queue='4G',

    number_of_enrollment_jobs=32,
    enrollment_queue='4G',

    number_of_scoring_jobs=32,
    scoring_queue='4G',
    )


week = Grid(
    training_queue='Week',
    
    number_of_preprocessing_jobs=32,
    preprocessing_queue='Week',

    number_of_extraction_jobs=32,
    extraction_queue='Week',

    number_of_projection_jobs=32,
    projection_queue='Week',

    number_of_enrollment_jobs=32,
    enrollment_queue='Week',

    number_of_scoring_jobs=32,
    scoring_queue='Week',
    )
