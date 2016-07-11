#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import bob.bio.base.grid import Grid


grid = Grid(
  grid = 'local',
  number_of_parallel_processes = 4
)

grid_p16 = Grid(
  number_of_preprocessing_jobs = 50,
  number_of_extraction_jobs = 50,
  number_of_projection_jobs = 50,
  number_of_enrollment_jobs = 10,
  number_of_scoring_jobs = 10,
  grid = 'local',
  number_of_parallel_processes = 4
)
