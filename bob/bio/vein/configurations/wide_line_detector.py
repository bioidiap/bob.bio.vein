#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tue 27 Sep 2016 16:48:16 CEST

"""Huang's Wide-Line Detector and Miura Matching baseline

References:

1. [HDLTL10]_
2. [TV13]_
3. [TVM14]_

"""

from bob.bio.base.transformers import PreprocessorTransformer
from bob.bio.base.transformers import ExtractorTransformer
from bob.bio.base.pipelines.vanilla_biometrics import (
    VanillaBiometricsPipeline,
    BioAlgorithmLegacy,
)
from sklearn.pipeline import make_pipeline
from bob.pipelines import wrap

from bob.bio.vein.preprocessor import (
    NoCrop,
    TomesLeeMask,
    HuangNormalization,
    NoFilter,
    Preprocessor,
)

preprocessor = PreprocessorTransformer(
    Preprocessor(
        crop=NoCrop(),
        mask=TomesLeeMask(),
        normalize=HuangNormalization(),
        filter=NoFilter(),
    )
)


"""Preprocessing using gray-level based finger cropping and no post-processing
"""

from bob.bio.vein.extractor import WideLineDetector

extractor = ExtractorTransformer(WideLineDetector())
"""Features are the output of the maximum curvature algorithm, as described on
[HDLTL10]_.

Defaults taken from [TV13]_.
"""

# Notice the values of ch and cw are different than those from the
# repeated-line tracking **and** maximum curvature baselines.
from bob.bio.vein.algorithm import MiuraMatch

biometric_algorithm = BioAlgorithmLegacy(
    MiuraMatch(ch=18, cw=28), base_dir="/idiap/temp/vbros/pipeline_test/verafinger"
)
"""Miura-matching algorithm with specific settings for search displacement

Defaults taken from [TV13]_.
"""
transformer = make_pipeline(wrap(["sample"], preprocessor), wrap(["sample"], extractor))

pipeline = VanillaBiometricsPipeline(transformer, biometric_algorithm)
