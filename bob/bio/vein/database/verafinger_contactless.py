#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Victor <vbros@idiap.ch>

"""
  VERA-Fingervein-Contactless database implementation
"""

from bob.bio.base.database import CSVDataset
from bob.bio.base.database import CSVToSampleLoaderBiometrics
from bob.extension import rc
from bob.extension.download import get_file
import bob.io.image


class VerafingerContactless(CSVDataset):

    def __init__(self, protocol):
        urls = VerafingerContactless.urls()
        filename = get_file(
            "verafinger_contactless.tar.gz",
            urls,
            file_hash="c664a83b8fcba3396b010c4d3e60e425e14b32111c4b955892072e5d687485bd",
        )

        super().__init__(
            name="verafinger_contactless",
            dataset_protocol_path=filename,
            protocol=protocol,
            csv_to_sample_loader=CSVToSampleLoaderBiometrics(
                data_loader=bob.io.image.load,
                dataset_original_directory=rc.get(
                    "bob.bio.vein.verafinger_contactless.directory", ""
                ),
                extension='',
                reference_id_equal_subject_id=False
            ),
            allow_scoring_with_all_biometric_references=True,
        )

    @staticmethod
    def protocols():
        # TODO: Until we have (if we have) a function that dumps the protocols, let's use this one.
        return ["nom"]

    @staticmethod
    def urls():
        return ["https://www.idiap.ch/software/bob/databases/latest/verafinger_contactless-c664a83b.tar.gz",
                "http://www.idiap.ch/software/bob/databases/latest/verafinger_contactless-c664a83b.tar.gz",
                ]
