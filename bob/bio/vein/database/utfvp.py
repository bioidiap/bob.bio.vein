#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Victor <vbros@idiap.ch>

"""
  Utfvp database implementation
"""

from bob.bio.base.database import CSVDataset
from bob.bio.base.database import CSVToSampleLoaderBiometrics
from bob.extension import rc
from bob.extension.download import get_file
import bob.io.image


class UtfvpDatabase(CSVDataset):
    """
    The University of Twente Finger Vascular Pattern dataset
    """

    def __init__(self, protocol):
        # Downloading model if not exists
        urls = UtfvpDatabase.urls()
        filename = get_file(
            "utfvp_csv.tar.gz",
            urls,
            file_hash="0b22a4ea6a78d54879dc3d866a22108b7513d169cd5c1bceb044854a871f200a",
        )

        super().__init__(
            name="utfvp",
            dataset_protocol_path=filename,
            protocol=protocol,
            csv_to_sample_loader=CSVToSampleLoaderBiometrics(
                data_loader=bob.io.image.load,
                dataset_original_directory=rc.get(
                    "bob.bio.vein.utfvp.directory", ""
                ),
                extension='',
                reference_id_equal_subject_id=False
            ),
            allow_scoring_with_all_biometric_references=True,
        )

    @staticmethod
    def protocols():
        # TODO: Until we have (if we have) a function that dumps the protocols, let's use this one.
        return ["nom", "full", "1vsall",
                "nomLeftRing", "nomRightRing", "nomLeftMiddle", "nomRightMiddle", "nomLeftIndex", "nomRightIndex",
                "fullLeftRing", "fullRightRing", "fullLeftMiddle", "fullRightMiddle", "fullLeftIndex", "fullRightIndex"]

    @staticmethod
    def urls():
        return ["https://www.idiap.ch/software/bob/databases/latest/utfvp_csv-0b22a4ea.tar.gz",
                "http://www.idiap.ch/software/bob/databases/latest/utfvp_csv-0b22a4ea.tar.gz",
                ]
