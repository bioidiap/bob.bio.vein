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
    """
    The VERA Fingervein Contactless database contains 1330 finger vein images of 133 persons,
    with id ranging from 1 to 137 (with 4 defects).
    Both hands of the subjects have been imaged over 5 different sessions, with a total of 10 images per person.
    The sensor is described in `this paper <https://ieeexplore.ieee.org/abstract/document/7314596>`_.

    .. warning::

      To use this dataset protocol, you need to have the original files of the VERA Fingervein Contactless dataset.
      Once you have it downloaded, please run the following command to set the path for Bob

        .. code-block:: sh

            bob config set bob.bio.vein.verafinger_contactless.directory [DATABASE PATH]


    **Metadata**

    Associated to each sample, you may find the metadata configuration for each capture :

    * EXPOSURE : exposure time (in ms)
    * ORTHOLED : Power of the ORTHOLED (in % of the max power)
    * CENTERLED : Power of the CENTERLED (in % of the max power)
    * CROSSLED : Power of the CROSSLED (in % of the max power)


    **Protocols**

    **NOM (Normal Operation Mode) protocol**

    * Development set : Even subject ids
    * Evaluation set : Odd subject ids
    * Enrolled : session 1 & 2
    * Probes : session 3, 4 &5

    """

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
