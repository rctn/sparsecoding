from sparsecoding.test_utils import (bars_datas_fixture, bars_datasets_fixture,
                                     bars_dictionary_fixture,
                                     dataset_size_fixture, patch_size_fixture,
                                     priors_fixture)

# We import and define all fixtures in this file.
# This allows users to avoid any dependency fixtures.
# NOTE: This means pytest should only be run from this directory.
__all__ = ['dataset_size_fixture', 'patch_size_fixture', 'bars_datas_fixture', 'bars_datasets_fixture', 'bars_dictionary_fixture', 'priors_fixture']
