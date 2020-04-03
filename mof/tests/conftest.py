import mof, pytest, os, sys, _pickle
from os.path import join, dirname, abspath, exists

# addoption doesn't work for me
# def pytest_addoption(parser):
#     parser.addoption(
#         "--runslow", action="store_true", default=False, help="run slow tests"
#     )
#
#
# def pytest_collection_modifyitems(config, items):
#     if config.getoption("--runslow"):
#         # --runslow given in cli: do not skip slow tests
#         return
#     skip_slow = pytest.mark.skip(reason="need --runslow option to run")
#     for item in items:
#         if "slow" in item.keywords:
#             item.add_marker(skip_slow)

@pytest.fixture(scope="session")
def data_dir():
   assert exists(mof.data.data_dir)
   return data_dir

@pytest.fixture(scope="session")
def c3_peptide():
   return mof.data.c3_peptide()

@pytest.fixture(scope="session")
def rotcloud_asp():
   return mof.data.rotcloud_asp_small()

@pytest.fixture(scope="session")
def rotcloud_his():
   return mof.data.rotcloud_his_small()
