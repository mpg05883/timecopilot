from pathlib import Path

import pytest
from mktestdocs import check_md_file


@pytest.mark.docs
@pytest.mark.parametrize(
    "fpath",
    Path("docs").glob("**/*.md"),
    ids=str,
)
def test_docs(fpath):
    check_md_file(fpath=fpath, memory=True)


@pytest.mark.docs
def test_readme():
    check_md_file("README.md", memory=True)
