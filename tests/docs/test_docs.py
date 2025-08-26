from pathlib import Path

import pytest
from mktestdocs import check_md_file

paths = [p for p in Path("docs").glob("**/*.md") if "changelogs" not in p.parts]


@pytest.mark.docs
@pytest.mark.parametrize("fpath", paths, ids=str)
def test_docs(fpath):
    check_md_file(fpath=fpath, memory=True)


@pytest.mark.docs
def test_readme():
    check_md_file("README.md", memory=True)
