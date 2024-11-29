SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

livehtml:
	rm -rf source/api/reference
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)"/html $(SPHINXOPTS) $(O)

check:
	pylint --rcfile=../.pylintrc --enable=C0114,C0115,C0116 ../spaceai

.PHONY: help Makefile

%: Makefile
	rm -rf source/api/reference
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
