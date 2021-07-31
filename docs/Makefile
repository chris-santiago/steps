# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = steps
SOURCEDIR     = source
BUILDDIR      = build
SPHINXAPI     = sphinx-apidoc

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

html: apidoc
	$(SPHINXBUILD) -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html

html-fail-on-warnings: apidoc
	$(SPHINXBUILD) -b html -W --keep-going $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(SPHINXOPTS) $(SOURCEDIR) ./

apidoc:
	$(SPHINXAPI) -f ../$(SPHINXPROJ) -o source/
	@echo "regenerated API docs in ./source"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
