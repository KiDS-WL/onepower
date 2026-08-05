Developing OnePower
===================

If you're interested in developing OnePower -- welcome! You should first read
the guide to contributing. This page is about more technical details of how
OnePower is developed, and its philosophy.


Branching and Releasing
-----------------------
The aim is to make OnePowers's releases as useful, comprehendible, and automatic
as possible. This section lays out explicitly how this works (mostly for the benefit of
the admin(s)).

Versioning
~~~~~~~~~~
The first thing to mention is that we will try to use strict `semantic versioning <https://semver.org>`_.
Thus the versions are ``MAJOR.MINOR.PATCH``, with ``MAJOR`` including
API-breaking changes, ``MINOR`` including new features, and ``PATCH`` fixing bugs or
documentation etc. If you depend on onepower, you can set your dependency as
``onepower >= X.Y < X+1`` and not worry that we'll break your code with an update.

To mechanically handle versioning within the package, we use two methods that we make
to work together automatically. The "true" version of the package is set with
`setuptools-scm <https://pypi.org/project/setuptools-scm/>`_. This stores the version
in the git tag. There are many benefits to this -- one is that the version is unique
for every single change in the code, with commits on top of a release changing the
version. This means that versions accessed via ``onepower.__version__`` are unique and track
the exact code in the package (useful for reproducing results). To get the current
version from command line, simply do ``python setup.py --version`` in the top-level
directory.

Branching
~~~~~~~~~
For branching, the plan is to use a very similar model to `git-flow <https://nvie.com/posts/a-successful-git-branching-model/>`_.
That is, we will have a ``dev`` branch which acts as the current truth against which to develop,
and ``main`` essentially as a deployment branch.
I.e., the ``dev`` branch is where all features are merged (and some
non-urgent bugfixes). ``main`` is always production-ready, and corresponds
to a particular version on PyPI. Features should be branched from ``dev``,
and merged back to ``dev``. Hotfixes can be branched directly from ``main``,
and merged back there directly, *as well as* back into ``dev``.
*Breaking changes* must only be merged to ``dev`` when it has been decided that the next
version will be a major version. We do not do any long-term support of releases
(so can't make hotfixes to ``v2.x`` when the latest version is ``2.(x+1)``, or make a
new minor version in 2.x when the latest version is 3.x). We have set the default
branch to ``dev`` so that by default, branches are merged there. This is deemed best
for other developers (not maintainers/admins) to get involved, so the default thing is
usually right.

.. note:: Why not a more simple workflow like Github flow? The simple answer is it just
          doesn't really make sense for a library with semantic versioning. You get into
          trouble straight away if you want to merge a feature but don't want to update
          the version number yet (you want to merge multiple features into a nice release).
          In practice, this happens quite a lot.

.. note:: OK then, why not just use ``main`` to accrue features and fixes until such
          time we're ready to release? The problem here is that if you've merged a few
          features into main, but then realize a patch fix is required, there's no
          easy way to release that patch without releasing all the merged features, thus
          updating the minor version of the code (which may not be desirable). You could
          then just keep all features in their own branches until you're ready to release,
          but this is super annoying, and doesn't give you the chance to see how they
          interact.
