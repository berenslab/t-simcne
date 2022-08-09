#!/usr/bin/env python
# -*- mode: python-mode -*-

import sys
import inspect
import time

from pathlib import Path

from cnexp import redo, dispatch

if __name__ == "__main__":
    # redo will pass the target name as the second arg.  The directory
    # part is the relevant one for instantiating the object, so we
    # retrieve that via the parent attribute.
    name = Path(sys.argv[2]).parent
    algo = dispatch.from_string(name)

    filedeps = set(
        [
            mod.__file__
            for mod in [inspect.getmodule(m) for m in algo.__class__.mro()]
            if hasattr(mod, "__file__")
        ]
    )

    datadeps = algo.get_datadeps()
    redo.redo_ifchange(list(filedeps) + datadeps)

    t0 = time.time()
    algo()
    t1 = time.time()

    with open(sys.argv[3], "w") as f:
        f.write(f"{t1 - t0:.5f}\n")
