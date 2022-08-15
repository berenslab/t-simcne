#!/usr/bin/env python
# -*- mode: python-mode -*-

import inspect
import sys
import time
from pathlib import Path

from cnexp import dispatch, redo

if __name__ == "__main__":
    # redo will pass the target name as the second arg.  The directory
    # part is the relevant one for instantiating the object, so we
    # retrieve that via the parent attribute.
    name = Path(sys.argv[2]).parent
    algo = dispatch.from_string(name)

    filedeps = [
        mod.__file__
        for mod in [inspect.getmodule(m) for m in algo.__class__.mro()]
        if hasattr(mod, "__file__")
    ]

    deps = algo.get_deps()
    redo.redo_ifchange(deps + list(set(filedeps)))

    t0 = time.time()
    algo()
    t1 = time.time()

    with open(sys.argv[3], "w") as f:
        f.write(f"{t1 - t0:.3f}\n")
