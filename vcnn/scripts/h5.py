"""Thin helpers around PyTables for reading/writing the dataset `.h5` files.

Cleaned up from the original:
  * `dt is 'obj'` -> `dt == 'obj'` (identity comparison on str literals is a bug).
  * Dropped the legacy `tables.openFile` fallback (removed in modern PyTables).
"""

import os

import numpy
import tables


def h5f_(fname, mode="r", title="", force=False):
    """Open (and optionally create) an HDF5 file, guarding against accidental 'w'."""
    assert mode != "w" or force, "should set force=True to open in write mode"

    _path = os.path.dirname(fname)
    if _path and not os.path.exists(_path):
        os.makedirs(_path)
    if mode == "r+" and not os.path.exists(fname):
        mode = "w"

    return tables.open_file(fname, mode=mode, title=title)


def _dict2ttype(m):
    return {k: tables.Col.from_dtype(numpy.dtype(v)) for k, v in m.items()}


def atom_(dt):
    if dt == "obj":
        return tables.ObjectAtom()
    if dt == "bool":
        return tables.BoolAtom()
    return tables.Atom.from_dtype(numpy.dtype(dt))


_FilterConf = {
    "zip": tables.Filters(complevel=5, complib="zlib"),
    "blosc": tables.Filters(complevel=3, complib="blosc"),
    "_default": tables.Filters(complevel=5, complib="zlib"),
}

_ByteAtom = atom_("b")
_Int32Atom = atom_("i4")
_BoolAtom = atom_("bool")


def _resolve_atom(dtype):
    if isinstance(dtype, str):
        return atom_(dtype)
    if isinstance(dtype, tables.atom.Atom):
        return dtype
    raise ValueError("dtype(%s) should be str or tables.atom.Atom!" % dtype)


def _resolve_filters(args):
    _filter = args.get("filters", "_default")
    if isinstance(_filter, str):
        args["filters"] = _FilterConf[_filter]
    return args


def create_carray(h5f, where, name, dtype, shape, **args):
    """Fixed-shape compressed array (all columns same type)."""
    args = _resolve_filters(args)
    atom = _resolve_atom(dtype)
    return h5f.create_carray(where, name, atom, shape, **args)


def create_earray(h5f, where, name, dtype, **args):
    """Enlargeable array (rows can grow)."""
    args = _resolve_filters(args)
    atom = _resolve_atom(dtype)
    return h5f.create_earray(where, name, atom, **args)


def create_vlarray(h5f, where, name, dtype, **args):
    """Variable-length array (both axes can grow)."""
    args = _resolve_filters(args)
    atom = _resolve_atom(dtype)
    return h5f.create_vlarray(where, name, atom, **args)


def create_table(h5f, where, name, dtype, **args):
    args = _resolve_filters(args)
    if isinstance(dtype, dict):
        dtype = _dict2ttype(dtype)  # NOTE: dict reorders fields; use a list if order matters
    elif not isinstance(dtype, (tables.atom.Atom, numpy.dtype)):
        raise ValueError("dtype(%s) should be dict or tables.atom.Atom!" % dtype)
    return h5f.create_table(where, name, dtype, **args)


def rm(h5f, where, r=False):
    """Remove node at `where` (recursively if r=True)."""
    if where not in h5f:
        return
    pth, fn = os.path.dirname(where), os.path.basename(where)
    h5f.remove_node(pth, fn, r)


def mv(h5f, src, dest, replace=False, fields=None):
    """Move all (or selected) child nodes from `src` group to `dest` group."""
    assert src != dest, "src should not be same as dest!"
    if dest not in h5f:
        mkdir(h5f, dest)
    fields = fields or [e._v_name for e in h5f.list_nodes(src)]
    for k in fields:
        h5f.move_node(src, newparent=dest, name=k, overwrite=replace)
    if not h5f.list_nodes(src):
        rm(h5f, src)


def mkdir(h5f, path, mode="r+"):
    """Recursively create groups up to `path`."""
    if path == "/" or (mode != "w" and (path in h5f)):
        return
    if mode == "w":
        rm(h5f, path, r=True)
    pth, fn = os.path.dirname(path), os.path.basename(path)
    mkdir(h5f, pth)
    return h5f.create_group(pth, fn)
