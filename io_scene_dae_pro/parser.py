# SPDX-License-Identifier: GPL-2.0-or-later
"""
Pure-Python COLLADA (.dae) parser.

Produces a lightweight intermediate representation (IR) that the Blender
side of the importer consumes.  This module deliberately has *no* bpy
imports so it can be unit-tested outside Blender.

Handles COLLADA 1.4.x and 1.5.x, namespaced and namespace-free documents,
and is intentionally forgiving towards sloppy exporters (mixed sources).
"""

import math
import xml.etree.ElementTree as ET
from urllib.parse import unquote

MAX_NODE_DEPTH = 128
MAX_WARNINGS = 100


class ColladaParseError(Exception):
    """Raised when the file is not usable COLLADA at all."""


# --------------------------------------------------------------------------
# Small XML helpers (namespace agnostic)
# --------------------------------------------------------------------------

def _ln(tag):
    """Local name of an XML tag, ignoring any namespace."""
    if not isinstance(tag, str):
        return ""
    return tag.rpartition("}")[2]


def _children(el, name=None):
    if el is None:
        return []
    if name is None:
        return list(el)
    return [c for c in el if _ln(c.tag) == name]


def _child(el, name):
    if el is None:
        return None
    for c in el:
        if _ln(c.tag) == name:
            return c
    return None


def _iter_local(el, name):
    """Recursive iteration over descendants with a given local name."""
    if el is None:
        return []
    return [c for c in el.iter() if c is not el and _ln(c.tag) == name]


def _strip_url(url):
    """'#foo' -> 'foo'."""
    if not url:
        return ""
    return url.lstrip("#")


def _floats(text):
    if not text:
        return []
    text = text.replace(",", " ")
    out = []
    for v in text.split():
        try:
            out.append(float(v))
        except ValueError:
            pass
    return out


def _ints(text):
    if not text:
        return []
    out = []
    for v in text.replace(",", " ").split():
        try:
            out.append(int(float(v)))
        except ValueError:
            pass
    return out


def _pad_color(vals):
    vals = list(vals)
    while len(vals) < 3:
        vals.append(0.0)
    if len(vals) < 4:
        vals.append(1.0)
    return tuple(vals[:4])


def _clean_path(p):
    """Normalize a texture URI coming from a Collada file."""
    if not p:
        return ""
    p = p.strip()
    if p.lower().startswith("file://"):
        p = p[7:]
        if p.lower().startswith("localhost/"):
            p = p[len("localhost/"):]
    return unquote(p).replace("\\", "/")


# --------------------------------------------------------------------------
# Tiny pure-python mat4 (column-major, COLLADA convention)
# --------------------------------------------------------------------------

def mat_identity():
    return [1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0]


def mat_mul(a, b):
    """Return a @ b for column-major flat 4x4 arrays."""
    out = [0.0] * 16
    for col in range(4):
        bcol0, bcol1, bcol2, bcol3 = b[col * 4: col * 4 + 4]
        for row in range(4):
            out[col * 4 + row] = (a[row] * bcol0 + a[4 + row] * bcol1 +
                                  a[8 + row] * bcol2 + a[12 + row] * bcol3)
    return out


def mat_translate(x, y, z):
    return [1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            x, y, z, 1.0]


def mat_scale(x, y, z):
    return [x, 0.0, 0.0, 0.0,
            0.0, y, 0.0, 0.0,
            0.0, 0.0, z, 0.0,
            0.0, 0.0, 0.0, 1.0]


def mat_rotate(axis, angle_deg):
    """Axis-angle rotation matrix (column-major)."""
    x, y, z = axis
    l = math.sqrt(x * x + y * y + z * z)
    if l < 1e-12:
        return mat_identity()
    x, y, z = x / l, y / l, z / l
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    t = 1.0 - c
    return [t * x * x + c,     t * x * y + s * z, t * x * z - s * y, 0.0,
            t * x * y - s * z, t * y * y + c,     t * y * z + s * x, 0.0,
            t * x * z + s * y, t * y * z - s * x, t * z * z + c,     0.0,
            0.0, 0.0, 0.0, 1.0]


def compose_transforms(elems):
    """Compose a node's transform elements in document order.

    COLLADA semantics: transforms listed first are outermost, i.e. the
    resulting matrix is M1 @ M2 @ ... @ Mn (points are transformed by Mn
    first).  This matches e.g. Blender's own Collada exporter (T * R * S).
    """
    m = mat_identity()
    for kind, vals in elems:
        if kind == "matrix" and len(vals) >= 16:
            # COLLADA <matrix> text is row-major; convert to column-major
            v = vals[:16]
            cm = [v[0], v[4], v[8], v[12],
                  v[1], v[5], v[9], v[13],
                  v[2], v[6], v[10], v[14],
                  v[3], v[7], v[11], v[15]]
            m = mat_mul(m, cm)
        elif kind == "translate" and len(vals) >= 3:
            m = mat_mul(m, mat_translate(vals[0], vals[1], vals[2]))
        elif kind == "rotate" and len(vals) >= 4:
            m = mat_mul(m, mat_rotate(vals[0:3], vals[3]))
        elif kind == "scale" and len(vals) >= 3:
            m = mat_mul(m, mat_scale(vals[0], vals[1], vals[2]))
    return m


# --------------------------------------------------------------------------
# IR containers
# --------------------------------------------------------------------------

class Source:
    __slots__ = ("id", "stride", "data")

    def __init__(self, sid, stride, data):
        self.id = sid
        self.stride = max(1, int(stride))
        self.data = data

    def count(self):
        return len(self.data) // self.stride if self.stride > 0 else 0


class Input:
    __slots__ = ("semantic", "source", "offset", "set")

    def __init__(self, semantic, source, offset, iset):
        self.semantic = semantic
        self.source = source      # raw source id (may be a <vertices> id)
        self.offset = offset
        self.set = iset


class Primitive:
    """One primitive element.

    `faces` holds (group_index, [corner_start, ...]) tuples.  A corner's
    value for an input with offset `o` is p_groups[gi][corner_start + o].
    """
    __slots__ = ("kind", "material", "inputs", "stride", "faces",
                 "p_groups", "vcount")

    def __init__(self, kind, material, inputs, stride, faces, p_groups, vcount):
        self.kind = kind
        self.material = material
        self.inputs = inputs
        self.stride = stride
        self.faces = faces
        self.p_groups = p_groups
        self.vcount = vcount

    def raw_index(self, gi, corner_start, offset):
        group = self.p_groups[gi]
        idx = corner_start + offset
        return group[idx] if idx < len(group) else -1

    def input_for(self, semantic, iset=None):
        """Find an input; `iset` filters TEXCOORD by set number."""
        best = None
        for inp in self.inputs:
            if inp.semantic != semantic:
                continue
            if semantic == "TEXCOORD" and iset is not None and inp.set != iset:
                continue
            best = inp
            if semantic != "TEXCOORD" or iset is not None:
                break
        return best

    def texcoord_sets(self):
        return sorted({inp.set for inp in self.inputs if inp.semantic == "TEXCOORD"})


class Geometry:
    __slots__ = ("id", "name", "sources", "verts_map", "primitives")

    def __init__(self, gid, name):
        self.id = gid
        self.name = name
        self.sources = {}      # source id -> Source
        self.verts_map = {}    # <vertices> id -> {semantic: source id}
        self.primitives = []

    def resolve_source(self, input_src):
        """Resolve an input's source id (through <vertices> indirection)."""
        if input_src in self.sources:
            return self.sources[input_src]
        vmap = self.verts_map.get(input_src)
        if vmap:
            pos = vmap.get("POSITION")
            if pos and pos in self.sources:
                return self.sources[pos]
            for src_id in vmap.values():
                if src_id in self.sources:
                    return self.sources[src_id]
        return None


class TexRef:
    __slots__ = ("sampler", "texcoord", "image_id")

    def __init__(self, sampler, texcoord):
        self.sampler = sampler      # sampler/image id or path as written
        self.texcoord = texcoord    # texcoord semantic string
        self.image_id = None        # resolved library_images id (or None)


class ColorTex:
    """A Collada color-slot value: flat color, texture, or param reference."""
    __slots__ = ("kind", "color", "tex")

    def __init__(self, kind, color=None, tex=None):
        self.kind = kind            # 'color' | 'texture'
        self.color = color          # (r, g, b, a)
        self.tex = tex              # TexRef


class Effect:
    __slots__ = ("id", "shader", "emission", "diffuse", "specular", "shininess",
                 "reflective", "reflectivity", "transparent", "transparency",
                 "opaque_mode", "double_sided", "bump", "bump_is_normal")

    def __init__(self, eid):
        self.id = eid
        self.shader = "phong"
        self.emission = ColorTex("color", (0.0, 0.0, 0.0, 1.0))
        self.diffuse = ColorTex("color", (0.8, 0.8, 0.8, 1.0))
        self.specular = ColorTex("color", (0.5, 0.5, 0.5, 1.0))
        self.shininess = 30.0
        self.reflective = None
        self.reflectivity = 0.0
        self.transparent = None
        self.transparency = 1.0
        self.opaque_mode = "A_ONE"
        self.double_sided = False
        self.bump = None
        self.bump_is_normal = False


class Material:
    __slots__ = ("id", "name", "effect_id")

    def __init__(self, mid, name, effect_id):
        self.id = mid
        self.name = name
        self.effect_id = effect_id


class ImageEntry:
    __slots__ = ("id", "path")

    def __init__(self, iid, path):
        self.id = iid
        self.path = path


class InstanceMaterial:
    __slots__ = ("symbol", "target", "uvbinds")

    def __init__(self, symbol, target, uvbinds):
        self.symbol = symbol
        self.target = target      # material id
        self.uvbinds = uvbinds    # {texcoord semantic: uv set index}


class Node:
    __slots__ = ("id", "sid", "name", "node_type", "matrix",
                 "children", "geom_instances", "ctrl_instances", "node_refs",
                 "has_camera", "has_light")

    def __init__(self):
        self.id = ""
        self.sid = ""
        self.name = "Node"
        self.node_type = "NODE"
        self.matrix = mat_identity()
        self.children = []
        self.geom_instances = []    # [(geometry id, [InstanceMaterial])]
        self.ctrl_instances = []    # [(controller id, [InstanceMaterial])]
        self.node_refs = []         # instance_node urls
        self.has_camera = False
        self.has_light = False


class Document:
    def __init__(self):
        self.images = {}        # id -> ImageEntry
        self.effects = {}       # id -> Effect
        self.materials = {}     # id -> Material
        self.geometries = {}    # id -> Geometry
        self.controllers = {}   # id -> geometry id (bind-pose fallback)
        self.nodes = {}         # id -> Node
        self.roots = []
        self.up_axis = "Y_UP"
        self.unit_scale = 1.0
        self.filepath = ""
        self.warnings = []
        self._warn_truncated = False

    def warn(self, msg):
        if len(self.warnings) < MAX_WARNINGS:
            self.warnings.append(msg)
        elif not self._warn_truncated:
            self.warnings.append("... further warnings suppressed")
            self._warn_truncated = True


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def parse_collada(filepath):
    doc = Document()
    doc.filepath = filepath
    try:
        tree = ET.parse(filepath)
    except ET.ParseError as ex:
        raise ColladaParseError("XML parse error: %s" % ex)
    except OSError as ex:
        raise ColladaParseError("Cannot read file: %s" % ex)

    root = tree.getroot()
    if _ln(root.tag) != "COLLADA":
        raise ColladaParseError("Root element is not <COLLADA>")

    _parse_asset(_child(root, "asset"), doc)
    _parse_images(_child(root, "library_images"), doc)
    _parse_effects(_child(root, "library_effects"), doc)
    _parse_materials(_child(root, "library_materials"), doc)
    _parse_geometries(_child(root, "library_geometries"), doc)
    _parse_controllers(_child(root, "library_controllers"), doc)
    _parse_visual_scene(root, doc)
    return doc


# ---------------- asset ----------------

def _parse_asset(el, doc):
    if el is None:
        doc.warn("No <asset> element: assuming Y-up, unit scale 1.0")
        return
    unit = _child(el, "unit")
    if unit is not None:
        try:
            doc.unit_scale = float(unit.get("meter", "1.0"))
        except ValueError:
            pass
    up = _child(el, "up_axis")
    if up is not None and (up.text or "").strip():
        doc.up_axis = up.text.strip().upper()


# ---------------- images ----------------

def _image_path_from_init_from(init_el):
    if init_el is None:
        return ""
    text = (init_el.text or "").strip()
    if not text:
        ref = _child(init_el, "ref")           # Collada 1.5 style
        if ref is not None:
            text = (ref.text or "").strip()
    if not text:
        for d in init_el.iter():               # last resort: any descendant text
            if d is not init_el and (d.text or "").strip():
                text = d.text.strip()
                break
    return text


def _parse_images(el, doc):
    for img in _children(el, "image"):
        iid = img.get("id") or ""
        path = _image_path_from_init_from(_child(img, "init_from"))
        if not path:
            texts = [t.strip() for t in img.itertext() if t and t.strip()]
            path = texts[0] if texts else ""
        if iid:
            doc.images[iid] = ImageEntry(iid, _clean_path(path))


# ---------------- effects ----------------

def _parse_color_tex(el, params):
    if el is None:
        return None
    tex = _child(el, "texture")
    if tex is not None:
        return ColorTex("texture", tex=TexRef(_strip_url(tex.get("texture", "")),
                                              tex.get("texcoord", "")))
    col = _child(el, "color")
    if col is not None:
        return ColorTex("color", _pad_color(_floats(col.text)))
    par = _child(el, "param")
    if par is not None:
        ref = par.get("ref", "")
        kind, val = params.get(ref, (None, None))
        if kind == "color":
            return ColorTex("color", _pad_color(val))
        if kind == "float":
            return ColorTex("color", (val, val, val, 1.0))
    return None


def _parse_newparams(container, params):
    for np_el in _children(container, "newparam"):
        sid = np_el.get("sid", "")
        if not sid or sid in params:
            continue
        surface = _child(np_el, "surface")
        if surface is not None:
            init = _child(surface, "init_from")
            txt = (init.text or "").strip() if init is not None else ""
            if not txt and init is not None:
                ref = _child(init, "ref")
                txt = (ref.text or "").strip() if ref is not None else ""
            params[sid] = ("surface", txt)
            continue
        sampler = _child(np_el, "sampler2D")
        if sampler is not None:
            src = _child(sampler, "source")
            params[sid] = ("sampler", (src.text or "").strip() if src is not None else "")
            continue
        color = _child(np_el, "color")
        if color is not None:
            params[sid] = ("color", _pad_color(_floats(color.text)))
            continue
        for fk in ("float", "float2", "float3", "float4"):
            fel = _child(np_el, fk)
            if fel is not None:
                vals = _floats(fel.text)
                params[sid] = ("float", vals[0] if vals else 0.0)
                break


def _resolve_texture_sampler(texref, params, doc):
    """Resolve TexRef.sampler through sampler->surface->image chains.

    Returns a library_images id, or '' when unresolvable (the raw value is
    then likely a path itself, which the texture resolver will try)."""
    sid = texref.sampler
    seen = set()
    for _ in range(6):  # bounded chain walk
        if sid in seen:
            break
        seen.add(sid)
        kind, val = params.get(sid, (None, None))
        if kind == "sampler":
            sid = val
        elif kind == "surface":
            if val:
                return val
            break
        else:
            break
    if sid in doc.images:
        return sid
    return ""


def _param_float(el, params, default):
    if el is None:
        return default
    txt = (el.text or "").strip()
    par = _child(el, "param")
    if par is not None:
        ref = par.get("ref", "")
        kind, val = params.get(ref, (None, None))
        if kind == "float":
            return val
    vals = _floats(txt)
    return vals[0] if vals else default


def _pick_profile(eff_el):
    profiles = [c for c in eff_el if _ln(c.tag).startswith("profile_")]
    for p in profiles:
        if _ln(p.tag) == "profile_COMMON":
            return p
    return profiles[0] if profiles else None


def _parse_effect(eff_el, doc):
    eff = Effect(eff_el.get("id") or "")
    params = {}
    chosen = _pick_profile(eff_el)

    if chosen is None:
        # Exotic profile: salvage a diffuse value anywhere in the effect
        for tex in _iter_local(eff_el, "texture"):
            tr = TexRef(_strip_url(tex.get("texture", "")), tex.get("texcoord", ""))
            tr.image_id = tr.sampler
            eff.diffuse = ColorTex("texture", tex=tr)
            eff.shader = "other"
            break
        else:
            for col in _iter_local(eff_el, "diffuse"):
                eff.diffuse = _parse_color_tex(col, {}) or eff.diffuse
                break
        return eff

    _parse_newparams(chosen, params)
    _parse_newparams(eff_el, params)   # effect-level newparams (rare)

    shader_el = None
    for t in _iter_local(chosen, "technique"):
        for name in ("phong", "blinn", "lambert", "constant"):
            s = _child(t, name)
            if s is not None:
                shader_el = s
                break
        if shader_el is not None:
            break
    if shader_el is None:
        techs = _iter_local(chosen, "technique")
        shader_el = techs[0] if techs else chosen
        eff.shader = "other"
    else:
        eff.shader = _ln(shader_el.tag)

    eff.emission = _parse_color_tex(_child(shader_el, "emission"), params) or eff.emission
    eff.diffuse = _parse_color_tex(_child(shader_el, "diffuse"), params) or eff.diffuse
    eff.specular = _parse_color_tex(_child(shader_el, "specular"), params) or eff.specular
    eff.reflective = _parse_color_tex(_child(shader_el, "reflective"), params)
    eff.shininess = _param_float(_child(shader_el, "shininess"), params, eff.shininess)
    eff.reflectivity = _param_float(_child(shader_el, "reflectivity"), params, eff.reflectivity)
    eff.reflectivity = max(0.0, min(1.0, eff.reflectivity))

    trans_el = _child(shader_el, "transparent")
    if trans_el is not None:
        eff.transparent = _parse_color_tex(trans_el, params) or ColorTex("color", (1.0, 1.0, 1.0, 1.0))
        eff.opaque_mode = (trans_el.get("opaque") or "A_ONE").upper()
    eff.transparency = _param_float(_child(shader_el, "transparency"), params, eff.transparency)

    # double_sided from vendor extras (GOOGLEEARTH etc.)
    for ds in _iter_local(eff_el, "double_sided"):
        vals = _floats((ds.text or "").strip())
        if vals and vals[0] > 0.5:
            eff.double_sided = True
        break

    # bump / normal map from vendor extras (3DSMAX, FCOLLADA, MAYA, ...)
    for bump in _iter_local(eff_el, "bump"):
        tex = _child(bump, "texture")
        if tex is not None:
            eff.bump = TexRef(_strip_url(tex.get("texture", "")), tex.get("texcoord", ""))
            img_id = _resolve_texture_sampler(eff.bump, params, doc)
            eff.bump.image_id = img_id
            img = doc.images.get(img_id) if img_id else None
            name = (img.path if img else eff.bump.sampler) or ""
            low = name.lower()
            eff.bump_is_normal = any(k in low for k in
                                     ("normal", "_ddn", "_nrm", "norm-map", "normalmap"))
            break

    # Resolve main texture slots to concrete image ids where possible
    for ct in (eff.diffuse, eff.emission, eff.specular, eff.transparent,
               eff.reflective):
        if ct is not None and ct.kind == "texture" and ct.tex.image_id is None:
            ct.tex.image_id = _resolve_texture_sampler(ct.tex, params, doc) or ct.tex.sampler
    if eff.bump is not None and eff.bump.image_id is None:
        eff.bump.image_id = _resolve_texture_sampler(eff.bump, params, doc) or eff.bump.sampler
    return eff


def _parse_effects(el, doc):
    for eff_el in _children(el, "effect"):
        eid = eff_el.get("id") or ""
        if not eid or eid in doc.effects:
            continue
        doc.effects[eid] = _parse_effect(eff_el, doc)


# ---------------- materials ----------------

def _parse_materials(el, doc):
    for m in _children(el, "material"):
        mid = m.get("id") or ""
        if not mid or mid in doc.materials:
            continue
        name = m.get("name") or mid
        eff_id = ""
        inst = _child(m, "instance_effect")
        if inst is not None:
            eff_id = _strip_url(inst.get("url", ""))
        doc.materials[mid] = Material(mid, name, eff_id)


# ---------------- geometries ----------------

def _parse_source(src_el):
    sid = src_el.get("id") or ""
    arr = _child(src_el, "float_array")
    if arr is None:
        arr = _child(src_el, "int_array")
    if arr is None:
        return None
    data = _floats(arr.text)
    stride = 0
    tech = _child(src_el, "technique_common")
    acc = _child(tech, "accessor") if tech is not None else None
    if acc is not None:
        try:
            stride = int(acc.get("stride", "0"))
        except ValueError:
            stride = 0
        if stride <= 0:
            nparams = len(_children(acc, "param"))
            count = 0
            try:
                count = int(acc.get("count", "0"))
            except ValueError:
                pass
            stride = nparams if nparams else (len(data) // count if count else 3)
    if stride <= 0:
        stride = 3
    return Source(sid, stride, data)


def _parse_primitive(el, kind, doc):
    material = el.get("material") or ""
    inputs = []
    for inp in _children(el, "input"):
        semantic = inp.get("semantic") or ""
        if not semantic:
            continue
        try:
            off = int(inp.get("offset", "0"))
        except ValueError:
            off = 0
        iset = -1
        if inp.get("set") is not None:
            try:
                iset = int(inp.get("set"))
            except ValueError:
                iset = -1
        inputs.append(Input(semantic, _strip_url(inp.get("source", "")), off, iset))
    if not inputs:
        return None
    stride = max(inp.offset for inp in inputs) + 1

    p_groups = []
    vcount = []
    if kind == "polygons":
        vc_el = _child(el, "vcount")
        if vc_el is not None:
            vcount = _ints(vc_el.text)
        for p_el in _children(el, "p"):
            p_groups.append(_ints(p_el.text))
        if not p_groups:
            return None
        if vcount:
            total = sum(len(g) for g in p_groups)
            if sum(vcount) * stride == total:
                # vcount describes corners across all <p> groups: split them
                flat = [i for g in p_groups for i in g]
                faces_raw = []
                pos = 0
                for n in vcount:
                    faces_raw.append(flat[pos: pos + n * stride])
                    pos += n * stride
                p_groups = faces_raw
    else:
        p_el = _child(el, "p")
        if p_el is None:
            return None
        p_groups.append(_ints(p_el.text))
        if kind in ("polylist", "tristrips", "trifans"):
            vc_el = _child(el, "vcount")
            if vc_el is None:
                return None
            vcount = _ints(vc_el.text)

    faces = []
    if kind == "triangles":
        for gi, p in enumerate(p_groups):
            ncorners = len(p) // stride
            for i in range(0, ncorners - 2, 3):
                faces.append((gi, [(i + k) * stride for k in range(3)]))
    elif kind == "polylist":
        p = p_groups[0]
        pos = 0
        for n in vcount:
            if n < 3 or pos + n * stride > len(p):
                break
            faces.append((0, [pos + k * stride for k in range(n)]))
            pos += n * stride
    elif kind == "polygons":
        for gi, p in enumerate(p_groups):
            ncorners = len(p) // stride
            if ncorners >= 3:
                faces.append((gi, [k * stride for k in range(ncorners)]))
    elif kind == "tristrips":
        for gi, p in enumerate(p_groups):
            base = 0
            counts = vcount if vcount else [len(p) // stride]
            for n in counts:
                for j in range(n - 2):
                    a, b, c = base + j, base + j + 1, base + j + 2
                    if j % 2 == 1:
                        a, b = b, a
                    if max(a, b, c) * stride + stride - 1 < len(p):
                        faces.append((gi, [a * stride, b * stride, c * stride]))
                base += n
    elif kind == "trifans":
        for gi, p in enumerate(p_groups):
            base = 0
            counts = vcount if vcount else [len(p) // stride]
            for n in counts:
                for j in range(1, n - 1):
                    faces.append((gi, [base * stride, (base + j) * stride,
                                       (base + j + 1) * stride]))
                base += n
    else:
        return None

    if not faces:
        return None
    return Primitive(kind, material, inputs, stride, faces, p_groups, vcount)


def _parse_geometries(el, doc):
    for geo_el in _children(el, "geometry"):
        gid = geo_el.get("id") or ""
        if not gid or gid in doc.geometries:
            continue
        geo = Geometry(gid, geo_el.get("name") or gid)
        mesh = _child(geo_el, "mesh")
        if mesh is None:
            if _child(geo_el, "spline") is not None:
                doc.warn("Geometry '%s' is a spline: not imported" % gid)
            continue

        for src_el in _children(mesh, "source"):
            src = _parse_source(src_el)
            if src is not None and src.id:
                geo.sources[src.id] = src

        for v_el in _children(mesh, "vertices"):
            vid = v_el.get("id") or ""
            if not vid:
                continue
            m = {}
            for inp in _children(v_el, "input"):
                sem = inp.get("semantic") or ""
                if sem:
                    m[sem] = _strip_url(inp.get("source", ""))
            geo.verts_map[vid] = m

        supported = ("triangles", "polylist", "polygons", "tristrips", "trifans")
        for p_el in mesh:
            ln = _ln(p_el.tag)
            if ln in ("lines", "linestrips"):
                doc.warn("Geometry '%s' contains line primitives: skipped" % gid)
            elif ln in supported:
                prim = _parse_primitive(p_el, ln, doc)
                if prim is not None:
                    geo.primitives.append(prim)

        if not geo.primitives:
            doc.warn("Geometry '%s' has no supported primitives" % gid)
        doc.geometries[gid] = geo


# ---------------- controllers (bind-pose fallback) ----------------

def _parse_controllers(el, doc):
    if el is None:
        return
    for ctrl in _children(el, "controller"):
        cid = ctrl.get("id") or ""
        if not cid:
            continue
        skin = _child(ctrl, "skin")
        if skin is not None:
            doc.controllers[cid] = _strip_url(skin.get("source", ""))
        else:
            morph = _child(ctrl, "morph")
            if morph is not None:
                doc.controllers[cid] = _strip_url(morph.get("source", ""))
                doc.warn("Morph controller '%s' imported as its base mesh" % cid)


# ---------------- nodes / scene ----------------

def _parse_bind_material(inst_el):
    binds = []
    bm = _child(inst_el, "bind_material")
    if bm is None:
        return binds
    tech = _child(bm, "technique_common")
    for im in _children(tech, "instance_material"):
        symbol = im.get("symbol") or ""
        target = _strip_url(im.get("target", ""))
        uvbinds = {}
        for bvi in _children(im, "bind_vertex_input"):
            semantic = bvi.get("semantic") or ""
            inp_sem = (bvi.get("input_semantic") or "").upper()
            if semantic and inp_sem == "TEXCOORD":
                try:
                    uvbinds[semantic] = int(bvi.get("input_set", "0"))
                except ValueError:
                    uvbinds[semantic] = 0
        binds.append(InstanceMaterial(symbol, target, uvbinds))
    return binds


def _parse_node(el, doc, depth):
    if depth > MAX_NODE_DEPTH:
        doc.warn("Node hierarchy deeper than %d levels: truncated" % MAX_NODE_DEPTH)
        return None
    node = Node()
    node.id = el.get("id") or ""
    node.sid = el.get("sid") or ""
    node.name = el.get("name") or node.id or "Node"
    node.node_type = el.get("type") or "NODE"

    elems = []
    for c in el:
        ln = _ln(c.tag)
        if ln == "matrix":
            v = _floats(c.text)
            if len(v) >= 16:
                elems.append(("matrix", v[:16]))
        elif ln == "translate":
            v = _floats(c.text)
            if len(v) >= 3:
                elems.append(("translate", v))
        elif ln == "rotate":
            v = _floats(c.text)
            if len(v) >= 4:
                elems.append(("rotate", v))
        elif ln == "scale":
            v = _floats(c.text)
            if len(v) >= 3:
                elems.append(("scale", v))
        elif ln in ("lookat", "skew"):
            doc.warn("Unsupported transform <%s> on node '%s' ignored" % (ln, node.name))
        elif ln == "instance_geometry":
            url = _strip_url(c.get("url", ""))
            if url:
                node.geom_instances.append((url, _parse_bind_material(c)))
        elif ln == "instance_controller":
            url = _strip_url(c.get("url", ""))
            if url:
                node.ctrl_instances.append((url, _parse_bind_material(c)))
        elif ln == "instance_node":
            url = _strip_url(c.get("url", ""))
            if url:
                node.node_refs.append(url)
        elif ln == "instance_camera":
            node.has_camera = True
        elif ln == "instance_light":
            node.has_light = True
        elif ln == "node":
            sub = _parse_node(c, doc, depth + 1)
            if sub is not None:
                node.children.append(sub)

    node.matrix = compose_transforms(elems)
    if node.id:
        doc.nodes[node.id] = node
    return node


def _parse_visual_scene(root, doc):
    lib = _child(root, "library_visual_scenes")
    scene_el = _child(root, "scene")
    vs = None
    if scene_el is not None:
        ivs = _child(scene_el, "instance_visual_scene")
        if ivs is not None:
            url = _strip_url(ivs.get("url", ""))
            for c in _children(lib, "visual_scene"):
                if c.get("id") == url:
                    vs = c
                    break
    if vs is None:
        vs = _child(lib, "visual_scene")
    if vs is None:
        doc.warn("No visual scene found: nothing to import")
        return
    for c in vs:
        if _ln(c.tag) == "node":
            node = _parse_node(c, doc, 0)
            if node is not None:
                doc.roots.append(node)