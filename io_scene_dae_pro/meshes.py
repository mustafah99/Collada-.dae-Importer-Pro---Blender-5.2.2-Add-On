# SPDX-License-Identifier: GPL-2.0-or-later
"""Build Blender mesh datablocks from parsed Collada geometry (IR)."""

import numpy as np

import bpy
from mathutils import Matrix

from .materials import uv_layer_name


def matrix_from_collada(cm):
    """Column-major flat 16-float COLLADA matrix -> mathutils 4x4 Matrix."""
    return Matrix(((cm[0], cm[4], cm[8], cm[12]),
                   (cm[1], cm[5], cm[9], cm[13]),
                   (cm[2], cm[6], cm[10], cm[14]),
                   (cm[3], cm[7], cm[11], cm[15])))


def _safe_name(n):
    if not n:
        return "Mesh"
    return n.replace("/", "_").replace("\\", "_").strip()[:63] or "Mesh"


class MeshService:
    """Creates (and caches) Blender meshes for geometry + material binds."""

    def __init__(self, doc, material_service, use_custom_normals=True):
        self.doc = doc
        self.ms = material_service
        self.use_custom_normals = use_custom_normals
        self._cache = {}             # (geo_id, binds_sig) -> bpy mesh
        self.warnings = []
        self.stats = {"meshes": 0}

    def warn(self, msg):
        if len(self.warnings) < 100:
            self.warnings.append(msg)

    def binds_signature(self, binds):
        return tuple(sorted(
            (b.symbol, b.target, tuple(sorted(b.uvbinds.items())))
            for b in binds))

    def get_mesh(self, geo_id, binds):
        geo = self.doc.geometries.get(geo_id)
        if geo is None:
            self.warn("Referenced geometry '%s' not found" % geo_id)
            return None
        key = (geo_id, self.binds_signature(binds))
        mesh = self._cache.get(key)
        if mesh is not None and mesh.name in bpy.data.meshes:
            return mesh
        mesh = self._build(geo, binds)
        if mesh is not None:
            self._cache[key] = mesh
        return mesh

    # ------------------------------------------------------------------

    def _build(self, geo, binds):
        prims = [p for p in geo.primitives if p.faces]
        if not prims:
            return None

        # ---- material slots --------------------------------------------
        slot_map = {}                 # symbol (or None) -> slot index
        slot_mats = []
        for prim in prims:
            sym = prim.material if prim.material else None
            if sym not in slot_map:
                slot_map[sym] = len(slot_mats)
                slot_mats.append(self.ms.get_for_binds(binds, prim.material))

        # ---- POSITION sources: assign a base offset per used source ------
        vert_slots = {}               # source id -> (base, count)
        vert_blocks = []              # ordered np arrays (n, 3)
        total_verts = 0
        prim_vert = []                # per prim: (vi, base, count)
        for prim in prims:
            vi = prim.input_for("VERTEX") or prim.input_for("POSITION")
            base = count = None
            if vi is not None:
                src = geo.resolve_source(vi.source)
                if src is not None and src.stride >= 3:
                    if src.id not in vert_slots:
                        n = src.count()
                        vert_slots[src.id] = (total_verts, n)
                        arr = np.asarray(src.data, dtype=np.float32)
                        usable = (len(arr) // src.stride) * src.stride
                        arr = arr[:usable].reshape(-1, src.stride)[:, 0:3]
                        vert_blocks.append(np.ascontiguousarray(arr, dtype=np.float32))
                        total_verts += n
                    base, count = vert_slots[src.id]
            prim_vert.append((vi, base, count))
            if vi is None or base is None:
                self.warn("Geometry '%s': primitive without usable POSITION "
                          "data skipped" % geo.name)

        if total_verts == 0:
            return None

        # ---- per-prim attribute sources ----------------------------------
        prim_attr = []                # (nrm_src, ci, col_src, uv_sources)
        all_uv_sets = set()
        any_normals = False
        any_colors = False
        for prim in prims:
            ni = prim.input_for("NORMAL")
            nrm_src = geo.resolve_source(ni.source) if ni is not None else None
            ci = prim.input_for("COLOR")
            col_src = geo.resolve_source(ci.source) if ci is not None else None
            uv_sources = {}
            for inp in prim.inputs:
                if inp.semantic != "TEXCOORD":
                    continue
                src = geo.resolve_source(inp.source)
                if src is None or src.stride < 2:
                    continue
                s = inp.set if inp.set >= 0 else 0
                if s not in uv_sources:
                    uv_sources[s] = src
                    all_uv_sets.add(s)
            prim_attr.append((nrm_src, ci, col_src, uv_sources))
            if nrm_src is not None:
                any_normals = True
            if col_src is not None:
                any_colors = True

        uv_sets = sorted(all_uv_sets)
        uv_data = {s: [] for s in uv_sets}          # per loop (u, v)
        loops_vert = []
        loops_nrm = []                # per loop (x, y, z) authored normal
        loops_col = []
        faces_start = []
        faces_size = []
        faces_mat = []

        # ---- main corner assembly loop ------------------------------------
        for pi, prim in enumerate(prims):
            vi, vbase, vcount = prim_vert[pi]
            if vi is None or vbase is None:
                continue
            nrm_src, ci, col_src, uv_sources = prim_attr[pi]
            n_data = nrm_src.data if nrm_src is not None else None
            n_stride = nrm_src.stride if nrm_src is not None else 0
            c_data = col_src.data if col_src is not None else None
            c_stride = col_src.stride if col_src is not None else 0
            uv_acc = {}
            for s in uv_sets:
                src = uv_sources.get(s)
                uv_acc[s] = (src.data, src.stride, prim.input_for("TEXCOORD", s)) \
                    if src is not None else (None, 0, None)

            mat_idx = slot_map.get(prim.material if prim.material else None, 0)

            for gi, corners in prim.faces:
                nc = len(corners)
                if nc < 3:
                    continue
                start = len(loops_vert)
                added = 0
                for c in corners:
                    v = prim.raw_index(gi, c, vi.offset)
                    if v < 0 or v >= vcount:
                        continue
                    loops_vert.append(v + vbase)
                    # authored normal for this corner
                    if n_data is not None:
                        n_idx = prim.raw_index(gi, c, ni.offset)
                        base_i = n_idx * n_stride
                        if n_idx >= 0 and base_i + 2 < len(n_data):
                            loops_nrm.append((n_data[base_i], n_data[base_i + 1],
                                              n_data[base_i + 2]))
                        else:
                            loops_nrm.append((0.0, 0.0, 0.0))
                    elif any_normals:
                        loops_nrm.append((0.0, 0.0, 0.0))
                    # vertex color for this corner
                    if c_data is not None:
                        c_idx = prim.raw_index(gi, c, ci.offset) if ci else -1
                        base_i = c_idx * c_stride
                        if c_idx >= 0 and base_i + 2 < len(c_data):
                            a = c_data[base_i + 3] \
                                if c_stride >= 4 and base_i + 3 < len(c_data) else 1.0
                            loops_col.append((c_data[base_i], c_data[base_i + 1],
                                              c_data[base_i + 2], a))
                        else:
                            loops_col.append((1.0, 1.0, 1.0, 1.0))
                    elif any_colors:
                        loops_col.append((1.0, 1.0, 1.0, 1.0))
                    # uvs for every layer
                    for s in uv_sets:
                        u_data, u_stride, ui = uv_acc[s]
                        if u_data is not None and ui is not None:
                            u_idx = prim.raw_index(gi, c, ui.offset)
                            base_i = u_idx * u_stride
                            if u_idx >= 0 and base_i + 1 < len(u_data):
                                uv_data[s].append((u_data[base_i],
                                                   u_data[base_i + 1]))
                            else:
                                uv_data[s].append((0.0, 0.0))
                        else:
                            uv_data[s].append((0.0, 0.0))
                    added += 1
                if added >= 3:
                    faces_start.append(start)
                    faces_size.append(added)
                    faces_mat.append(mat_idx)
                else:
                    del loops_vert[start:]
                    del loops_nrm[start:]
                    if loops_col:
                        del loops_col[start:]
                    for s in uv_sets:
                        del uv_data[s][start:]

        nloops = len(loops_vert)
        nfaces = len(faces_start)
        if nloops < 3 or nfaces == 0:
            self.warn("Geometry '%s': no valid faces" % geo.name)
            return None

        # ---- create the mesh ------------------------------------------------
        mesh = bpy.data.meshes.new(_safe_name(geo.name))
        verts_flat = np.concatenate(vert_blocks).ravel() \
            if len(vert_blocks) > 1 else np.asarray(vert_blocks[0]).ravel()

        ok_fast = True
        try:
            mesh.vertices.add(total_verts)
            mesh.vertices.foreach_set("co", verts_flat)
            mesh.loops.add(nloops)
            mesh.loops.foreach_set("vertex_index",
                                   np.asarray(loops_vert, dtype=np.int32))
            mesh.polygons.add(nfaces)
            mesh.polygons.foreach_set("loop_start",
                                      np.asarray(faces_start, dtype=np.int32))
            mesh.polygons.foreach_set("loop_total",
                                      np.asarray(faces_size, dtype=np.int32))
        except (RuntimeError, ValueError):
            ok_fast = False

        if not ok_fast:
            # Robust fallback via from_pydata
            try:
                bpy.data.meshes.remove(mesh)
            except RuntimeError:
                pass
            mesh = bpy.data.meshes.new(_safe_name(geo.name))
            verts_py = [tuple(v) for v in verts_flat.reshape(-1, 3)]
            faces_py = []
            for start, size in zip(faces_start, faces_size):
                faces_py.append(list(loops_vert[start:start + size]))
            try:
                mesh.from_pydata(verts_py, [], faces_py)
            except ValueError:
                self.warn("Geometry '%s': mesh construction failed" % geo.name)
                return None

        mesh.update(calc_edges=True)
        mesh.validate()

        if len(mesh.polygons) == 0 or len(mesh.loops) == 0:
            try:
                bpy.data.meshes.remove(mesh)
            except RuntimeError:
                pass
            self.warn("Geometry '%s': mesh became empty after validation"
                      % geo.name)
            return None

        counts_ok = (len(mesh.polygons) == nfaces and len(mesh.loops) == nloops)

        # ---- UV layers ----------------------------------------------------
        for si, s in enumerate(uv_sets):
            layer = mesh.uv_layers.new(name=uv_layer_name(s))
            if layer is not None and counts_ok:
                flat = np.asarray(uv_data[s], dtype=np.float32).ravel()
                if len(flat) == len(layer.data) * 2:
                    layer.data.foreach_set("uv", flat)
            if layer is not None and si == 0:
                layer.active = True
                layer.active_render = True

        # ---- vertex colors --------------------------------------------------
        if any_colors and len(loops_col) == nloops:
            try:
                ca = mesh.color_attributes.new(name="Col", type="FLOAT_COLOR",
                                               domain="CORNER")
                flat = np.asarray(loops_col, dtype=np.float32).ravel()
                if len(flat) == len(ca.data) * 4:
                    try:
                        ca.data.foreach_set("color", flat)
                    except AttributeError:
                        ca.foreach_set("color", flat)
            except (RuntimeError, ValueError) as ex:
                self.warn("Geometry '%s': color attribute failed (%s)"
                          % (geo.name, ex))

        # ---- materials --------------------------------------------------------
        for m in slot_mats:
            mesh.materials.append(m)
        if counts_ok:
            mesh.polygons.foreach_set("material_index",
                                      np.asarray(faces_mat, dtype=np.int32))

        # ---- shading --------------------------------------------------------
        if any_normals and len(loops_nrm) == nloops:
            # Smooth shading everywhere; mark edges sharp where authored
            # corner normals disagree for the same vertex.
            mesh.polygons.foreach_set("use_smooth", np.ones(nfaces, dtype=bool))
            first = {}
            sharp = np.zeros(len(mesh.edges), dtype=bool)
            for li in range(nloops):
                v = loops_vert[li]
                x, y, z = loops_nrm[li]
                if x == 0.0 and y == 0.0 and z == 0.0:
                    continue  # missing normal for this corner: inconclusive
                key = (round(x, 5), round(y, 5), round(z, 5))
                prev = first.get(v)
                if prev is None:
                    first[v] = key
                elif prev != key:
                    try:
                        sharp[mesh.loops[li].edge_index] = True
                    except IndexError:
                        pass
            if sharp.any():
                mesh.edges.foreach_set("use_edge_sharp", sharp)

            if self.use_custom_normals and counts_ok:
                _apply_custom_normals(mesh, nloops, loops_nrm)

        self.stats["meshes"] += 1
        return mesh


# ---------------------------------------------------------------------------
# Custom split normals (Blender 5.x API with graceful degradation)
# ---------------------------------------------------------------------------

def _apply_custom_normals(mesh, nloops, loops_nrm):
    """Author the file's exact per-corner normals; fall back silently."""
    try:
        flat = np.asarray(loops_nrm, dtype=np.float32).ravel()
        if len(flat) != nloops * 3:
            return
        try:
            # Works in Blender 5.2 (per-corner custom split normals)
            mesh.normals_split_custom_set(
                [tuple(loops_nrm[i]) for i in range(nloops)])
            return
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass
        try:
            # Future API fallback
            mesh.normals.foreach_set("vector", flat)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass
    except Exception:
        pass