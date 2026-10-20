# SPDX-License-Identifier: GPL-2.0-or-later
"""Convert Collada effects into Blender node materials (Principled BSDF)."""

import os

import bpy


def uv_layer_name(set_index):
    """Canonical UV layer name for a Collada TEXCOORD set index.

    Must stay in sync with the mesh builder, which creates layers with the
    same naming rule."""
    if set_index in (0, -1):
        return "UVMap"
    return "UVMap.%d" % set_index


def _luminance(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class TextureResolver:
    """Finds real texture files for the paths referenced by a Collada file."""

    def __init__(self, doc, search_subdirs=True):
        self.doc = doc
        self.dae_dir = os.path.dirname(os.path.abspath(doc.filepath))
        self.search_subdirs = search_subdirs
        self._name_map = None
        self._image_cache = {}       # resolved path -> bpy image
        self.missing = set()         # raw paths that could not be found

    # -- filesystem search ------------------------------------------------

    def _build_name_map(self):
        self._name_map = {}
        if not self.search_subdirs or not os.path.isdir(self.dae_dir):
            return
        for root, _dirs, files in os.walk(self.dae_dir):
            for fn in files:
                low = fn.lower()
                if low not in self._name_map:
                    self._name_map[low] = os.path.join(root, fn)

    def resolve(self, raw_path):
        """Return an absolute filesystem path for a Collada image path."""
        if not raw_path:
            return None
        p = raw_path
        candidates = []
        if os.path.isabs(p):
            candidates.append(os.path.normpath(p))
        else:
            candidates.append(os.path.normpath(os.path.join(self.dae_dir, p)))
        base = os.path.basename(p)
        candidates.append(os.path.normpath(os.path.join(self.dae_dir, base)))
        if self._name_map is None:
            self._build_name_map()
        if self._name_map:
            hit = self._name_map.get(base.lower())
            if hit:
                candidates.append(hit)
        for c in candidates:
            if os.path.isfile(c):
                return c
        return None

    # -- blender images ----------------------------------------------------

    def get_image(self, raw_path, non_color=False):
        """Return a bpy image for a raw Collada path (or None)."""
        resolved = self.resolve(raw_path)
        if resolved is None:
            if raw_path and raw_path not in self.missing:
                self.missing.add(raw_path)
            return None
        img = self._image_cache.get(resolved)
        if img is None or img.name not in bpy.data.images:
            try:
                img = bpy.data.images.load(resolved, check_existing=True)
            except RuntimeError:
                self.missing.add(raw_path)
                return None
            self._image_cache[resolved] = img
        try:
            img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
        except (TypeError, AttributeError):
            pass
        return img


class MaterialService:
    """Builds & caches Blender materials for (material id, uv binding) keys."""

    def __init__(self, doc, resolver):
        self.doc = doc
        self.resolver = resolver
        self.cache = {}              # (mat_id, uvbinds_key) -> material
        self.stats = {"materials": 0, "textures": 0, "missing_textures": 0}
        self._fallback_cache = {}    # symbol -> default material

    # -- public API ---------------------------------------------------------

    def get_for_binds(self, binds, symbol):
        """Material for a primitive's `material` symbol given the
        instance_material list bound on the node."""
        inst = None
        if binds:
            for b in binds:
                if b.symbol == symbol:
                    inst = b
                    break
        if inst is None:
            # Some exporters put the material id directly as the symbol
            target = symbol
            uv_key = ()
            if target in self.doc.materials:
                return self._get_material(target, uv_key)
            return self._get_fallback(symbol)

        target = inst.target
        if target not in self.doc.materials:
            # target might reference an effect id directly
            if target in self.doc.effects:
                pseudo = "%s-fx" % target
                if pseudo not in self._fallback_cache:
                    mat = self._build_from_effect(self.doc.effects[target],
                                                  inst.uvbinds, pseudo)
                    self._fallback_cache[pseudo] = mat
                return self._fallback_cache[pseudo]
            return self._get_fallback(symbol)

        uv_key = tuple(sorted(inst.uvbinds.items()))
        return self._get_material(target, uv_key)

    def resolve_uv_set(self, uvbinds, texcoord_semantic):
        """UV set index for a texture's texcoord semantic under given binds."""
        if uvbinds and texcoord_semantic:
            s = uvbinds.get(texcoord_semantic)
            if s is not None:
                return s
            # case-insensitive fallback
            low = texcoord_semantic.lower()
            for k, v in uvbinds.items():
                if k.lower() == low:
                    return v
        return 0

    # -- internals ------------------------------------------------------

    def _get_material(self, mat_id, uv_key):
        key = (mat_id, uv_key)
        mat = self.cache.get(key)
        if mat is not None and mat.name in bpy.data.materials:
            return mat
        mat_ir = self.doc.materials[mat_id]
        effect = self.doc.effects.get(mat_ir.effect_id)
        if effect is None:
            mat = self._get_fallback(mat_ir.name or mat_id)
        else:
            uvbinds = dict(uv_key)
            mat = self._build_from_effect(effect, uvbinds, mat_ir.name or mat_id)
        self.cache[key] = mat
        return mat

    def _get_fallback(self, symbol):
        mat = self._fallback_cache.get(symbol)
        if mat is not None and mat.name in bpy.data.materials:
            return mat
        mat = bpy.data.materials.new(_safe_name(symbol) or "DaeMaterial")
        mat.use_nodes = True
        bsdf = _find_bsdf(mat)
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        self._fallback_cache[symbol] = mat
        self.stats["materials"] += 1
        return mat

    # -- the actual node graph -------------------------------------------

    def _build_from_effect(self, effect, uvbinds, name):
        mat = bpy.data.materials.new(_safe_name(name) or "DaeMaterial")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()

        out = nt.nodes.new("ShaderNodeOutputMaterial")
        out.location = (420, 0)
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (80, 0)
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        self.stats["materials"] += 1

        # ---- Base Color -------------------------------------------------
        diffuse = effect.diffuse
        diffuse_img = None
        if diffuse is not None and diffuse.kind == "texture":
            raw = self._image_path(effect, diffuse.tex)
            diffuse_img = self.resolver.get_image(raw, non_color=False)
            if diffuse_img is not None:
                tex = _image_node(nt, diffuse_img, (-300, 260))
                _link_uv(nt, tex, uvbinds, diffuse.tex.texcoord, self)
                _link(nt, tex.outputs["Color"], bsdf.inputs["Base Color"])
                self.stats["textures"] += 1
            else:
                bsdf.inputs["Base Color"].default_value = (1.0, 0.0, 1.0, 1.0)
        elif diffuse is not None and diffuse.kind == "color":
            bsdf.inputs["Base Color"].default_value = diffuse.color

        # ---- Alpha / transparency ----------------------------------------
        alpha_used = False
        if effect.transparent is not None:
            alpha_img = None
            if effect.transparent.kind == "texture":
                raw = self._image_path(effect, effect.transparent.tex)
                # Reuse the diffuse image when both point to the same file
                alpha_img = self.resolver.get_image(raw, non_color=False)
                if alpha_img is not None:
                    self.stats["textures"] += 1
            transparency = _clamp(effect.transparency, 0.0, 1.0)

            if alpha_img is not None:
                tex = _image_node(nt, alpha_img, (-300, -140))
                _link_uv(nt, tex, uvbinds, effect.transparent.tex.texcoord, self)
                src = tex.outputs["Alpha"]
                if transparency < 0.999:
                    math_n = nt.nodes.new("ShaderNodeMath")
                    math_n.operation = "MULTIPLY"
                    math_n.location = (-60, -140)
                    math_n.inputs[1].default_value = transparency
                    nt.links.new(src, math_n.inputs[0])
                    src = math_n.outputs[0]
                _link(nt, src, bsdf.inputs["Alpha"])
                alpha_used = True
            elif transparency < 0.999:
                a = transparency
                if effect.opaque_mode.startswith("RGB_ZERO") and \
                        effect.transparent.kind == "color":
                    a = 1.0 - _luminance(effect.transparent.color) * transparency
                bsdf.inputs["Alpha"].default_value = _clamp(a, 0.0, 1.0)
                alpha_used = True

        if alpha_used:
            _set_blend_mode(mat)

        # ---- Emission -----------------------------------------------------
        emission = effect.emission
        if emission is not None:
            if emission.kind == "texture":
                raw = self._image_path(effect, emission.tex)
                img = self.resolver.get_image(raw, non_color=False)
                if img is not None:
                    tex = _image_node(nt, img, (-300, 520))
                    _link_uv(nt, tex, uvbinds, emission.tex.texcoord, self)
                    _link(nt, tex.outputs["Color"],
                          _in(bsdf, "Emission Color", "Emission"))
                    _set_input(bsdf, "Emission Strength", 1.0)
                    self.stats["textures"] += 1
            elif emission.kind == "color" and max(emission.color[:3]) > 0.001:
                _in(bsdf, "Emission Color", "Emission").default_value = emission.color
                _set_input(bsdf, "Emission Strength", 1.0)

        # ---- Specular / roughness -----------------------------------------
        if effect.shader in ("phong", "blinn"):
            spec_col = effect.specular.color if (
                effect.specular is not None and effect.specular.kind == "color") \
                else (0.5, 0.5, 0.5, 1.0)
            spec_level = _clamp(_luminance(spec_col), 0.0, 1.0)
            _set_input(bsdf, "Specular IOR Level", spec_level, "Specular")
            tint = _in(bsdf, "Specular Tint")
            if tint is not None:
                tint.default_value = (spec_col[0], spec_col[1], spec_col[2], 1.0)
            rough = math_sqrt(2.0 / (max(effect.shininess, 0.0) + 2.0))
            _set_input(bsdf, "Roughness", _clamp(rough, 0.03, 1.0))
        elif effect.shader == "lambert":
            _set_input(bsdf, "Specular IOR Level", 0.0, "Specular")
            _set_input(bsdf, "Roughness", 1.0)
        # 'constant' and unknown shaders: keep Principled defaults

        # ---- Bump / normal map from vendor extras ---------------------------
        if effect.bump is not None:
            raw = self._image_path(effect, effect.bump)
            img = self.resolver.get_image(raw, non_color=True)
            if img is not None:
                tex = _image_node(nt, img, (-600, -420))
                _link_uv(nt, tex, uvbinds, effect.bump.texcoord, self)
                self.stats["textures"] += 1
                if effect.bump_is_normal:
                    nmap = nt.nodes.new("ShaderNodeNormalMap")
                    nmap.location = (-160, -420)
                    nt.links.new(tex.outputs["Color"], nmap.inputs["Color"])
                    _link(nt, nmap.outputs["Normal"], bsdf.inputs["Normal"])
                else:
                    bump = nt.nodes.new("ShaderNodeBump")
                    bump.location = (-160, -420)
                    bump.inputs["Strength"].default_value = 1.0
                    nt.links.new(tex.outputs["Color"], bump.inputs["Height"])
                    _link(nt, bump.outputs["Normal"], bsdf.inputs["Normal"])

        return mat

    def _image_path(self, effect, texref):
        """Best-effort raw path for a TexRef (image id -> path, or the id)."""
        img_id = texref.image_id
        entry = self.doc.images.get(img_id) if img_id else None
        if entry is not None and entry.path:
            return entry.path
        # texture attr may itself have been a path / filename
        if img_id and ("/" in img_id or "\\" in img_id or "." in os.path.basename(img_id)):
            return img_id
        # Last resort: look for any image whose id matches loosely
        entry = self.doc.images.get(texref.sampler)
        if entry is not None and entry.path:
            return entry.path
        return texref.sampler or img_id or ""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def math_sqrt(v):
    return v ** 0.5


def _safe_name(n):
    if not n:
        return ""
    return n.replace("/", "_").replace("\\", "_").strip()[:63]


def _find_bsdf(mat):
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    return None


def _in(node, *names):
    """Return the first existing input socket among `names`."""
    for n in names:
        s = node.inputs.get(n)
        if s is not None:
            return s
    return None


def _set_input(node, name, value, *fallback_names):
    s = _in(node, name, *fallback_names)
    if s is not None:
        try:
            s.default_value = value
        except (TypeError, ValueError):
            pass
    return s


def _link(nt, from_socket, to_socket):
    if from_socket is not None and to_socket is not None:
        nt.links.new(from_socket, to_socket)


def _image_node(nt, img, location):
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.location = location
    tex.image = img
    tex.interpolation = "Linear"
    return tex


def _link_uv(nt, tex_node, uvbinds, texcoord_semantic, service):
    """Add a UV Map node pointing at the layer matching the bind, if it is
    not the default first layer."""
    set_idx = service.resolve_uv_set(uvbinds, texcoord_semantic)
    lname = uv_layer_name(set_idx)
    if lname == "UVMap":
        # still wire the explicit UV node: harmless & keeps multi-uv stable
        pass
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = lname
    uv.location = (tex_node.location.x - 200, tex_node.location.y - 60)
    _link(nt, uv.outputs["UV"], tex_node.inputs["Vector"])


def _set_blend_mode(mat):
    """Enable alpha blending across Blender versions."""
    try:
        mat.surface_render_method = "BLENDED"     # Blender 4.2+
    except (AttributeError, TypeError):
        pass
    try:
        mat.blend_method = "BLEND"                # Blender <= 4.1 (removed after)
    except (AttributeError, TypeError):
        pass