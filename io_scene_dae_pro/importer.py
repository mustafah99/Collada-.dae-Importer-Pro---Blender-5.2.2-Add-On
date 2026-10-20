# SPDX-License-Identifier: GPL-2.0-or-later
"""Scene import orchestration: parse -> materials -> meshes -> objects."""

import os

import bpy
from mathutils import Matrix

from .materials import MaterialService, TextureResolver
from .meshes import MeshService, matrix_from_collada
from .parser import parse_collada


def _axis_matrix(up_axis):
    """Rotation converting the file's up axis to Blender's +Z up."""
    if up_axis == "Y_UP":
        return Matrix.Rotation(1.5707963267948966, 4, "X")     # +90 deg
    if up_axis == "X_UP":
        return Matrix.Rotation(-1.5707963267948966, 4, "Y")    # -90 deg
    return Matrix.Identity(4)                                   # Z_UP


def import_dae(context, filepath,
               apply_unit_scale=True,
               manual_scale=1.0,
               up_axis_mode="AUTO",
               use_custom_normals=True,
               search_textures=True,
               new_collection=True):
    """Import a Collada file into `context.scene`.

    Returns a summary dict with counts and warnings.  Raises
    parser.ColladaParseError for unusable files; other errors propagate.
    """
    doc = parse_collada(filepath)

    resolver = TextureResolver(doc, search_subdirs=search_textures)
    mat_service = MaterialService(doc, resolver)
    mesh_service = MeshService(doc, mat_service,
                               use_custom_normals=use_custom_normals)

    up_axis = doc.up_axis if up_axis_mode == "AUTO" else up_axis_mode
    fix = Matrix.Scale(manual_scale, 4)
    if apply_unit_scale and abs(doc.unit_scale - 1.0) > 1e-12:
        fix = Matrix.Scale(doc.unit_scale, 4) @ fix
    fix = _axis_matrix(up_axis) @ fix

    scene = context.scene
    collection = None
    if new_collection:
        base = os.path.splitext(os.path.basename(filepath))[0] or "dae_import"
        collection = bpy.data.collections.new(base[:63])
        scene.collection.children.link(collection)

    created = []
    identity = Matrix.Identity(4)
    node_world = {}

    def _link(obj):
        if collection is not None:
            collection.objects.link(obj)
        else:
            scene.collection.objects.link(obj)

    def _instantiate(node, world):
        instances = list(node.geom_instances)
        for ctrl_id, binds in node.ctrl_instances:
            geo_id = doc.controllers.get(ctrl_id)
            if geo_id:
                instances.append((geo_id, binds))
            else:
                doc.warn("Controller '%s' references unknown skin/morph "
                         "source: skipped" % ctrl_id)

        for i, (geo_id, binds) in enumerate(instances):
            try:
                mesh = mesh_service.get_mesh(geo_id, binds)
            except Exception as ex:
                doc.warn("Geometry '%s' failed: %s" % (geo_id, ex))
                continue
            if mesh is None:
                continue
            if len(instances) > 1:
                name = "%s.%d" % (node.name, i + 1)
            else:
                name = node.name
            obj = bpy.data.objects.new(name, mesh)
            obj.matrix_world = fix @ world
            _link(obj)
            created.append(obj)

    def _visit(node, parent_world, path):
        world = parent_world @ matrix_from_collada(node.matrix)
        if node.id:
            node_world[node.id] = world
        _instantiate(node, world)
        for url in node.node_refs:
            if url in path:
                doc.warn("Cyclic instance_node reference '%s' skipped" % url)
                continue
            target = doc.nodes.get(url)
            if target is not None:
                _visit(target, world, path | {url})
            else:
                doc.warn("instance_node '%s' not found" % url)
        for child in node.children:
            _visit(child, world, path)

    for root in doc.roots:
        _visit(root, identity, frozenset())

    if hasattr(context, "view_layer") and context.view_layer is not None:
        try:
            context.view_layer.update()
        except RuntimeError:
            pass

    if doc.controllers:
        doc.warn("%d skin controller(s) imported as static bind-pose meshes "
                 "(no rigging in this importer)" % len(doc.controllers))
    cam_light = sum(1 for _ in _iter_cameras_lights(doc))
    if cam_light:
        doc.warn("%d camera/light node(s) skipped (mesh+materials importer)"
                 % cam_light)

    missing = sorted(resolver.missing)
    for m in missing:
        doc.warn("Texture not found: %s" % m)

    return {
        "objects": len(created),
        "materials": mat_service.stats["materials"],
        "textures": mat_service.stats["textures"],
        "missing_textures": len(missing),
        "meshes": mesh_service.stats["meshes"],
        "warnings": doc.warnings,
    }


def _iter_cameras_lights(doc):
    def walk(node):
        if node.has_camera or node.has_light:
            yield node
        for c in node.children:
            yield from walk(c)
    for root in doc.roots:
        yield from walk(root)