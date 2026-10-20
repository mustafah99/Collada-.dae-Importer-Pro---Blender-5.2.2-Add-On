# SPDX-License-Identifier: GPL-2.0-or-later
"""Collada (.dae) Importer Pro for Blender 5.2.x.

Adds  File > Import > Collada (.dae)  with a material-focused importer:
Principled BSDF materials, diffuse/emission/alpha/bump textures, multiple
UV sets, vertex colors, authored normals, instancing and unit/up-axis
handling.  Pure Python, no external dependencies.
"""

bl_info = {
    "name": "Collada (.dae) Importer Pro",
    "author": "dae-pro contributors",
    "version": (1, 0, 0),
    "blender": (5, 2, 0),
    "location": "File > Import > Collada (.dae)",
    "description": "Import Collada .dae files with full material, texture, "
                   "UV, vertex-color and normal support",
    "doc_url": "",
    "support": "COMMUNITY",
    "category": "Import-Export",
}

import traceback

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                       StringProperty)
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper

from . import importer as _importer
from .parser import ColladaParseError


class IMPORT_OT_collada_pro(Operator, ImportHelper):
    """Import a Collada (.dae) scene with materials and textures"""
    bl_idname = "import_scene.collada_pro"
    bl_label = "Import Collada (.dae)"
    bl_options = {"REGISTER", "UNDO", "PRESET"}

    filename_ext = ".dae"
    filter_glob: StringProperty(default="*.dae", options={"HIDDEN"})

    apply_unit_scale: BoolProperty(
        name="Apply Unit Scale",
        description="Scale objects by the file's <unit> declaration "
                    "(e.g. centimeter files become correctly sized)",
        default=True,
    )
    manual_scale: FloatProperty(
        name="Manual Scale",
        description="Extra scale factor applied on top of the unit scale",
        default=1.0, min=0.0001, soft_min=0.001, soft_max=100.0,
    )
    up_axis: EnumProperty(
        name="Up Axis",
        description="Convert the file's up axis to Blender's Z-up",
        items=(
            ("AUTO", "Automatic", "Use the up axis stored in the file's <asset>"),
            ("Z_UP", "Z Up", "Treat file coordinates as already Z-up"),
            ("Y_UP", "Y Up", "Treat file coordinates as Y-up"),
            ("X_UP", "X Up", "Treat file coordinates as X-up"),
        ),
        default="AUTO",
    )
    use_custom_normals: BoolProperty(
        name="Use File Normals",
        description="Author the normals stored in the file (smooth/hard "
                    "edges as exported), instead of recomputing them",
        default=True,
    )
    search_textures: BoolProperty(
        name="Search Textures in Subfolders",
        description="When a texture cannot be found at its stored path, "
                    "search the .dae's folder (and subfolders) by filename",
        default=True,
    )

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        box = layout.box()
        box.label(text="Transform", icon="OBJECT_ORIGIN")
        col = box.column(align=True)
        col.prop(self, "apply_unit_scale")
        col.prop(self, "manual_scale")
        col.prop(self, "up_axis")

        box = layout.box()
        box.label(text="Shading", icon="SHADERFX")
        box.prop(self, "use_custom_normals")

        box = layout.box()
        box.label(text="Textures", icon="TEXTURE")
        box.prop(self, "search_textures")

    def execute(self, context):
        try:
            summary = _importer.import_dae(
                context,
                self.filepath,
                apply_unit_scale=self.apply_unit_scale,
                manual_scale=self.manual_scale,
                up_axis_mode=self.up_axis,
                use_custom_normals=self.use_custom_normals,
                search_textures=self.search_textures,
            )
        except ColladaParseError as ex:
            self.report({"ERROR"}, "Collada import: %s" % ex)
            return {"CANCELLED"}
        except Exception:
            traceback.print_exc()
            self.report({"ERROR"}, "Import failed - see Console for details")
            return {"CANCELLED"}

        msg = ("Imported %d object(s), %d material(s), %d texture(s)"
               % (summary["objects"], summary["materials"],
                  summary["textures"]))
        if summary["missing_textures"]:
            self.report({"WARNING"},
                        "%s - %d texture(s) missing (see Console)"
                        % (msg, summary["missing_textures"]))
        elif summary["warnings"]:
            self.report({"WARNING"},
                        "%s - %d warning(s) (see Console)"
                        % (msg, len(summary["warnings"])))
        else:
            self.report({"INFO"}, msg)

        for w in summary["warnings"]:
            print("[dae_pro]", w)
        return {"FINISHED"}


def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_collada_pro.bl_idname,
                         text="Collada (.dae)")


classes = (IMPORT_OT_collada_pro,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()