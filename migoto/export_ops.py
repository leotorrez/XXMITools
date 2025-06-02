import collections
import json
import time
from pathlib import Path
from typing import Callable
import textwrap
import shutil
import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    PointerProperty,
    StringProperty,
    IntProperty,
)
from bpy.types import Context, Mesh, Object, Operator, PropertyGroup
from bpy_extras.io_utils import ExportHelper

from .data.byte_buffer import (
    AbstractSemantic,
    BufferLayout,
    Semantic,
)
from .data.dxgi_format import DXGIType
from .datahandling import (
    Fatal,
    custom_attributes_float,
    custom_attributes_int,
    keys_to_ints,
    mesh_triangulate,
)
from .datastructures import (
    GameEnum,
    HashableVertex,
    IndexBuffer,
    InputLayout,
    VertexBufferGroup,
    game_enum,
)
from .exporter import ModExporter


def normal_export_translation(
    layouts: list[BufferLayout], semantic: Semantic, flip: bool
) -> Callable:
    unorm = False
    for layout in layouts:
        try:
            unorm = layout.get_element(AbstractSemantic(semantic)).format.dxgi_type in [
                DXGIType.UNORM8,
                DXGIType.UNORM16,
            ]
        except ValueError:
            continue
    if unorm:
        # Scale normal range -1:+1 to UNORM range 0:+1
        if flip:
            return lambda x: -x / 2.0 + 0.5
        return lambda x: x / 2.0 + 0.5
    if flip:
        return lambda x: -x
    return lambda x: x


def apply_modifiers_and_shapekeys(context: Context, obj: Object) -> Mesh:
    """Apply all modifiers to a mesh with shapekeys. Preserves shapekeys named Deform"""
    start_timer = time.time()
    deform_SKs = []
    total_applied = 0
    desgraph = context.evaluated_depsgraph_get()
    modifiers_to_apply = [mod for mod in obj.modifiers if mod.show_viewport]
    if obj.data.shape_keys is not None:
        deform_SKs = [
            sk.name
            for sk in obj.data.shape_keys.key_blocks
            if "deform" in sk.name.lower()
        ]
        total_applied = len(obj.data.shape_keys.key_blocks) - len(deform_SKs)

    if len(deform_SKs) == 0:
        mesh = obj.evaluated_get(desgraph).to_mesh()
    else:
        mesh = obj.to_mesh()
        result_obj = obj.copy()
        result_obj.data = mesh.copy()
        context.collection.objects.link(result_obj)
        for sk in obj.data.shape_keys.key_blocks:
            if sk.name not in deform_SKs:
                result_obj.shape_key_remove(sk)
        list_properties = []
        vert_count = -1
        bpy.ops.object.select_all(action="DESELECT")
        result_obj.select_set(True)
        bpy.ops.object.duplicate_move(
            OBJECT_OT_duplicate={"linked": False, "mode": "TRANSLATION"},
            TRANSFORM_OT_translate={
                "value": (0, 0, 0),
                "orient_type": "GLOBAL",
                "orient_matrix": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                "orient_matrix_type": "GLOBAL",
                "constraint_axis": (False, False, False),
                "mirror": True,
                "use_proportional_edit": False,
                "proportional_edit_falloff": "SMOOTH",
                "proportional_size": 1,
                "use_proportional_connected": False,
                "use_proportional_projected": False,
                "snap": False,
                "snap_target": "CLOSEST",
                "snap_point": (0, 0, 0),
                "snap_align": False,
                "snap_normal": (0, 0, 0),
                "gpencil_strokes": False,
                "cursor_transform": False,
                "texture_space": False,
                "remove_on_cancel": False,
                "release_confirm": False,
                "use_accurate": False,
            },
        )
        copy_obj = context.view_layer.objects.active
        copy_obj.select_set(False)
        context.view_layer.objects.active = result_obj
        result_obj.select_set(True)
        # Store key shape properties
        for key_b in obj.data.shape_keys.key_blocks:
            properties_object = {}
            properties_object["name"] = key_b.name
            properties_object["mute"] = key_b.mute
            properties_object["interpolation"] = key_b.interpolation
            properties_object["relative_key"] = key_b.relative_key.name
            properties_object["slider_max"] = key_b.slider_max
            properties_object["slider_min"] = key_b.slider_min
            properties_object["value"] = key_b.value
            properties_object["vertex_group"] = key_b.vertex_group
            list_properties.append(properties_object)
            result_obj.shape_key_remove(key_b)
        # Set up Basis
        result_obj = result_obj.evaluated_get(desgraph)
        # bpy.ops.object.shape_key_add(from_mix=False)
        # for mod in modifiers_to_apply:
        #     bpy.ops.object.modifier_apply(modifier=mod.name)
        mesh = result_obj.to_mesh()
        vert_count = len(result_obj.data.vertices)
        result_obj.select_set(False)
        # Create a temp object to apply modifiers into once per SK
        for i in range(1, obj.data.shape_keys.key_blocks):
            context.view_layer.objects.active = copy_obj
            copy_obj.select_set(True)
            bpy.ops.object.duplicate_move(
                OBJECT_OT_duplicate={"linked": False, "mode": "TRANSLATION"},
                TRANSFORM_OT_translate={
                    "value": (0, 0, 0),
                    "orient_type": "GLOBAL",
                    "orient_matrix": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                    "orient_matrix_type": "GLOBAL",
                    "constraint_axis": (False, False, False),
                    "mirror": True,
                    "use_proportional_edit": False,
                    "proportional_edit_falloff": "SMOOTH",
                    "proportional_size": 1,
                    "use_proportional_connected": False,
                    "use_proportional_projected": False,
                    "snap": False,
                    "snap_target": "CLOSEST",
                    "snap_point": (0, 0, 0),
                    "snap_align": False,
                    "snap_normal": (0, 0, 0),
                    "gpencil_strokes": False,
                    "cursor_transform": False,
                    "texture_space": False,
                    "remove_on_cancel": False,
                    "release_confirm": False,
                    "use_accurate": False,
                },
            )
            temp_obj = context.view_layer.objects.active
            for k in temp_obj.data.shape_keys.key_blocks:
                temp_obj.shape_key_remove(k)

            copy_obj.select_set(True)
            copy_obj.active_shape_key_index = i

            bpy.ops.object.shape_key_transfer(use_clamp=True)
            context.object.active_shape_key_index = 0
            bpy.ops.object.shape_key_remove()
            bpy.ops.object.shape_key_remove(all=True)
            for mod in modifiers_to_apply:
                bpy.ops.object.modifier_apply(modifier=mod.name)
            if vert_count != len(temp_obj.data.vertices):
                raise Fatal(
                    f"After modifier application, object {obj.name} has a different vertex count in shape key {i} than in the basis shape key. Manual resolution required."
                )
            copy_obj.select_set(False)
            context.view_layer.objects.active = result_obj
            result_obj.select_set(True)
            bpy.ops.object.join_shapes()
            result_obj.select_set(False)
            context.view_layer.objects.active = temp_obj
            bpy.ops.object.delete(use_global=False)
        # Restore shape key properties like name, mute etc.
        context.view_layer.objects.active = result_obj
        for i in range(0, obj.data.shape_keys.key_blocks):
            key_b = context.view_layer.objects.active.data.shape_keys.key_blocks[i]
            key_b.name = list_properties[i]["name"]
            key_b.interpolation = list_properties[i]["interpolation"]
            key_b.mute = list_properties[i]["mute"]
            key_b.slider_max = list_properties[i]["slider_max"]
            key_b.slider_min = list_properties[i]["slider_min"]
            key_b.value = list_properties[i]["value"]
            key_b.vertex_group = list_properties[i]["vertex_group"]
            rel_key = list_properties[i]["relative_key"]

            for j in range(0, obj.data.shape_keys.key_blocks):
                key_brel = context.view_layer.objects.active.data.shape_keys.key_blocks[
                    j
                ]
                if rel_key == key_brel.name:
                    key_b.relative_key = key_brel
                    break
            context.view_layer.objects.active.data.update()
        result_obj.select_set(False)
        context.view_layer.objects.active = copy_obj
        copy_obj.select_set(True)
        bpy.ops.object.delete(use_global=False)
        bpy.ops.object.select_all(action="DESELECT")
        context.view_layer.objects.active = result_obj
        context.view_layer.objects.active.select_set(True)
        mesh = result_obj.data
        bpy.ops.object.delete(use_global=False)

    print(
        f"\tApplied {len(modifiers_to_apply)} modifiers, {total_applied} shapekeys and stored {len(deform_SKs)} shapekeys in {time.time() - start_timer:.5f} seconds"
    )
    return mesh


class Export3DMigoto(Operator, ExportHelper):
    """Export a mesh for re-injection into a game with 3DMigoto"""

    bl_idname = "export_mesh.migoto"
    bl_label = "Export 3DMigoto Vertex & Index Buffers"

    filename_ext = ".vb0"
    filter_glob: StringProperty(
        default="*.vb*",
        options={"HIDDEN"},
    )

    def invoke(self, context, event):
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        try:
            file_path = Path(self.filepath)
            vb_path = file_path.parent / (file_path.stem + ".vb")
            ib_path = file_path.parent / (file_path.stem + ".ib")
            fmt_path = file_path.parent / (file_path.stem + ".fmt")
            ini_path = file_path.parent / (file_path.stem + "_generated.ini")
            obj = context.object
            self.flip_normal = obj.get("3DMigoto:FlipNormal", False)
            self.flip_tangent = obj.get("3DMigoto:FlipTangent", False)
            self.flip_winding = obj.get("3DMigoto:FlipWinding", False)
            self.flip_mesh = obj.get("3DMigoto:FlipMesh", False)
            # FIXME: ExportHelper will check for overwriting vb_path, but not ib_path
            export_3dmigoto(self, context, vb_path, ib_path, fmt_path, ini_path)
        except Fatal as e:
            self.report({"ERROR"}, str(e))
        return {"FINISHED"}


class Export3DMigotoXXMI(Operator, ExportHelper):
    """Export a mesh for re-injection into a game with 3DMigoto"""

    bl_idname = "export_mesh_xxmi.migoto"
    bl_label = "Export mod folder"
    bl_options = {"PRESET", "UNDO"}

    filename_ext = ".vb*"
    filter_glob: StringProperty(
        default="*.vb*",
        options={"HIDDEN"},
    )

    ignore_hidden: BoolProperty(
        name="Ignore hidden objects",
        description="Does not use objects in the Blender window that are hidden while exporting mods",
        default=True,
    )

    only_selected: BoolProperty(
        name="Only export selected",
        description="Uses only the selected objects when deciding which meshes to export",
        default=False,
    )

    no_ramps: BoolProperty(
        name="Ignore shadow ramps/metal maps/diffuse guide",
        description="Skips exporting shadow ramps, metal maps and diffuse guides",
        default=True,
    )

    copy_textures: BoolProperty(
        name="Mod textures",
        description="ENABLED: Writes to the INI file and copy missing texture files to the export folder.\nDISABLED: The INI file will not contain entries that mod the textures. i.e. Mod uses vanilla textures.",
        default=True,
    )

    ignore_duplicate_textures: BoolProperty(
        name="Ignore duplicated textures",
        description="Ignore new textures with the same hash as already copied ones.",
        default=False,
    )

    credit: StringProperty(
        name="Credit",
        description="Name that pops up on screen when mod is loaded. If left blank, will result in no pop up",
        default="",
    )

    game: EnumProperty(
        name="Game to mod",
        description="Select the game you are modding to optimize the mod for that game",
        items=game_enum,
    )
    apply_modifiers_and_shapekeys: BoolProperty(
        name="Apply modifiers and shapekeys",
        description="Applies shapekeys and modifiers(unless marked MASK); then joins meshes to a single object. The criteria to join is as follows, the objects imported from dump are considered containers; collections starting with their same name are going to be joint into said containers",
        default=False,
    )
    normalize_weights: BoolProperty(
        name="Normalize weights to format",
        description="Limits weights to match export format. Also normalizes the remaining weights",
        default=False,
    )
    outline_optimization: EnumProperty(
        name="Outline Optimization",
        description="Recalculate outlines. Recommended for final export. Check more options below to improve quality",
        items=[
            ("OFF", "Deactivate", "No outline optimization"),
            (
                "ON",
                "Traditional",
                "Traditional outline optimization, used in old script. May be slow for high vertex count meshes",
            ),
            (
                "EXPERIMENTAL",
                "Fast Experimental",
                "Experimental fast outline optimization, may produce artifacts",
            ),
        ],
        default="OFF",
    )
    outline_rounding_precision: IntProperty(
        name="Outline decimal rounding precision",
        description="Higher values merge farther away vertices into a single outline vertex. Lower values produce more accurate outlines, but may result in split edges",
        default=4,
        min=1,
        max=10,
    )
    export_shapekeys: BoolProperty(
        name="Export shape keys",
        description="Exports marked shape keys for the selected object. Also generates the necessary sections in ini file",
        default=False,
    )
    write_buffers: BoolProperty(
        name="Write buffers",
        description="Writes the vertex and index buffers to disk. Disabling this won't refresh the buffers in the mod folder, useful for debugging.",
        default=True,
    )
    write_ini: BoolProperty(
        name="Write ini",
        description="Writes the ini file to disk. Disabling this won't refresh the ini file in the mod folder, useful for debugging.",
        default=True,
    )

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.prop(self, "game")
        col.prop(self, "ignore_hidden")
        col.prop(self, "only_selected")
        col.prop(self, "apply_modifiers_and_shapekeys")
        col.prop(self, "normalize_weights")
        # col.prop(self, 'export_shapekeys')
        col.prop(self, "outline_optimization")
        if self.outline_optimization != "OFF":
            col.prop(self, "outline_rounding_precision")
        col.prop(self, "copy_textures")
        if self.copy_textures:
            col.prop(self, "no_ramps")
            col.prop(self, "ignore_duplicate_textures")
        col.prop(self, "write_buffers")
        col.prop(self, "write_ini")
        col.prop(self, "credit")

    def execute(self, context):
        try:
            mod_exporter: ModExporter = ModExporter(
                context=context,
                operator=self,
                dump_path=Path(self.filepath),
                destination=Path(""),
                game=GameEnum[self.properties.game],
                ignore_hidden=self.ignore_hidden,
                only_selected=self.only_selected,
                no_ramps=self.no_ramps,
                copy_textures=self.copy_textures,
                ignore_duplicate_textures=self.ignore_duplicate_textures,
                credit=self.credit,
                outline_optimization=self.outline_optimization,
                apply_modifiers=self.apply_modifiers_and_shapekeys,
                normalize_weights=self.normalize_weights,
                write_ini=self.write_ini,
                write_buffers=self.write_buffers,
            )
            mod_exporter.export()
        except Fatal as e:
            self.report({"ERROR"}, str(e))
        return {"FINISHED"}


class XXMIProperties(PropertyGroup):
    """Properties for XXMITools"""

    destination_path: StringProperty(
        name="Output Folder",
        description="Output Folder:",
        default="",
        maxlen=1024,
    )

    dump_path: StringProperty(
        name="Dump Folder",
        description="Dump Folder:",
        default="",
        maxlen=1024,
    )
    filter_glob: StringProperty(
        default="*.vb*",
        options={"HIDDEN"},
    )
    use_custom_template: BoolProperty(
        name="Use custom template",
        description="Use a custom template file for the mod. If unchecked, the default template will be used",
        default=False,
    )
    template_path: StringProperty(
        name="Template Path",
        description="Path to the template file.",
        maxlen=1024,
    )

    flip_winding: BoolProperty(
        name="Flip Winding Order",
        description="Flip winding order during export (automatically set to match the import option)",
        default=False,
    )

    ignore_hidden: BoolProperty(
        name="Ignore hidden objects",
        description="Does not use objects in the Blender window that are hidden while exporting mods",
        default=True,
    )

    only_selected: BoolProperty(
        name="Only export selected",
        description="Uses only the selected objects when deciding which meshes to export",
        default=False,
    )

    no_ramps: BoolProperty(
        name="Ignore shadow ramps/metal maps/diffuse guide",
        description="Skips exporting shadow ramps, metal maps and diffuse guides",
        default=True,
    )

    copy_textures: BoolProperty(
        name="Mod textures",
        description="ENABLED: Writes to the INI file and copy missing texture files to the export folder.\nDISABLED: The INI file will not contain entries that mod the textures. i.e. Mod uses vanilla textures.",
        default=True,
    )

    ignore_duplicate_textures: BoolProperty(
        name="Ignore duplicated textures",
        description="Ignore new textures with the same hash as already copied ones",
        default=False,
    )

    credit: StringProperty(
        name="Credit",
        description="Name that pops up on screen when mod is loaded. If left blank, will result in no pop up",
        default="",
    )

    outline_optimization: BoolProperty(
        name="Outline Optimization",
        description="Recalculate outlines. Recommended for final export. Check more options below to improve quality",
        default=False,
    )
    outline_rounding_precision: IntProperty(
        name="Outline decimal rounding precision",
        description="Higher values merge farther away vertices into a single outline vertex. Lower values produce more accurate outlines, but may result in split edges",
        default=4,
        min=1,
        max=10,
    )
    game: EnumProperty(
        name="Game to mod",
        description="Select the game you are modding to optimize the mod for that game",
        items=game_enum,
        default=None,
    )
    apply_modifiers_and_shapekeys: BoolProperty(
        name="Apply modifiers and shapekeys",
        description="Applies shapekeys and modifiers(unless marked MASK); then joins meshes to a single object. The criteria to join is as follows, the objects imported from dump are considered containers; collections starting with their same name are going to be joint into said containers",
        default=False,
    )
    normalize_weights: BoolProperty(
        name="Normalize weights to format",
        description="Limits weights to match export format. Also normalizes the remaining weights",
        default=False,
    )
    export_shapekeys: BoolProperty(
        name="Export shape keys",
        description="Exports marked shape keys for the selected object. Also generates the necessary sections in ini file",
        default=False,
    )
    batch_pattern: StringProperty(
        name="Batch pattern",
        description="Pattern to name export folders. Example: name_###",
        default="",
    )
    write_buffers: BoolProperty(
        name="Write buffers",
        description="Writes the vertex and index buffers to disk. Disabling this won't refresh the buffers in the mod folder, useful for debugging.",
        default=True,
    )
    write_ini: BoolProperty(
        name="Write ini",
        description="Writes the ini file to disk. Disabling this won't refresh the ini file in the mod folder, useful for debugging.",
        default=True,
    )


class DestinationSelector(Operator, ExportHelper):
    """Export single mod based on current frame"""

    bl_idname = "destination.selector"
    bl_label = "Destination"
    filename_ext = "."
    use_filter_folder = True
    filter_glob: StringProperty(
        default=".",
        options={"HIDDEN"},
    )

    def execute(self, context):
        userpath = Path(self.properties.filepath)
        if not userpath.is_dir():
            userpath = userpath.parent
        context.scene.xxmi.destination_path = str(userpath)
        bpy.ops.ed.undo_push(message="XXMI Tools: destination selected")
        return {"FINISHED"}


class DumpSelector(Operator, ExportHelper):
    """Export single mod based on current frame"""

    bl_idname = "dump.selector"
    bl_label = "Dump folder selector"
    filename_ext = "."
    use_filter_folder = True
    filter_glob: StringProperty(
        default=".",
        options={"HIDDEN"},
    )

    def execute(self, context):
        userpath = Path(self.properties.filepath)
        self.properties.filepath = str(userpath.parent)
        context.scene.xxmi.dump_path = str(userpath.parent)
        bpy.ops.ed.undo_push(message="XXMI Tools: dump path selected")
        return {"FINISHED"}


class TemplateSelector(Operator, ExportHelper):
    """Export single mod based on current frame"""

    bl_idname = "template.selector"
    bl_label = "Tempalte file selector"
    filename_ext = ".j2"
    use_filter_folder = True
    filter_glob: StringProperty(
        default="*.j2",
        options={"HIDDEN"},
    )

    def execute(self, context):
        xxmi: XXMIProperties = context.scene.xxmi
        if xxmi.game == "":
            xxmi.template_path = ""
            self.report(
                {"ERROR"},
                "Please select a valid game before chosing a template files.",
            )
            return {"CANCELLED"}

        template_path: Path = Path(self.properties.filepath)
        if template_path.is_dir():
            self.report({"ERROR"}, "Template path must be a file, not a folder")
            return {"CANCELLED"}

        if not template_path.exists():
            addon_path: Path = Path(__file__).parent.parent
            shutil.copy(
                addon_path / "templates" / (GameEnum[xxmi.game] + ".ini.j2"),
                template_path,
            )
            self.report(
                {"INFO"},
                f"Template file for the game {GameEnum[xxmi.game].value} created at: {str(template_path)}",
            )

        xxmi.template_path = self.properties.filepath
        bpy.ops.ed.undo_push(message="XXMI Tools: template path selected")
        return {"FINISHED"}


class ExportAdvancedOperator(Operator):
    """Export operation base class"""

    bl_idname = "xxmi.exportadvanced"
    bl_label = "Export Mod"
    bl_description = "Export mod"
    bl_options = {"REGISTER"}
    operations = []

    def execute(self, context):
        scene = bpy.context.scene
        xxmi: XXMIProperties = scene.xxmi
        if not xxmi.use_custom_template:
            xxmi.template_path = ""
        try:
            if xxmi.game == "":
                self.report(
                    {"ERROR"},
                    "Please select a valid game before continuing.",
                )
                return {"CANCELLED"}
            mod_exporter: ModExporter = ModExporter(
                context=context,
                operator=self,
                dump_path=Path(xxmi.dump_path),
                destination=Path(xxmi.destination_path),
                game=GameEnum[xxmi.game],
                ignore_hidden=xxmi.ignore_hidden,
                only_selected=xxmi.only_selected,
                no_ramps=xxmi.no_ramps,
                copy_textures=xxmi.copy_textures,
                ignore_duplicate_textures=xxmi.ignore_duplicate_textures,
                credit=xxmi.credit,
                outline_optimization=xxmi.outline_optimization,
                apply_modifiers=xxmi.apply_modifiers_and_shapekeys,
                normalize_weights=xxmi.normalize_weights,
                write_buffers=xxmi.write_buffers,
                write_ini=xxmi.write_ini,
                template=Path(xxmi.template_path)
                if xxmi.use_custom_template != ""
                else None,
            )
            mod_exporter.export()
        except Fatal as e:
            self.report({"ERROR"}, str(e))
        return {"FINISHED"}


class ExportAdvancedBatchedOperator(Operator):
    """Export operation base class"""

    bl_idname = "xxmi.exportadvancedbatched"
    bl_label = "Batch export"
    bl_description = "Exports 1 mod per frame of blender timeline as a single mod. Folder names follow the pattern specified in the batch pattern"
    bl_options = {"REGISTER"}
    operations = []

    def invoke(self, context, event):
        scene = bpy.context.scene
        if bpy.app.version < (4, 1, 0):
            return context.window_manager.invoke_confirm(operator=self, event=event)
        return context.window_manager.invoke_confirm(
            operator=self,
            event=event,
            message=f"Exporting {scene.frame_end + 1 - scene.frame_start} copies of the mod. This may take a while. Continue?",
            title="Batch export",
            icon="WARNING",
            confirm_text="Continue",
        )

    def execute(self, context):
        scene = bpy.context.scene
        xxmi: XXMIProperties = scene.xxmi
        start_time = time.time()
        base_dir = Path(xxmi.destination_path)
        wildcards = ("#####", "####", "###", "##", "#")
        try:
            for frame in range(scene.frame_start, scene.frame_end + 1):
                context.scene.frame_set(frame)
                for w in wildcards:
                    frame_folder = Path(
                        xxmi.batch_pattern.replace(w, str(frame).zfill(len(w)))
                    )
                    if frame_folder != xxmi.batch_pattern:
                        break
                else:
                    self.report(
                        {"ERROR"},
                        "Batch pattern must contain any number of # wildcard characters for the frame number to be written into it. Example name_### -> name_001",
                    )
                    return False
                xxmi.destination_path = base_dir / frame_folder
                bpy.ops.xxmi.exportadvanced()
                print(
                    f"Exported frame {frame + 1 - scene.frame_start}/{scene.frame_end + 1 - scene.frame_start}"
                )
            print(f"Batch export took {time.time() - start_time} seconds")
        except Fatal as e:
            self.report({"ERROR"}, str(e))
        xxmi.destination_path = base_dir
        return {"FINISHED"}


def write_fmt_file(f, vb: VertexBufferGroup, ib: IndexBuffer, strides: list[int]):
    for vbuf_idx, stride in strides.items():
        if vbuf_idx.isnumeric():
            f.write("vb%s stride: %i\n" % (vbuf_idx, stride))
        else:
            f.write("stride: %i\n" % stride)
    f.write("topology: %s\n" % vb.topology)
    if ib is not None:
        f.write("format: %s\n" % ib.format)
    f.write(vb.layout.to_string())


def write_ini_file(
    f,
    vb: VertexBufferGroup,
    vb_path,
    ib: IndexBuffer,
    ib_path,
    strides: list[int],
    obj: Object,
    topology: str,
):
    backup = True
    # topology='trianglestrip' # Testing
    bind_section = ""
    backup_section = ""
    restore_section = ""
    resource_section = ""
    resource_bak_section = ""

    draw_section = "handling = skip\n"
    if ib is not None:
        draw_section += "drawindexed = auto\n"
    else:
        draw_section += "draw = auto\n"

    if ib is not None:
        bind_section += "ib = ResourceIB\n"
        resource_section += textwrap.dedent("""
            [ResourceIB]
            type = buffer
            format = {}
            filename = {}
            """).format(ib.format, ib_path)
        if backup:
            resource_bak_section += "[ResourceBakIB]\n"
            backup_section += "ResourceBakIB = ref ib\n"
            restore_section += "ib = ResourceBakIB\n"

    for vbuf_idx, stride in strides.items():
        bind_section += "vb{0} = ResourceVB{0}\n".format(vbuf_idx or 0)
        resource_section += textwrap.dedent("""
            [ResourceVB{}]
            type = buffer
            stride = {}
            filename = {}
            """).format(vbuf_idx, stride, vb_path + vbuf_idx)
        if backup:
            resource_bak_section += "[ResourceBakVB{0}]\n".format(vbuf_idx or 0)
            backup_section += "ResourceBakVB{0} = ref vb{0}\n".format(vbuf_idx or 0)
            restore_section += "vb{0} = ResourceBakVB{0}\n".format(vbuf_idx or 0)

    # FIXME: Maybe split this into several ini files that the user may or may
    # not choose to generate? One that just lists resources, a second that
    # lists the TextureOverrides to replace draw calls, and a third with the
    # ShaderOverride sections (or a ShaderRegex for foolproof replacements)...?
    f.write(
        textwrap.dedent("""
            ; Automatically generated file, be careful not to overwrite if you
            ; make any manual changes

            ; Please note - it is not recommended to place the [ShaderOverride]
            ; here, as you only want checktextureoverride executed once per
            ; draw call, so it's better to have all the shaders listed in a
            ; common file instead to avoid doubling up and to allow common code
            ; to enable/disable the mods, backup/restore buffers, etc. Plus you
            ; may need to locate additional shaders to take care of shadows or
            ; other render passes. But if you understand what you are doing and
            ; need a quick 'n' dirty way to enable the reinjection, fill this in
            ; and uncomment it:
            ;[ShaderOverride{suffix}]
            ;hash = FILL ME IN...
            ;checktextureoverride = vb0

            [TextureOverride{suffix}]
            ;hash = FILL ME IN...
            """)
        .lstrip()
        .format(
            suffix="",
        )
    )
    if ib is not None and "3DMigoto:FirstIndex" in obj:
        f.write("match_first_index = {}\n".format(obj["3DMigoto:FirstIndex"]))
    elif ib is None and "3DMigoto:FirstVertex" in obj:
        f.write("match_first_vertex = {}\n".format(obj["3DMigoto:FirstVertex"]))

    if backup:
        f.write(backup_section)

    f.write(bind_section)

    if topology == "trianglestrip":
        f.write("run = CustomShaderOverrideTopology\n")
    else:
        f.write(draw_section)

    if backup:
        f.write(restore_section)

    if topology == "trianglestrip":
        f.write(
            textwrap.dedent("""
            [CustomShaderOverrideTopology]
            topology = triangle_list
            """)
            + draw_section
        )

    if backup:
        f.write("\n" + resource_bak_section)

    f.write(resource_section)


def blender_vertex_to_3dmigoto_vertex(
    mesh: Mesh,
    obj: Object,
    blender_loop_vertex,
    layout,
    texcoords,
    blender_vertex,
    translate_normal,
    translate_tangent,
    export_outline=None,
):
    if blender_loop_vertex is not None:
        blender_vertex = mesh.vertices[blender_loop_vertex.vertex_index]
    vertex = {}
    blp_normal = list(blender_loop_vertex.normal)

    # TODO: Warn if vertex is in too many vertex groups for this layout,
    # ignoring groups with weight=0.0
    vertex_groups = sorted(blender_vertex.groups, key=lambda x: x.weight, reverse=True)

    for elem in layout:
        if elem.InputSlotClass != "per-vertex" or elem.reused_offset:
            continue

        semantic_translations = layout.get_semantic_remap()
        translated_elem_name, translated_elem_index = semantic_translations.get(
            elem.name, (elem.name, elem.SemanticIndex)
        )

        # Some games don't follow the official DirectX UPPERCASE semantic naming convention:
        translated_elem_name = translated_elem_name.upper()

        if translated_elem_name == "POSITION":
            if "POSITION.w" in custom_attributes_float(mesh):
                vertex[elem.name] = list(blender_vertex.undeformed_co) + [
                    custom_attributes_float(mesh)["POSITION.w"]
                    .data[blender_vertex.index]
                    .value
                ]
            else:
                vertex[elem.name] = elem.pad(list(blender_vertex.undeformed_co), 1.0)
        elif translated_elem_name.startswith("COLOR"):
            if elem.name in mesh.vertex_colors:
                vertex[elem.name] = elem.clip(
                    list(
                        mesh.vertex_colors[elem.name]
                        .data[blender_loop_vertex.index]
                        .color
                    )
                )
            else:
                vertex[elem.name] = list(
                    mesh.vertex_colors[elem.name + ".RGB"]
                    .data[blender_loop_vertex.index]
                    .color
                )[:3] + [
                    mesh.vertex_colors[elem.name + ".A"]
                    .data[blender_loop_vertex.index]
                    .color[0]
                ]
        elif translated_elem_name == "NORMAL":
            if "NORMAL.w" in custom_attributes_float(mesh):
                vertex[elem.name] = list(
                    map(translate_normal, blender_loop_vertex.normal)
                ) + [
                    custom_attributes_float(mesh)["NORMAL.w"]
                    .data[blender_vertex.index]
                    .value
                ]
            elif blender_loop_vertex:
                vertex[elem.name] = elem.pad(
                    list(map(translate_normal, blender_loop_vertex.normal)), 0.0
                )
            else:
                # XXX: point list topology, these normals are probably going to be pretty poor, but at least it's something to export
                vertex[elem.name] = elem.pad(
                    list(map(translate_normal, blender_vertex.normal)), 0.0
                )
        elif translated_elem_name.startswith("TANGENT"):
            if export_outline:
                # Genshin optimized outlines
                vertex[elem.name] = elem.pad(
                    list(
                        map(
                            translate_tangent,
                            export_outline.get(
                                blender_loop_vertex.vertex_index, blp_normal
                            ),
                        )
                    ),
                    blender_loop_vertex.bitangent_sign,
                )
            # DOAXVV has +1/-1 in the 4th component. Not positive what this is,
            # but guessing maybe the bitangent sign? Not even sure it is used...
            # FIXME: Other games
            elif blender_loop_vertex:
                vertex[elem.name] = elem.pad(
                    list(map(translate_tangent, blender_loop_vertex.tangent)),
                    blender_loop_vertex.bitangent_sign,
                )
            else:
                # XXX Blender doesn't save tangents outside of loops, so unless
                # we save these somewhere custom when importing they are
                # effectively lost. We could potentially calculate a tangent
                # from blender_vertex.normal, but there is probably little
                # point given that normal will also likely be garbage since it
                # wasn't imported from the mesh.
                pass
        elif translated_elem_name.startswith("BINORMAL"):
            # Some DOA6 meshes (skirts) use BINORMAL, but I'm not certain it is
            # actually the binormal. These meshes are weird though, since they
            # use 4 dimensional positions and normals, so they aren't something
            # we can really deal with at all. Therefore, the below is untested,
            # FIXME: So find a mesh where this is actually the binormal,
            # uncomment the below code and test.
            # normal = blender_loop_vertex.normal
            # tangent = blender_loop_vertex.tangent
            # binormal = numpy.cross(normal, tangent)
            # XXX: Does the binormal need to be normalised to a unit vector?
            # binormal = binormal / numpy.linalg.norm(binormal)
            # vertex[elem.name] = elem.pad(list(map(translate_binormal, binormal)), 0.0)
            pass
        elif translated_elem_name.startswith("BLENDINDICES"):
            i = translated_elem_index * 4
            vertex[elem.name] = elem.pad([x.group for x in vertex_groups[i : i + 4]], 0)
        elif translated_elem_name.startswith("BLENDWEIGHT"):
            # TODO: Warn if vertex is in too many vertex groups for this layout
            i = translated_elem_index * 4
            vertex[elem.name] = elem.pad(
                [x.weight for x in vertex_groups[i : i + 4]], 0.0
            )
        elif translated_elem_name.startswith("TEXCOORD") and elem.is_float():
            uvs = []
            for uv_name in ("%s.xy" % elem.remapped_name, "%s.zw" % elem.remapped_name):
                if uv_name in texcoords:
                    uvs += list(texcoords[uv_name][blender_loop_vertex.index])
            # Handle 1D + 3D TEXCOORDs. Order is important - 1D TEXCOORDs won't
            # match anything in above loop so only .x below, 3D TEXCOORDS will
            # have processed .xy part above, and .z part below
            for uv_name in ("%s.x" % elem.remapped_name, "%s.z" % elem.remapped_name):
                if uv_name in texcoords:
                    uvs += [texcoords[uv_name][blender_loop_vertex.index].x]
            vertex[elem.name] = uvs
        else:
            # Unhandled semantics are saved in vertex layers
            data = []
            for component in "xyzw":
                layer_name = "%s.%s" % (elem.name, component)
                if layer_name in custom_attributes_int(mesh):
                    data.append(
                        custom_attributes_int(mesh)[layer_name]
                        .data[blender_vertex.index]
                        .value
                    )
                elif layer_name in custom_attributes_float(mesh):
                    data.append(
                        custom_attributes_float(mesh)[layer_name]
                        .data[blender_vertex.index]
                        .value
                    )
            if data:
                # print('Retrieved unhandled semantic %s %s from vertex layer' % (elem.name, elem.Format), data)
                vertex[elem.name] = data

        if elem.name not in vertex:
            print("NOTICE: Unhandled vertex element: %s" % elem.name)
        # else:
        #    print('%s: %s' % (elem.name, repr(vertex[elem.name])))

    return vertex


def export_3dmigoto(
    operator: Operator, context: Context, vb_path, ib_path, fmt_path, ini_path
):
    obj = context.object
    if obj is None:
        raise Fatal("No object selected")

    strides = {
        x[11:-6]: obj[x]
        for x in obj.keys()
        if x.startswith("3DMigoto:VB") and x.endswith("Stride")
    }
    layout = InputLayout(obj["3DMigoto:VBLayout"])
    topology = "trianglelist"
    if "3DMigoto:Topology" in obj:
        topology = obj["3DMigoto:Topology"]
        if topology == "trianglestrip":
            operator.report(
                {"WARNING"},
                "trianglestrip topology not supported for export, and has been converted to trianglelist. Override draw call topology using a [CustomShader] section with topology=triangle_list",
            )
            topology = "trianglelist"
    if hasattr(context, "evaluated_depsgraph_get"):  # 2.80
        mesh = obj.evaluated_get(context.evaluated_depsgraph_get()).to_mesh()
    else:  # 2.79
        mesh = obj.to_mesh(context.scene, True, "PREVIEW", calc_tessface=False)
    mesh_triangulate(mesh)

    try:
        ib_format = obj["3DMigoto:IBFormat"]
    except KeyError:
        ib = None
    else:
        ib = IndexBuffer(ib_format)

    # Calculates tangents and makes loop normals valid (still with our
    # custom normal data from import time):
    try:
        mesh.calc_tangents()
    except RuntimeError as e:
        operator.report(
            {"WARNING"},
            "Tangent calculation failed, the exported mesh may have bad normals/tangents/lighting. Original {}".format(
                str(e)
            ),
        )

    texcoord_layers = {}
    for uv_layer in mesh.uv_layers:
        texcoords = {}

        try:
            flip_texcoord_v = obj["3DMigoto:" + uv_layer.name]["flip_v"]
            if flip_texcoord_v:
                flip_uv = lambda uv: (uv[0], 1.0 - uv[1])
            else:
                flip_uv = lambda uv: uv
        except KeyError:
            flip_uv = lambda uv: uv

        for loop in mesh.loops:
            uv = flip_uv(uv_layer.data[loop.index].uv)
            texcoords[loop.index] = uv
        texcoord_layers[uv_layer.name] = texcoords

    translate_normal = normal_export_translation(
        layout, Semantic.Normal, operator.flip_normal
    )
    translate_tangent = normal_export_translation(
        layout, Semantic.Tangent, operator.flip_tangent
    )

    # Blender's vertices have unique positions, but may have multiple
    # normals, tangents, UV coordinates, etc - these are stored in the
    # loops. To export back to DX we need these combined together such that
    # a vertex is a unique set of all attributes, but we don't want to
    # completely blow this out - we still want to reuse identical vertices
    # via the index buffer. There might be a convenience function in
    # Blender to do this, but it's easy enough to do this ourselves
    indexed_vertices = collections.OrderedDict()
    vb = VertexBufferGroup(layout=layout, topology=topology)
    vb.flag_invalid_semantics()
    if vb.topology == "trianglelist":
        for poly in mesh.polygons:
            face = []
            for blender_lvertex in mesh.loops[
                poly.loop_start : poly.loop_start + poly.loop_total
            ]:
                vertex = blender_vertex_to_3dmigoto_vertex(
                    mesh,
                    obj,
                    blender_lvertex,
                    layout,
                    texcoord_layers,
                    None,
                    translate_normal,
                    translate_tangent,
                )
                if ib is not None:
                    face.append(
                        indexed_vertices.setdefault(
                            HashableVertex(vertex), len(indexed_vertices)
                        )
                    )
                else:
                    if operator.flip_winding:
                        raise Fatal(
                            "Flipping winding order without index buffer not implemented"
                        )
                    vb.append(vertex)
            if ib is not None:
                if operator.flip_winding:
                    face.reverse()
                ib.append(face)

        if ib is not None:
            for vertex in indexed_vertices:
                vb.append(vertex)
    elif vb.topology == "pointlist":
        for index, blender_vertex in enumerate(mesh.vertices):
            vb.append(
                blender_vertex_to_3dmigoto_vertex(
                    mesh,
                    obj,
                    None,
                    layout,
                    texcoord_layers,
                    blender_vertex,
                    translate_normal,
                    translate_tangent,
                )
            )
            if ib is not None:
                ib.append((index,))
    else:
        raise Fatal('topology "%s" is not supported for export' % vb.topology)

    vgmaps = {
        k[15:]: keys_to_ints(v)
        for k, v in obj.items()
        if k.startswith("3DMigoto:VGMap:")
    }

    if "" not in vgmaps:
        vb.write(vb_path, strides, operator=operator)

    for suffix, vgmap in vgmaps.items():
        ib_path = vb_path
        if suffix:
            ib_path = f"{vb_path.parent / vb_path.stem}-{suffix}{vb_path.suffix}"
        vgmap_path = (ib_path.parent / ib_path.stem) + ".vgmap"
        print("Exporting %s..." % ib_path)
        vb.remap_blendindices(obj, vgmap)
        vb.write(ib_path, strides, operator=operator)
        vb.revert_blendindices_remap()
        sorted_vgmap = collections.OrderedDict(
            sorted(vgmap.items(), key=lambda x: x[1])
        )
        json.dump(sorted_vgmap, open(vgmap_path, "w"), indent=2)

    if ib is not None:
        ib.write(open(ib_path, "wb"), operator=operator)

    # Write format reference file
    write_fmt_file(open(fmt_path, "w"), vb, ib, strides)

    # Not ready yet
    # if ini_path:
    #    write_ini_file(open(ini_path, 'w'), vb, vb_path, ib, ib_path, strides, obj, orig_topology)


def register():
    """Register all classes"""
    bpy.types.Scene.xxmi = PointerProperty(type=XXMIProperties)


def unregister():
    """Unregister all classes"""
    del bpy.types.Scene.xxmi
