import bpy
from bpy.props import BoolProperty, IntProperty, StringProperty
from bpy.types import Operator, AddonPreferences
from bpy_extras.io_utils import ImportHelper, orientation_helper

from .. import __name__ as package_name
from .. import addon_updater_ops
from .datahandling import (
    Fatal,
    apply_vgmap,
    import_pose,
    merge_armatures,
    update_vgmap,
)

from .datastructures import IOOBJOrientationHelper


class ApplyVGMap(Operator, ImportHelper):
    """Apply vertex group map to the selected object"""

    bl_idname = "mesh.migoto_vertex_group_map"
    bl_label = "Apply 3DMigoto vgmap"
    bl_options = {"UNDO"}

    filename_ext = ".vgmap"
    filter_glob: StringProperty(
        default="*.vgmap",
        options={"HIDDEN"},
    )

    # commit: BoolProperty(
    #        name="Commit to current mesh",
    #        description="Directly alters the vertex groups of the current mesh, rather than performing the mapping at export time",
    #        default=False,
    #        )

    rename: BoolProperty(
        name="Rename existing vertex groups",
        description="Rename existing vertex groups to match the vgmap file",
        default=True,
    )

    cleanup: BoolProperty(
        name="Remove non-listed vertex groups",
        description="Remove any existing vertex groups that are not listed in the vgmap file",
        default=False,
    )

    reverse: BoolProperty(
        name="Swap from & to",
        description="Switch the order of the vertex group map - if this mesh is the 'to' and you want to use the bones in the 'from'",
        default=False,
    )

    suffix: StringProperty(
        name="Suffix",
        description="Suffix to add to the vertex buffer filename when exporting, for bulk exports of a single mesh with multiple distinct vertex group maps",
        default="",
    )

    def invoke(self, context, event):
        self.suffix = ""
        return ImportHelper.invoke(self, context, event)

    def execute(self, context):
        try:
            keywords = self.as_keywords(ignore=("filter_glob",))
            apply_vgmap(self, context, **keywords)
        except Fatal as e:
            self.report({"ERROR"}, str(e))
        return {"FINISHED"}


class UpdateVGMap(Operator):
    """Assign new 3DMigoto vertex groups"""

    bl_idname = "mesh.update_migoto_vertex_group_map"
    bl_label = "Assign new 3DMigoto vertex groups"
    bl_options = {"UNDO"}

    vg_step: bpy.props.IntProperty(
        name="Vertex group step",
        description="If used vertex groups are 0,1,2,3,etc specify 1. If they are 0,3,6,9,12,etc specify 3",
        default=1,
        min=1,
    )

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):
        try:
            keywords = self.as_keywords()
            update_vgmap(self, context, **keywords)
        except Fatal as e:
            self.report({"ERROR"}, str(e))
        return {"FINISHED"}


@orientation_helper(axis_forward="-Z", axis_up="Y")
class Import3DMigotoPose(Operator, ImportHelper, IOOBJOrientationHelper):
    """Import a pose from a 3DMigoto constant buffer dump"""

    bl_idname = "armature.migoto_pose"
    bl_label = "Import 3DMigoto Pose"
    bl_options = {"UNDO"}

    filename_ext = ".txt"
    filter_glob: StringProperty(
        default="*.txt",
        options={"HIDDEN"},
    )

    limit_bones_to_vertex_groups: BoolProperty(
        name="Limit Bones to Vertex Groups",
        description="Limits the maximum number of bones imported to the number of vertex groups of the active object",
        default=True,
    )

    pose_cb_off: bpy.props.IntVectorProperty(
        name="Bone CB range",
        description="Indicate start and end offsets (in multiples of 4 component values) to find the matrices in the Bone CB",
        default=[0, 0],
        size=2,
        min=0,
    )

    pose_cb_step: bpy.props.IntProperty(
        name="Vertex group step",
        description="If used vertex groups are 0,1,2,3,etc specify 1. If they are 0,3,6,9,12,etc specify 3",
        default=1,
        min=1,
    )

    def execute(self, context):
        try:
            keywords = self.as_keywords(ignore=("filter_glob",))
            import_pose(self, context, **keywords)
        except Fatal as e:
            self.report({"ERROR"}, str(e))
        return {"FINISHED"}


class Merge3DMigotoPose(Operator):
    """Merge identically posed bones of related armatures into one"""

    bl_idname = "armature.merge_pose"
    bl_label = "Merge 3DMigoto Poses"
    bl_options = {"UNDO"}

    def execute(self, context):
        try:
            merge_armatures(self, context)
        except Fatal as e:
            self.report({"ERROR"}, str(e))
        return {"FINISHED"}


class DeleteNonNumericVertexGroups(Operator):
    """Remove vertex groups with non-numeric names"""

    bl_idname = "vertex_groups.delete_non_numeric"
    bl_label = "Remove non-numeric vertex groups"
    bl_options = {"UNDO"}

    def execute(self, context):
        try:
            for obj in context.selected_objects:
                for vg in reversed(obj.vertex_groups):
                    if vg.name.isdecimal():
                        continue
                    print("Removing vertex group", vg.name)
                    obj.vertex_groups.remove(vg)
        except Fatal as e:
            self.report({"ERROR"}, str(e))
        return {"FINISHED"}


class Preferences(AddonPreferences):
    """Preferences updater"""

    bl_idname = package_name
    # Addon updater preferences.

    auto_check_update: BoolProperty(
        name="Auto-check for Update",
        description="If enabled, auto-check for updates using an interval",
        default=False,
    )

    updater_interval_months: IntProperty(
        name="Months",
        description="Number of months between checking for updates",
        default=0,
        min=0,
    )

    updater_interval_days: IntProperty(
        name="Days",
        description="Number of days between checking for updates",
        default=7,
        min=0,
        max=31,
    )

    updater_interval_hours: IntProperty(
        name="Hours",
        description="Number of hours between checking for updates",
        default=0,
        min=0,
        max=23,
    )

    updater_interval_minutes: IntProperty(
        name="Minutes",
        description="Number of minutes between checking for updates",
        default=0,
        min=0,
        max=59,
    )

    def draw(self, context):
        layout = self.layout
        print(addon_updater_ops.get_user_preferences(context))
        # Works best if a column, or even just self.layout.
        mainrow = layout.row()
        _ = mainrow.column()
        # Updater draw function, could also pass in col as third arg.
        addon_updater_ops.update_settings_ui(self, context)

        # Alternate draw function, which is more condensed and can be
        # placed within an existing draw function. Only contains:
        #   1) check for update/update now buttons
        #   2) toggle for auto-check (interval will be equal to what is set above)
        # addon_updater_ops.update_settings_ui_condensed(self, context, col)

        # Adding another column to help show the above condensed ui as one column
        # col = mainrow.column()
        # col.scale_y = 2
        # ops = col.operator("wm.url_open","Open webpage ")
        # ops.url=addon_updater_ops.updater.website


class VGROUP_SN_merge(bpy.types.Operator):
    bl_description = "Merge the vertex groups with shared name"
    bl_idname = "mesh.merge_shared_name_vgs"
    bl_label = "Merge shared name Vertex Groups"
    bl_options = {"UNDO"}
    selected = []

    @classmethod
    def poll(cls, context):
        return (
            context.object
            and context.object.type == "MESH"
            and context.object.vertex_groups
        )

    def invoke(self, context, event):
        if event.alt:
            self.selected = [*{context.object, *context.selected_objects}]
        return self.execute(context)

    def execute(self, context):
        if not self.selected:
            self.selected = [context.object]

        for ob in self.selected:
            if not getattr(ob, "type", None) == "MESH":
                continue

            # myyy
            vgroup_names = [x.name.split(".")[0] for x in ob.vertex_groups]
            for vname in vgroup_names:
                relevant = [
                    x.name
                    for x in ob.vertex_groups
                    if x.name.split(".")[0] == f"{vname}"
                ]
                if relevant:
                    vgroup = ob.vertex_groups.new(name=f"x{vname}")
                    for vert_id, vert in enumerate(ob.data.vertices):
                        available_groups = [
                            v_group_elem.group for v_group_elem in vert.groups
                        ]

                        combined = 0
                        for v in relevant:
                            if ob.vertex_groups[v].index in available_groups:
                                combined += ob.vertex_groups[v].weight(vert_id)
                                if combined > 0:
                                    vgroup.add([vert_id], combined, "ADD")
                    for vg in [
                        x
                        for x in ob.vertex_groups
                        if x.name.split(".")[0] == f"{vname}"
                    ]:
                        ob.vertex_groups.remove(vg)
                    for vg in ob.vertex_groups:
                        if vg.name[0].lower() == "x":
                            vg.name = vg.name[1:]

            bpy.ops.object.vertex_group_sort()

        return {"FINISHED"}


class VGROUP_SN_merge_ONE(bpy.types.Operator):
    bl_description = "Merge the vertex groups with shared name with the active VG"
    bl_idname = "mesh.merge_shared_name_vgs_one"
    bl_label = "Merge shared name VGs with the active Vertex Group"
    bl_options = {"UNDO"}
    selected = []

    @classmethod
    def poll(cls, context):
        return (
            context.object
            and context.object.type == "MESH"
            and context.object.vertex_groups
        )

    def invoke(self, context, event):
        if event.alt:
            self.selected = [*{context.object, *context.selected_objects}]
        return self.execute(context)

    def execute(self, context):
        if not self.selected:
            self.selected = [context.object]
        for ob in self.selected:
            if not getattr(ob, "type", None) == "MESH":
                continue

            # myyy
            vname = ob.vertex_groups.active.name
            if "." in vname:
                vname = vname.split(".")[0]
            vgroup_names = [x.name.split(".")[0] for x in ob.vertex_groups]
            relevant = [
                x.name for x in ob.vertex_groups if x.name.split(".")[0] == f"{vname}"
            ]

            if relevant:
                vgroup = ob.vertex_groups.new(name=f"x{vname}")
                for vert_id, vert in enumerate(ob.data.vertices):
                    available_groups = [
                        v_group_elem.group for v_group_elem in vert.groups
                    ]
                    combined = 0
                    for v in relevant:
                        if ob.vertex_groups[v].index in available_groups:
                            combined += ob.vertex_groups[v].weight(vert_id)
                            if combined > 0:
                                vgroup.add([vert_id], combined, "ADD")
                for vg in [
                    x for x in ob.vertex_groups if x.name.split(".")[0] == f"{vname}"
                ]:
                    ob.vertex_groups.remove(vg)
                for vg in ob.vertex_groups:
                    if vg.name[0].lower() == "x":
                        vg.name = vg.name[1:]

            bpy.ops.object.vertex_group_sort()

        return {"FINISHED"}


class VGROUP_SN_fill(bpy.types.Operator):
    bl_description = "Fill VGs from the lowest to highest existing one "
    bl_idname = "mesh.fill_vg"
    bl_label = "Fill gaps in Vertex Groups"
    bl_options = {"UNDO"}

    selected = []

    @classmethod
    def poll(cls, context):
        return (
            context.object
            and context.object.type == "MESH"
            and context.object.vertex_groups
        )

    def invoke(self, context, event):
        if event.alt:
            self.selected = [*{context.object, *context.selected_objects}]
        return self.execute(context)

    def execute(self, context):
        if not self.selected:
            self.selected = [context.object]
        for ob in self.selected:
            if not getattr(ob, "type", None) == "MESH":
                continue

            # myyy
            largest = 0
            for vg in ob.vertex_groups:
                try:
                    if int(vg.name.split(".")[0]) > largest:
                        largest = int(vg.name.split(".")[0])
                except ValueError:
                    print("Vertex group not named as integer, skipping")

            missing = set([f"{i}" for i in range(largest + 1)]) - set(
                [x.name.split(".")[0] for x in ob.vertex_groups]
            )
            for number in missing:
                ob.vertex_groups.new(name=f"{number}")

            bpy.ops.object.vertex_group_sort()

        return {"FINISHED"}


class VGROUP_SN_remove(bpy.types.Operator):
    bl_description = (
        "Remove unused VGs checks ONLY the weight paints, not shapekeys and others"
    )
    bl_idname = "mesh.remove_vg"
    bl_label = "Remove unused Vertex Groups"
    bl_options = {"UNDO"}

    selected = []

    @classmethod
    def poll(cls, context):
        return (
            context.object
            and context.object.type == "MESH"
            and context.object.vertex_groups
        )

    def invoke(self, context, event):
        if event.alt:
            self.selected = [*{context.object, *context.selected_objects}]
        return self.execute(context)

    def execute(self, context):
        if not self.selected:
            self.selected = [context.object]
        for ob in self.selected:
            if not getattr(ob, "type", None) == "MESH":
                continue

            # myyy
            used_groups = set()

            # Used groups from weight paint
            for id, vert in enumerate(ob.data.vertices):
                for vg in vert.groups:
                    vgi = vg.group
                    used_groups.add(vgi)
            # removing
            for vg in list(reversed(ob.vertex_groups)):
                if vg.index not in used_groups:
                    ob.vertex_groups.remove(vg)

        # bpy.ops.object.vertex_group_sort()

        return {"FINISHED"}


class CLEAN_UV_NAMES(bpy.types.Operator):
    bl_description = (
        "Ensures the format TEXCOORD[n].xy for UV names, as expected by 3DMigoto"
    )
    bl_idname = "mesh.clean_uv_names"
    bl_label = "Clean UV Names"
    bl_options = {"UNDO"}

    selected = []

    def invoke(self, context, event):
        if event.alt:
            self.selected = [*{context.object, *context.selected_objects}]
        return self.execute(context)

    def execute(self, context):
        if not self.selected:
            self.selected = [context.object]
        for ob in self.selected:
            if not getattr(ob, "type", None) == "MESH":
                continue

            for id, uv_layer in enumerate(ob.data.uv_layers):
                if id == 0:
                    expected_name = "TEXCOORD.xy"
                else:
                    expected_name = f"TEXCOORD{id}.xy"
                if uv_layer.name != expected_name:
                    uv_layer.name = expected_name
            return {"FINISHED"}


class RESET_VERTEX_COLORS(bpy.types.Operator):
    bl_description = "Resets vertex colors to a color selected by the user"
    bl_idname = "mesh.reset_vertex_colors"
    bl_label = "Reset Vertex Colors"
    bl_options = {"UNDO"}

    color: bpy.props.FloatVectorProperty(
        name="Color",
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 0.215861, 0.215861, 0.501961),  # RGBA default color
        description="Select the reset vertex color",
    )
    data_type: bpy.props.EnumProperty(
        name="Data Type",
        items=[
            ("BYTE_COLOR", "Byte Color", "Use byte color format (0-255)"),
            ("FLOAT_COLOR", "Float Color", "Use float color format (0.0-1.0)"),
        ],
        default="BYTE_COLOR",
        description="Choose the data type for vertex colors",
    )
    domain: bpy.props.EnumProperty(
        name="Domain",
        items=[
            ("POINT", "Point", "Apply vertex colors to points"),
            ("CORNER", "Corner", "Apply vertex colors to corners"),
            ("FACE", "Face", "Apply vertex colors to faces"),
        ],
        default="CORNER",
        description="Choose the domain for vertex colors",
    )

    def invoke(self, context, event):
        # Open a dialog with the color picker UI.
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "color", text="Reset Color")
        layout.prop(self, "data_type", text="Data Type")
        layout.prop(self, "domain", text="Domain")

    def execute(self, context):
        for obj in context.selected_objects:
            if obj.type != "MESH":
                continue

            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            context.view_layer.objects.active = obj
            if "COLOR" in obj.data.color_attributes:
                obj.data.color_attributes.remove(obj.data.color_attributes["COLOR"])
            bpy.ops.geometry.color_attribute_add(
                name="COLOR",
                domain="CORNER",
                data_type="BYTE_COLOR",
                color=self.color,
            )

        return {"FINISHED"}


def draw_menu(self, context):
    layout = self.layout
    layout.separator()
    layout.operator(VGROUP_SN_merge.bl_idname, icon="BRUSH_GRAB")
    layout.operator(VGROUP_SN_merge_ONE.bl_idname, icon="BRUSH_INFLATE")
    layout.operator(VGROUP_SN_fill.bl_idname, icon="BRUSH_FILL")
    layout.operator(VGROUP_SN_remove.bl_idname, icon="GPBRUSH_ERASE_STROKE")


def register():
    bpy.types.MESH_MT_vertex_group_context_menu.append(draw_menu)
    bpy.types.VIEW3D_MT_vertex_group.append(draw_menu)


def unregister():
    bpy.types.VIEW3D_MT_vertex_group.remove(draw_menu)
    bpy.types.MESH_MT_vertex_group_context_menu.remove(draw_menu)
