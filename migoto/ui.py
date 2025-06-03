import bpy
from bpy.types import Panel, UIList, Menu, UILayout
import addon_utils
from bl_ui.generic_ui_list import draw_ui_list
from .operators import (
    Import3DMigotoPose,
    ApplyVGMap,
)
from .import_ops import (
    ClearSemanticRemapList,
    PrefillSemanticRemapList,
    Import3DMigotoFrameAnalysis,
    Import3DMigotoRaw,
)
from .export_ops import (
    Export3DMigoto,
    Export3DMigotoXXMI,
)
from .. import addon_updater_ops
from .export_ops import XXMIProperties


class MIGOTO_UL_semantic_remap_list(UIList):
    def draw_item(
        self, context, layout, data, item, icon, active_data, active_propname
    ):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            layout.prop(item, "semantic_from", text="", emboss=False, icon_value=icon)
            if item.InputSlotClass == "per-instance":
                layout.label(text="Instanced Data")
                layout.enabled = False
            elif item.valid is False:
                layout.label(text="INVALID")
                layout.enabled = False
            else:
                layout.prop(item, "semantic_to", text="", emboss=False, icon_value=icon)
        elif self.layout_type == "GRID":
            # Doco says we must implement this layout type, but I don't see
            # that it would be particularly useful, and not sure if we actually
            # expect the list to render with this type in practice. Untested.
            layout.alignment = "CENTER"
            layout.label(text="", icon_value=icon)


class MIGOTO_MT_semantic_remap_menu(Menu):
    bl_label = "Semantic Remap Options"

    def draw(self, context):
        layout = self.layout

        layout.operator(ClearSemanticRemapList.bl_idname)
        layout.operator(PrefillSemanticRemapList.bl_idname)


class MigotoImportOptionsPanelBase(object):
    bl_space_type = "FILE_BROWSER"
    bl_region_type = "TOOL_PROPS"
    bl_parent_id = "FILE_PT_operator"

    @classmethod
    def poll(cls, context):
        operator = context.space_data.active_operator
        return operator.bl_idname == "IMPORT_MESH_OT_migoto_frame_analysis"

    def draw(self, context):
        self.layout.use_property_split = True
        self.layout.use_property_decorate = False


class MIGOTO_PT_ImportFrameAnalysisMainPanel(MigotoImportOptionsPanelBase, Panel):
    bl_label = ""
    bl_options = {"HIDE_HEADER"}
    bl_order = 0

    def draw(self, context):
        MigotoImportOptionsPanelBase.draw(self, context)
        operator = context.space_data.active_operator
        self.layout.prop(operator, "flip_texcoord_v")
        self.layout.prop(operator, "flip_winding")
        self.layout.prop(operator, "flip_normal")
        self.layout.prop(operator, "flip_mesh")


class MIGOTO_PT_ImportFrameAnalysisRelatedFilesPanel(
    MigotoImportOptionsPanelBase, Panel
):
    bl_label = ""
    bl_options = {"HIDE_HEADER"}
    bl_order = 1

    def draw(self, context):
        MigotoImportOptionsPanelBase.draw(self, context)
        operator = context.space_data.active_operator
        self.layout.enabled = not operator.load_buf
        self.layout.prop(operator, "load_related")
        self.layout.prop(operator, "load_related_so_vb")
        self.layout.prop(operator, "merge_meshes")


class MIGOTO_PT_ImportFrameAnalysisBufFilesPanel(MigotoImportOptionsPanelBase, Panel):
    bl_label = "Load .buf files instead"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 2

    def draw_header(self, context):
        operator = context.space_data.active_operator
        self.layout.prop(operator, "load_buf", text="")

    def draw(self, context):
        MigotoImportOptionsPanelBase.draw(self, context)
        operator = context.space_data.active_operator
        self.layout.enabled = operator.load_buf
        self.layout.prop(operator, "load_buf_limit_range")


class MIGOTO_PT_ImportFrameAnalysisBonePanel(MigotoImportOptionsPanelBase, Panel):
    bl_label = ""
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 3

    def draw_header(self, context):
        operator = context.space_data.active_operator
        self.layout.prop(operator, "pose_cb")

    def draw(self, context):
        MigotoImportOptionsPanelBase.draw(self, context)
        operator = context.space_data.active_operator
        self.layout.prop(operator, "pose_cb_off")
        self.layout.prop(operator, "pose_cb_step")


class MIGOTO_PT_ImportFrameAnalysisRemapSemanticsPanel(
    MigotoImportOptionsPanelBase, Panel
):
    bl_label = "Semantic Remap"
    # bl_options = {'DEFAULT_CLOSED'}
    bl_order = 4

    def draw(self, context):
        MigotoImportOptionsPanelBase.draw(self, context)
        operator = context.space_data.active_operator
        # TODO: Add layout.operator() to read selected file and fill in semantics

        if context.path_resolve is None:
            # Avoid exceptions in console - seems like draw() is called several
            # times (not sure why) and sometimes path_resolve isn't available.
            return
        draw_ui_list(
            self.layout,
            context,
            class_name="MIGOTO_UL_semantic_remap_list",
            menu_class_name="MIGOTO_MT_semantic_remap_menu",
            list_path="active_operator.properties.semantic_remap",
            active_index_path="active_operator.properties.semantic_remap_idx",
            unique_id="migoto_import_semantic_remap_list",
            item_dyntip_propname="tooltip",
        )


class MIGOTO_PT_ImportFrameAnalysisManualOrientation(
    MigotoImportOptionsPanelBase, Panel
):
    bl_label = "Orientation"
    bl_order = 5

    def draw(self, context):
        MigotoImportOptionsPanelBase.draw(self, context)
        operator = context.space_data.active_operator
        self.layout.prop(operator, "axis_forward")
        self.layout.prop(operator, "axis_up")


class MIGOTO_PT_ImportFrameAnalysisCleanUp(MigotoImportOptionsPanelBase, Panel):
    bl_label = "Clean Up mesh after import"
    bl_order = 6

    def draw(self, context):
        MigotoImportOptionsPanelBase.draw(self, context)
        operator = context.space_data.active_operator
        self.layout.prop(operator, "merge_verts")
        self.layout.prop(operator, "tris_to_quads")
        self.layout.prop(operator, "clean_loose")


class XXMI_PT_Sidebar(Panel):
    """Main Panel"""

    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "XXMI Tools"
    bl_idname = "XXMI_PT_Sidebar"
    bl_label = "XXMI Tools"
    bl_context = "objectmode"

    def draw_header(self, context):
        version = ""
        for module in addon_utils.modules():
            if module.bl_info.get("name") == "XXMI_Tools":
                version = module.bl_info.get("version", (-1, -1, -1))
                break
        version: str = ".".join(str(i) for i in version)
        layout: UILayout = self.layout
        row = layout.row()
        row.operator(
            "wm.url_open", text="", icon="HELP"
        ).url = "https://leotorrez.github.io/modding/guides/xxmi_tools"
        row.label(text=f"v{version}")

    def draw(self, context):
        layout = self.layout
        xxmi: XXMIProperties = context.scene.xxmi
        split = layout.split(factor=0.85)
        col_1 = split.column()
        col_2 = split.column()
        col_1.prop(xxmi, "dump_path")
        col_2.operator("dump.selector", icon="FILE_FOLDER", text="")
        col_1.prop(xxmi, "destination_path")
        col_2.operator("destination.selector", icon="FILE_FOLDER", text="")
        col = layout.column(align=True)
        layout.separator()
        col.prop(xxmi, "game")


class XXMISidebarOptionsPanelBase(object):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = "XXMI_PT_Sidebar"

    # @classmethod
    # def poll(cls, context):
    #     operator = context.space_data.active_operator
    #     return operator.bl_idname == "XXMI_PT_Sidebar"

    def draw(self, context):
        self.layout.use_property_split = False
        self.layout.use_property_decorate = False


class XXMI_PT_SidePanelExportSettings(XXMISidebarOptionsPanelBase, Panel):
    bl_label = "Export Settings"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 0

    def draw(self, context):
        XXMISidebarOptionsPanelBase.draw(self, context)
        xxmi: XXMIProperties = context.scene.xxmi
        layout: bpy.types.UILayout = self.layout
        box = layout.box()
        col = box.column(align=True)
        col.prop(xxmi, "ignore_hidden")
        col.prop(xxmi, "only_selected")
        col.prop(xxmi, "apply_modifiers_and_shapekeys")
        col.prop(xxmi, "normalize_weights")
        col.separator()
        col.prop(xxmi, "copy_textures")
        if xxmi.copy_textures:
            box_tex = col.box()
            box_tex.prop(xxmi, "no_ramps")
            box_tex.prop(xxmi, "ignore_duplicate_textures")
        col.prop(xxmi, "write_buffers")
        col.prop(xxmi, "write_ini")
        if xxmi.write_ini:
            box_ini = col.box()
            split = box_ini.split(factor=0.85)
            col_1 = split.column()
            col_2 = split.column()
            split_child = col_1.split(factor=0.05)
            col_1_1 = split_child.column()
            col_1_2 = split_child.column()
            col_1_1.prop(xxmi, "use_custom_template", text="")
            col_1_2.enabled = xxmi.use_custom_template
            col_2.enabled = xxmi.use_custom_template
            col_1_2.prop(xxmi, "template_path")
            col_2.operator("template.selector", icon="FILE_FOLDER", text="")
            box_ini.prop(xxmi, "credit")
        col = box.column(align=True)
        split = col.split(factor=0.25)
        col_1 = split.column()
        col_2 = split.column()
        col_1.prop(xxmi, "outline_optimization")
        col_2.enabled = xxmi.outline_optimization
        col_2.prop(xxmi, "outline_rounding_precision")
        # col.prop(xxmi, 'export_shapekeys')
        # col.prop(xxmi, "export_materials")


class XXMI_PT_SidePanelBatchExport(XXMISidebarOptionsPanelBase, Panel):
    bl_label = ""
    bl_options = {"HIDE_HEADER"}
    bl_order = 99

    def draw(self, context):
        XXMISidebarOptionsPanelBase.draw(self, context)
        xxmi: XXMIProperties = context.scene.xxmi
        row = self.layout.column(align=True)
        split = row.split(factor=0.5)
        col1 = split.column()
        col2 = split.column()
        if xxmi.write_buffers or xxmi.write_ini or xxmi.copy_textures:
            col1.prop(xxmi, "batch_pattern")
            col2.operator("xxmi.exportadvancedbatched", text="Start Batch export")


class XXMI_PT_SidePanelExport(XXMISidebarOptionsPanelBase, Panel):
    bl_label = ""
    bl_options = {"HIDE_HEADER"}
    bl_order = 98

    def draw(self, context):
        XXMISidebarOptionsPanelBase.draw(self, context)
        layout = self.layout
        row = layout.row()
        xxmi: XXMIProperties = context.scene.xxmi
        if not xxmi.write_buffers and not xxmi.write_ini and not xxmi.copy_textures:
            row.label(text="Nothing to export", icon="ERROR")
        else:
            row.operator("xxmi.exportadvanced", text="Export Mod")


class UpdaterPanel(Panel):
    """Update Panel"""

    bl_label = "Updater"
    bl_idname = "XXMI_PT_UpdaterPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "XXMI Tools"
    bl_order = 99
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout

        # Call to check for update in background.
        # Note: built-in checks ensure it runs at most once, and will run in
        # the background thread, not blocking or hanging blender.
        # Internally also checks to see if auto-check enabled and if the time
        # interval has passed.
        addon_updater_ops.check_for_update_background()
        col = layout.column()
        col.scale_y = 0.7
        # Could also use your own custom drawing based on shared variables.
        if addon_updater_ops.updater.update_ready:
            layout.label(text="There's a new update available!", icon="INFO")

        # Call built-in function with draw code/checks.
        addon_updater_ops.update_notice_box_ui(self, context)
        addon_updater_ops.update_settings_ui(self, context)


def menu_func_import_fa(self, context):
    self.layout.operator(
        Import3DMigotoFrameAnalysis.bl_idname,
        text="3DMigoto frame analysis dump (vb.txt + ib.txt)",
    )


def menu_func_import_raw(self, context):
    self.layout.operator(
        Import3DMigotoRaw.bl_idname, text="3DMigoto raw buffers (.vb + .ib)"
    )


def menu_func_import_pose(self, context):
    self.layout.operator(Import3DMigotoPose.bl_idname, text="3DMigoto pose (.txt)")


def menu_func_export(self, context):
    self.layout.operator(
        Export3DMigoto.bl_idname, text="3DMigoto raw buffers (.vb + .ib)"
    )


def menu_func_export_xxmi(self, context):
    self.layout.operator(Export3DMigotoXXMI.bl_idname, text="Exports Mod Folder")


def menu_func_apply_vgmap(self, context):
    self.layout.operator(
        ApplyVGMap.bl_idname,
        text="Apply 3DMigoto vertex group map to current object (.vgmap)",
    )


import_menu = bpy.types.TOPBAR_MT_file_import
export_menu = bpy.types.TOPBAR_MT_file_export


def register():
    import_menu.append(menu_func_import_fa)
    import_menu.append(menu_func_import_raw)
    export_menu.append(menu_func_export)
    export_menu.append(menu_func_export_xxmi)
    import_menu.append(menu_func_apply_vgmap)
    import_menu.append(menu_func_import_pose)


def unregister():
    import_menu.remove(menu_func_import_fa)
    import_menu.remove(menu_func_import_raw)
    export_menu.remove(menu_func_export)
    export_menu.remove(menu_func_export_xxmi)
    import_menu.remove(menu_func_apply_vgmap)
    import_menu.remove(menu_func_import_pose)
