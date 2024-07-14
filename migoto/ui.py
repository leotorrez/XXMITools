import bpy
from bl_ui.generic_ui_list import draw_ui_list
from .operators import ClearSemanticRemapList,PrefillSemanticRemapList, Import3DMigotoFrameAnalysis, Import3DMigotoRaw, Import3DMigotoPose, Export3DMigoto, ApplyVGMap, UpdateVGMap, Import3DMigotoReferenceInputFormat, Export3DMigotoXXMI, Merge3DMigotoPose, DeleteNonNumericVertexGroups

class MIGOTO_UL_semantic_remap_list(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.prop(item, "semantic_from", text="", emboss=False, icon_value=icon)
            if item.InputSlotClass == 'per-instance':
                layout.label(text="Instanced Data")
                layout.enabled = False
            elif item.valid == False:
                layout.label(text="INVALID")
                layout.enabled = False
            else:
                layout.prop(item, "semantic_to", text="", emboss=False, icon_value=icon)
        elif self.layout_type == 'GRID':
            # Doco says we must implement this layout type, but I don't see
            # that it would be particularly useful, and not sure if we actually
            # expect the list to render with this type in practice. Untested.
            layout.alignment = 'CENTER'
            layout.label(text="", icon_value=icon)

class MIGOTO_MT_semantic_remap_menu(bpy.types.Menu):
    bl_label = "Semantic Remap Options"

    def draw(self, context):
        layout = self.layout


        layout.operator(ClearSemanticRemapList.bl_idname)
        layout.operator(PrefillSemanticRemapList.bl_idname)

class MigotoImportOptionsPanelBase:
    bl_space_type = 'FILE_BROWSER'
    bl_region_type = 'TOOL_PROPS'
    bl_parent_id = "FILE_PT_operator"

    @classmethod
    def poll(cls, context):
        operator = context.space_data.active_operator
        return operator.bl_idname == "IMPORT_MESH_OT_migoto_frame_analysis"

    def draw(self, context):
        self.layout.use_property_split = True
        self.layout.use_property_decorate = False

class MIGOTO_PT_ImportFrameAnalysisMainPanel(MigotoImportOptionsPanelBase, bpy.types.Panel):
    bl_label = ""
    bl_options = {'HIDE_HEADER'}

    def draw(self, context):
        MigotoImportOptionsPanelBase.draw(self, context)
        operator = context.space_data.active_operator
        self.layout.prop(operator, "flip_texcoord_v")
        self.layout.prop(operator, "flip_winding")
        self.layout.prop(operator, "flip_normal")

class MIGOTO_PT_ImportFrameAnalysisRelatedFilesPanel(MigotoImportOptionsPanelBase, bpy.types.Panel):
    bl_label = ""
    bl_options = {'HIDE_HEADER'}

    def draw(self, context):
        MigotoImportOptionsPanelBase.draw(self, context)
        operator = context.space_data.active_operator
        self.layout.enabled = not operator.load_buf
        self.layout.prop(operator, "load_related")
        #self.layout.prop(operator, "load_related_so_vb")
        self.layout.prop(operator, "merge_meshes")

class MIGOTO_PT_ImportFrameAnalysisBufFilesPanel(MigotoImportOptionsPanelBase, bpy.types.Panel):
    bl_label = "Load .buf files instead"
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        operator = context.space_data.active_operator
        self.layout.prop(operator, "load_buf", text="")

    def draw(self, context):
        MigotoImportOptionsPanelBase.draw(self, context)
        operator = context.space_data.active_operator
        self.layout.enabled = operator.load_buf
        self.layout.prop(operator, "load_buf_limit_range")

class MIGOTO_PT_ImportFrameAnalysisBonePanel(MigotoImportOptionsPanelBase, bpy.types.Panel):
    bl_label = ""
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        operator = context.space_data.active_operator
        self.layout.prop(operator, "pose_cb")

    def draw(self, context):
        MigotoImportOptionsPanelBase.draw(self, context)
        operator = context.space_data.active_operator
        self.layout.prop(operator, "pose_cb_off")
        self.layout.prop(operator, "pose_cb_step")

class MIGOTO_PT_ImportFrameAnalysisRemapSemanticsPanel(MigotoImportOptionsPanelBase, bpy.types.Panel):
    bl_label = "Semantic Remap"

     #bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        MigotoImportOptionsPanelBase.draw(self, context)
        operator = context.space_data.active_operator

        if context.path_resolve is None:
            return
        draw_ui_list(self.layout, context,
                class_name='MIGOTO_UL_semantic_remap_list',
                menu_class_name='MIGOTO_MT_semantic_remap_menu',
                list_path='active_operator.properties.semantic_remap',
                active_index_path='active_operator.properties.semantic_remap_idx',
                unique_id='migoto_import_semantic_remap_list',
                item_dyntip_propname='tooltip',
                )

class MIGOTO_PT_ImportFrameAnalysisManualOrientation(MigotoImportOptionsPanelBase, bpy.types.Panel):
    bl_label = "Orientation"

    def draw(self, context):
        MigotoImportOptionsPanelBase.draw(self, context)
        operator = context.space_data.active_operator
        self.layout.prop(operator, "axis_forward")
        self.layout.prop(operator, "axis_up")

def menu_func_import_fa(self, context):
    self.layout.operator(Import3DMigotoFrameAnalysis.bl_idname, text="3DMigoto frame analysis dump (vb.txt + ib.txt)")

def menu_func_import_raw(self, context):
    self.layout.operator(Import3DMigotoRaw.bl_idname, text="3DMigoto raw buffers (.vb + .ib)")

def menu_func_import_pose(self, context):
    self.layout.operator(Import3DMigotoPose.bl_idname, text="3DMigoto pose (.txt)")

def menu_func_export(self, context):
    self.layout.operator(Export3DMigoto.bl_idname, text="3DMigoto raw buffers (.vb + .ib)")

def menu_func_export_xxmi(self, context):
    self.layout.operator(Export3DMigotoXXMI.bl_idname, text="Exports Mod Folder")

def menu_func_apply_vgmap(self, context):
    self.layout.operator(ApplyVGMap.bl_idname, text="Apply 3DMigoto vertex group map to current object (.vgmap)")

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
