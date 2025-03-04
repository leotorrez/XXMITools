from glob import glob
import re
import itertools
import os
import time
import bpy
from bpy_extras.io_utils import  ImportHelper, ExportHelper, orientation_helper
from bpy.props import BoolProperty, StringProperty, CollectionProperty, IntProperty
from .datahandling import load_3dmigoto_mesh, open_frame_analysis_log_file, find_stream_output_vertex_buffers, VBSOMapEntry, ImportPaths, Fatal, import_3dmigoto, import_3dmigoto_raw_buffers, import_pose, merge_armatures, apply_vgmap, update_vgmap, export_3dmigoto, game_enums, export_3dmigoto_xxmi, SemanticRemapItem, silly_lookup
from .. import addon_updater_ops
from .. import __name__ as package_name

IOOBJOrientationHelper = type('DummyIOOBJOrientationHelper', (object,), {})

class ClearSemanticRemapList(bpy.types.Operator):
    """Clear the semantic remap list"""
    bl_idname = "import_mesh.migoto_semantic_remap_clear"
    bl_label = "Clear list"

    def execute(self, context):
        import_operator = context.space_data.active_operator
        import_operator.properties.semantic_remap.clear()
        return {'FINISHED'}

class PrefillSemanticRemapList(bpy.types.Operator):
    """Add semantics from the selected files to the semantic remap list"""
    bl_idname = "import_mesh.migoto_semantic_remap_prefill"
    bl_label = "Prefill from selected files"

    def execute(self, context):
        import_operator = context.space_data.active_operator
        semantic_remap_list = import_operator.properties.semantic_remap
        semantics_in_list = { x.semantic_from for x in semantic_remap_list }

        paths = import_operator.get_vb_ib_paths(load_related=False)

        for p in paths:
            vb, ib, name, pose_path = load_3dmigoto_mesh(import_operator, [p])
            valid_semantics = vb.get_valid_semantics()
            for semantic in vb.layout:
                if semantic.name not in semantics_in_list:
                    remap = semantic_remap_list.add()
                    remap.semantic_from = semantic.name
                    # Store some extra information that can be helpful to guess the likely semantic:
                    remap.Format = semantic.Format
                    remap.InputSlot = semantic.InputSlot
                    remap.InputSlotClass = semantic.InputSlotClass
                    remap.AlignedByteOffset = semantic.AlignedByteOffset
                    remap.valid = semantic.name in valid_semantics
                    remap.update_tooltip()
                    semantics_in_list.add(semantic.name)

        return {'FINISHED'}

@orientation_helper(axis_forward='-Z', axis_up='Y')
class Import3DMigotoFrameAnalysis(bpy.types.Operator, ImportHelper, IOOBJOrientationHelper):
    """Import a mesh dumped with 3DMigoto's frame analysis"""
    bl_idname = "import_mesh.migoto_frame_analysis"
    bl_label = "Import 3DMigoto Frame Analysis Dump"
    bl_options = {'PRESET', 'UNDO'}

    filename_ext = '.txt'
    filter_glob: StringProperty(
            default='*.txt',
            options={'HIDDEN'},
            )

    files: CollectionProperty(
            name="File Path",
            type=bpy.types.OperatorFileListElement,
            )

    flip_texcoord_v: BoolProperty(
            name="Flip TEXCOORD V",
            description="Flip TEXCOORD V asix during importing",
            default=True,
            )

    flip_winding: BoolProperty(
            name="Flip Winding Order",
            description="Flip winding order (face orientation) during importing. Try if the model doesn't seem to be shading as expected in Blender and enabling the 'Face Orientation' overlay shows **RED** (if it shows BLUE, try 'Flip Normal' instead). Not quite the same as flipping normals within Blender as this only reverses the winding order without flipping the normals. Recommended for Unreal Engine",
            default=False,
            )
    
    flip_mesh: BoolProperty(
            name="Flip Mesh",
            description="Mirrors mesh over the X Axis on import, and invert the winding order.",
            default=False,
            )

    flip_normal: BoolProperty(
            name="Flip Normal",
            description="Flip Normals during importing. Try if the model doesn't seem to be shading as expected in Blender and enabling 'Face Orientation' overlay shows **BLUE** (if it shows RED, try 'Flip Winding Order' instead). Not quite the same as flipping normals within Blender as this won't reverse the winding order",
            default=False,
            )

    load_related: BoolProperty(
            name="Auto-load related meshes",
            description="Automatically load related meshes found in the frame analysis dump",
            default=True,
            )

    load_related_so_vb: BoolProperty(
            name="Load pre-SO buffers (EXPERIMENTAL)",
            description="Scans the frame analysis log file to find GPU pre-skinning Stream Output techniques in prior draw calls, and loads the unposed vertex buffers from those calls that are suitable for editing. Recommended for Unity games to load neutral poses",
            default=False,
            )

    load_buf: BoolProperty(
            name="Load .buf files instead",
            description="Load the mesh from the binary .buf dumps instead of the .txt files\nThis will load the entire mesh as a single object instead of separate objects from each draw call",
            default=False,
            )

    load_buf_limit_range: BoolProperty(
            name="Limit to draw range",
            description="Load just the vertices/indices used in the draw call (equivalent to loading the .txt files) instead of the complete buffer",
            default=False,
            )

    merge_meshes: BoolProperty(
            name="Merge meshes together",
            description="Merge all selected meshes together into one object. Meshes must be related",
            default=False,
            )

    pose_cb: StringProperty(
            name="Bone CB",
            description='Indicate a constant buffer slot (e.g. "vs-cb2") containing the bone matrices',
            default="",
            )

    pose_cb_off: bpy.props.IntVectorProperty(
            name="Bone CB range",
            description='Indicate start and end offsets (in multiples of 4 component values) to find the matrices in the Bone CB',
            default=[0,0],
            size=2,
            min=0,
            )

    pose_cb_step: bpy.props.IntProperty(
            name="Vertex group step",
            description='If used vertex groups are 0,1,2,3,etc specify 1. If they are 0,3,6,9,12,etc specify 3',
            default=1,
            min=1,
            )

    semantic_remap: bpy.props.CollectionProperty(type=SemanticRemapItem)
    semantic_remap_idx: bpy.props.IntProperty(
            name='Semantic Remap',
            description='Enter the SemanticName and SemanticIndex the game is using on the left (e.g. TEXCOORD3), and what type of semantic the script should treat it as on the right') # Needed for template_list

    merge_verts: BoolProperty(
            name="Merge Vertices",
            description="Merge by distance to remove duplicate vertices",
            default=False,
            )
    tris_to_quads: BoolProperty(
            name="Tris to Quads",
            description="Convert all tris to quads",
            default=False,
            )
    clean_loose: BoolProperty(
            name="Clean Loose",
            description="Remove loose geometry",
            default=False,
            )
    

    def get_vb_ib_paths(self, load_related=None):
        buffer_pattern = re.compile(r'''-(?:ib|vb[0-9]+)(?P<hash>=[0-9a-f]+)?(?=[^0-9a-f=])''')
        vb_regex = re.compile(r'''^(?P<draw_call>[0-9]+)-vb(?P<slot>[0-9]+)=''') # TODO: Combine with above? (careful not to break hold type frame analysis)

        dirname = os.path.dirname(self.filepath)
        ret = set()
        if load_related is None:
            load_related = self.load_related

        vb_so_map = {}
        if self.load_related_so_vb:
            try:
                fa_log = open_frame_analysis_log_file(dirname)
            except FileNotFoundError:
                self.report({'WARNING'}, 'Frame Analysis Log File not found, loading unposed meshes from GPU Stream Output pre-skinning passes will be unavailable')
            else:
                vb_so_map = find_stream_output_vertex_buffers(fa_log)

        files = set()
        if load_related:
            for filename in self.files:
                match = buffer_pattern.search(filename.name)
                if match is None or not match.group('hash'):
                    continue
                paths = glob(os.path.join(dirname, '*%s*.txt' % filename.name[match.start():match.end()]))
                files.update([os.path.basename(x) for x in paths])
        if not files:
            files = [x.name for x in self.files]
            if files == ['']:
                raise Fatal('No files selected')

        done = set()
        for filename in files:
            if filename in done:
                continue
            match = buffer_pattern.search(filename)
            if match is None:
                if filename.lower().startswith('log') or filename.lower() == 'shaderusage.txt':
                    # User probably just selected all files including the log
                    continue
                # TODO: Perhaps don't warn about extra files that may have been
                # dumped out that we aren't specifically importing (such as
                # constant buffers dumped with dump_cb or any buffer dumped
                # with dump=), maybe provided we are loading other files from
                # that draw call. Note that this is only applicable if 'load
                # related' is disabled, as that option effectively filters
                # these out above. For now just changed this to an error report
                # rather than a Fatal so other files will still get loaded.
                self.report({'ERROR'}, 'Unable to find corresponding buffers from "{}" - filename did not match vertex/index buffer pattern'.format(filename))
                continue

            use_bin = self.load_buf
            if not match.group('hash') and not use_bin:
                self.report({'INFO'}, 'Filename did not contain hash - if Frame Analysis dumped a custom resource the .txt file may be incomplete, Using .buf files instead')
                use_bin = True # FIXME: Ask

            ib_pattern = filename[:match.start()] + '-ib*' + filename[match.end():]
            vb_pattern = filename[:match.start()] + '-vb*' + filename[match.end():]
            ib_paths = glob(os.path.join(dirname, ib_pattern))
            vb_paths = glob(os.path.join(dirname, vb_pattern))
            done.update(map(os.path.basename, itertools.chain(vb_paths, ib_paths)))

            if vb_so_map:
                vb_so_paths = set()
                for vb_path in vb_paths:
                    vb_match = vb_regex.match(os.path.basename(vb_path))
                    if vb_match:
                        draw_call, slot = map(int, vb_match.group('draw_call', 'slot'))
                        so = vb_so_map.get(VBSOMapEntry(draw_call, slot))
                        if so:
                            # No particularly good way to determine which input
                            # vertex buffers we need from the stream-output
                            # pass, so for now add them all:
                            vb_so_pattern = f'{so.draw_call:06}-vb*.txt'
                            glob_result = glob(os.path.join(dirname, vb_so_pattern))
                            if not glob_result:
                                self.report({'WARNING'}, f'{vb_so_pattern} not found, loading unposed meshes from GPU Stream Output pre-skinning passes will be unavailable')
                            vb_so_paths.update(glob_result)
                # FIXME: Not sure yet whether the extra vertex buffers from the
                # stream output pre-skinning passes are best lumped in with the
                # existing vb_paths or added as a separate set of paths. Advantages
                # + disadvantages to each, and either way will need work.
                vb_paths.extend(sorted(vb_so_paths))

            if vb_paths and use_bin:
                vb_bin_paths = [ os.path.splitext(x)[0] + '.buf' for x in vb_paths ]
                ib_bin_paths = [ os.path.splitext(x)[0] + '.buf' for x in ib_paths ]
                if all([ os.path.exists(x) for x in itertools.chain(vb_bin_paths, ib_bin_paths) ]):
                    # When loading the binary files, we still need to process
                    # the .txt files as well, as they indicate the format:
                    ib_paths = list(zip(ib_bin_paths, ib_paths))
                    vb_paths = list(zip(vb_bin_paths, vb_paths))
                else:
                    self.report({'WARNING'}, 'Corresponding .buf files not found - using .txt files')
                    use_bin = False

            pose_path = None
            if self.pose_cb:
                pose_pattern = filename[:match.start()] + '*-' + self.pose_cb + '=*.txt'
                try:
                    pose_path = glob(os.path.join(dirname, pose_pattern))[0]
                except IndexError:
                    pass

            if len(ib_paths) > 1:
                raise Fatal('Error: excess index buffers in dump?')
            elif len(ib_paths) == 0:
                if use_bin:
                    name = os.path.basename(vb_paths[0][0])
                    ib_paths = [(None, None)]
                else:
                    name = os.path.basename(vb_paths[0])
                    ib_paths = [None]
                self.report({'WARNING'}, '{}: No index buffer present, support for this case is highly experimental'.format(name))
            ret.add(ImportPaths(tuple(vb_paths), ib_paths[0], use_bin, pose_path))
        return ret

    def execute(self, context):
        if self.load_buf:
            # Is there a way to have the mutual exclusivity reflected in
            # the UI? Grey out options or use radio buttons or whatever?
            if self.merge_meshes or self.load_related:
                self.report({'INFO'}, 'Loading .buf files selected: Disabled incompatible options')
            self.merge_meshes = False
            self.load_related = False

        try:
            keywords = self.as_keywords(ignore=('filepath', 'files',
                'filter_glob', 'load_related', 'load_related_so_vb',
                'load_buf', 'pose_cb', 'load_buf_limit_range',
                'semantic_remap', 'semantic_remap_idx'))
            paths = self.get_vb_ib_paths()

            import_3dmigoto(self, context, paths, **keywords)
            xxmi = context.scene.xxmi
            if not xxmi.dump_path:
                if os.path.exists(os.path.join(os.path.dirname(self.filepath), 'hash.json')):
                    xxmi.dump_path = os.path.dirname(self.filepath)
        except Fatal as e:
            self.report({'ERROR'}, str(e))
        return {'FINISHED'}

    def draw(self, context):
        # Overriding the draw method to disable automatically adding operator
        # properties to options panel, so we can define sub-panels to group
        # options and disable grey out mutually exclusive options.
        pass

@orientation_helper(axis_forward='-Z', axis_up='Y')
class Import3DMigotoRaw(bpy.types.Operator, ImportHelper, IOOBJOrientationHelper):
    """Import raw 3DMigoto vertex and index buffers"""
    bl_idname = "import_mesh.migoto_raw_buffers"
    bl_label = "Import 3DMigoto Raw Buffers"
    #bl_options = {'PRESET', 'UNDO'}
    bl_options = {'UNDO'}

    filename_ext = '.vb;.ib'
    filter_glob: StringProperty(
            default='*.vb*;*.ib',
            options={'HIDDEN'},
            )

    files: CollectionProperty(
            name="File Path",
            type=bpy.types.OperatorFileListElement,
            )

    flip_texcoord_v: BoolProperty(
            name="Flip TEXCOORD V",
            description="Flip TEXCOORD V axis during importing",
            default=True,
            )

    flip_winding: BoolProperty(
            name="Flip Winding Order",
            description="Flip winding order (face orientation) during importing. Try if the model doesn't seem to be shading as expected in Blender and enabling the 'Face Orientation' overlay shows **RED** (if it shows BLUE, try 'Flip Normal' instead). Not quite the same as flipping normals within Blender as this only reverses the winding order without flipping the normals. Recommended for Unreal Engine",
            default=False,
            )

    flip_normal: BoolProperty(
            name="Flip Normal",
            description="Flip Normals during importing. Try if the model doesn't seem to be shading as expected in Blender and enabling 'Face Orientation' overlay shows **BLUE** (if it shows RED, try 'Flip Winding Order' instead). Not quite the same as flipping normals within Blender as this won't reverse the winding order",
            default=False,
            )

    def get_vb_ib_paths(self, filename):
        vb_bin_path = glob(os.path.splitext(filename)[0] + '.vb*')
        ib_bin_path = os.path.splitext(filename)[0] + '.ib'
        fmt_path = os.path.splitext(filename)[0] + '.fmt'
        vgmap_path = os.path.splitext(filename)[0] + '.vgmap'
        if len(vb_bin_path) < 1:
            raise Fatal('Unable to find matching .vb* file(s) for %s' % filename)
        if not os.path.exists(ib_bin_path):
            ib_bin_path = None
        if not os.path.exists(fmt_path):
            fmt_path = None
        if not os.path.exists(vgmap_path):
            vgmap_path = None
        return (vb_bin_path, ib_bin_path, fmt_path, vgmap_path)

    def execute(self, context):
        # I'm not sure how to find the Import3DMigotoReferenceInputFormat
        # instance that Blender instantiated to pass the values from one
        # import dialog to another, but since everything is modal we can
        # just use globals:
        global migoto_raw_import_options
        migoto_raw_import_options = self.as_keywords(ignore=('filepath', 'files', 'filter_glob'))

        done = set()
        dirname = os.path.dirname(self.filepath)
        for filename in self.files:
            try:
                (vb_path, ib_path, fmt_path, vgmap_path) = self.get_vb_ib_paths(os.path.join(dirname, filename.name))
                vb_path_norm = set(map(os.path.normcase, vb_path))
                if vb_path_norm.intersection(done) != set():
                    continue
                done.update(vb_path_norm)

                if fmt_path is not None:
                    import_3dmigoto_raw_buffers(self, context, fmt_path, fmt_path, vb_path=vb_path, ib_path=ib_path, vgmap_path=vgmap_path, **migoto_raw_import_options)
                else:
                    migoto_raw_import_options['vb_path'] = vb_path
                    migoto_raw_import_options['ib_path'] = ib_path
                    bpy.ops.import_mesh.migoto_input_format('INVOKE_DEFAULT')
            except Fatal as e:
                self.report({'ERROR'}, str(e))
        return {'FINISHED'}

class Import3DMigotoReferenceInputFormat(bpy.types.Operator, ImportHelper):
    bl_idname = "import_mesh.migoto_input_format"
    bl_label = "Select a .txt file with matching format"
    bl_options = {'UNDO', 'INTERNAL'}

    filename_ext = '.txt;.fmt'
    filter_glob: StringProperty(
            default='*.txt;*.fmt',
            options={'HIDDEN'},
            )

    def get_vb_ib_paths(self):
        if os.path.splitext(self.filepath)[1].lower() == '.fmt':
            return (self.filepath, self.filepath)

        buffer_pattern = re.compile(r'''-(?:ib|vb[0-9]+)(?P<hash>=[0-9a-f]+)?(?=[^0-9a-f=])''')

        dirname = os.path.dirname(self.filepath)
        filename = os.path.basename(self.filepath)

        match = buffer_pattern.search(filename)
        if match is None:
            raise Fatal('Reference .txt filename does not look like a 3DMigoto timestamped Frame Analysis Dump')
        ib_pattern = filename[:match.start()] + '-ib*' + filename[match.end():]
        vb_pattern = filename[:match.start()] + '-vb*' + filename[match.end():]
        ib_paths = glob(os.path.join(dirname, ib_pattern))
        vb_paths = glob(os.path.join(dirname, vb_pattern))
        if len(ib_paths) < 1 or len(vb_paths) < 1:
            raise Fatal('Unable to locate reference files for both vertex buffer and index buffer format descriptions')
        return (vb_paths[0], ib_paths[0])

    def execute(self, context):
        global migoto_raw_import_options

        try:
            vb_fmt_path, ib_fmt_path = self.get_vb_ib_paths()
            import_3dmigoto_raw_buffers(self, context, vb_fmt_path, ib_fmt_path, **migoto_raw_import_options)
        except Fatal as e:
            self.report({'ERROR'}, str(e))
        return {'FINISHED'}

class Export3DMigoto(bpy.types.Operator, ExportHelper):
    """Export a mesh for re-injection into a game with 3DMigoto"""
    bl_idname = "export_mesh.migoto"
    bl_label = "Export 3DMigoto Vertex & Index Buffers"

    filename_ext = '.vb0'
    filter_glob: StringProperty(
            default='*.vb*',
            options={'HIDDEN'},
            )

    def invoke(self, context, event):
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        try:
            vb_path = os.path.splitext(self.filepath)[0] + '.vb'
            ib_path = os.path.splitext(vb_path)[0] + '.ib'
            fmt_path = os.path.splitext(vb_path)[0] + '.fmt'
            ini_path = os.path.splitext(vb_path)[0] + '_generated.ini'
            obj = context.object
            self.flip_normal = obj.get("3DMigoto:FlipNormal", False)
            self.flip_tangent = obj.get("3DMigoto:FlipTangent", False)
            self.flip_winding = obj.get("3DMigoto:FlipWinding", False)
            self.flip_mesh = obj.get("3DMigoto:FlipMesh", False)
            # FIXME: ExportHelper will check for overwriting vb_path, but not ib_path
            export_3dmigoto(self, context, vb_path, ib_path, fmt_path, ini_path)
        except Fatal as e:
            self.report({'ERROR'}, str(e))
        return {'FINISHED'}

class ApplyVGMap(bpy.types.Operator, ImportHelper):
    """Apply vertex group map to the selected object"""
    bl_idname = "mesh.migoto_vertex_group_map"
    bl_label = "Apply 3DMigoto vgmap"
    bl_options = {'UNDO'}

    filename_ext = '.vgmap'
    filter_glob: StringProperty(
            default='*.vgmap',
            options={'HIDDEN'},
            )

    #commit: BoolProperty(
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
            default='',
            )

    def invoke(self, context, event):
        self.suffix = ''
        return ImportHelper.invoke(self, context, event)

    def execute(self, context):
        try:
            keywords = self.as_keywords(ignore=('filter_glob',))
            apply_vgmap(self, context, **keywords)
        except Fatal as e:
            self.report({'ERROR'}, str(e))
        return {'FINISHED'}

class UpdateVGMap(bpy.types.Operator):
    """Assign new 3DMigoto vertex groups"""
    bl_idname = "mesh.update_migoto_vertex_group_map"
    bl_label = "Assign new 3DMigoto vertex groups"
    bl_options = {'UNDO'}

    vg_step: bpy.props.IntProperty(
            name="Vertex group step",
            description='If used vertex groups are 0,1,2,3,etc specify 1. If they are 0,3,6,9,12,etc specify 3',
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
            self.report({'ERROR'}, str(e))
        return {'FINISHED'}

@orientation_helper(axis_forward='-Z', axis_up='Y')
class Import3DMigotoPose(bpy.types.Operator, ImportHelper, IOOBJOrientationHelper):
    """Import a pose from a 3DMigoto constant buffer dump"""
    bl_idname = "armature.migoto_pose"
    bl_label = "Import 3DMigoto Pose"
    bl_options = {'UNDO'}

    filename_ext = '.txt'
    filter_glob: StringProperty(
            default='*.txt',
            options={'HIDDEN'},
            )

    limit_bones_to_vertex_groups: BoolProperty(
            name="Limit Bones to Vertex Groups",
            description="Limits the maximum number of bones imported to the number of vertex groups of the active object",
            default=True,
            )

    pose_cb_off: bpy.props.IntVectorProperty(
            name="Bone CB range",
            description='Indicate start and end offsets (in multiples of 4 component values) to find the matrices in the Bone CB',
            default=[0,0],
            size=2,
            min=0,
            )

    pose_cb_step: bpy.props.IntProperty(
            name="Vertex group step",
            description='If used vertex groups are 0,1,2,3,etc specify 1. If they are 0,3,6,9,12,etc specify 3',
            default=1,
            min=1,
            )

    def execute(self, context):
        try:
            keywords = self.as_keywords(ignore=('filter_glob',))
            import_pose(self, context, **keywords)
        except Fatal as e:
            self.report({'ERROR'}, str(e))
        return {'FINISHED'}

class Merge3DMigotoPose(bpy.types.Operator):
    """Merge identically posed bones of related armatures into one"""
    bl_idname = "armature.merge_pose"
    bl_label = "Merge 3DMigoto Poses"
    bl_options = {'UNDO'}

    def execute(self, context):
        try:
            merge_armatures(self, context)
        except Fatal as e:
            self.report({'ERROR'}, str(e))
        return {'FINISHED'}

class DeleteNonNumericVertexGroups(bpy.types.Operator):
    """Remove vertex groups with non-numeric names"""
    bl_idname = "vertex_groups.delete_non_numeric"
    bl_label = "Remove non-numeric vertex groups"
    bl_options = {'UNDO'}

    def execute(self, context):
        try:
            for obj in context.selected_objects:
                for vg in reversed(obj.vertex_groups):
                    if vg.name.isdecimal():
                        continue
                    print('Removing vertex group', vg.name)
                    obj.vertex_groups.remove(vg)
        except Fatal as e:
            self.report({'ERROR'}, str(e))
        return {'FINISHED'}

class Export3DMigotoXXMI(bpy.types.Operator, ExportHelper):
    """Export a mesh for re-injection into a game with 3DMigoto"""
    bl_idname = "export_mesh_xxmi.migoto"
    bl_label = "Export mod folder"
    bl_options = {'PRESET', 'UNDO'}

    filename_ext = '.vb*'
    filter_glob: StringProperty(
            default='*.vb*',
            options={'HIDDEN'},
            )

    use_foldername : BoolProperty(
        name="Use foldername when exporting",
        description="Sets the export name equal to the foldername you are exporting to. Keep true unless you have changed the names",
        default=True,
    )

    ignore_hidden : BoolProperty(
        name="Ignore hidden objects",
        description="Does not use objects in the Blender window that are hidden while exporting mods",
        default=True,
    )

    only_selected : BoolProperty(
        name="Only export selected",
        description="Uses only the selected objects when deciding which meshes to export",
        default=False,
    )

    no_ramps : BoolProperty(
        name="Ignore shadow ramps/metal maps/diffuse guide",
        description="Skips exporting shadow ramps, metal maps and diffuse guides",
        default=True,
    )

    delete_intermediate : BoolProperty(
        name="Delete intermediate files",
        description="Deletes the intermediate vb/ib files after a successful export to reduce clutter",
        default=True,
    )

    copy_textures : BoolProperty(
        name="Copy textures",
        description="Copies the texture files to the mod folder, useful for the initial export but might be redundant afterwards.",
        default=True,
    )

    credit : StringProperty(
        name="Credit",
        description="Name that pops up on screen when mod is loaded. If left blank, will result in no pop up",
        default='',
    )
    
    outline_optimization : BoolProperty(
        name="Outline Optimization",
        description="Recalculate outlines. Recommended for final export. Check more options below to improve quality. This option is tailored for Genshin Impact and may not work as well for other games. Use with caution.",
        default=False,
    )
    
    toggle_rounding_outline : BoolProperty(
        name="Round vertex positions",
        description="Rounding of vertex positions to specify which are the overlapping vertices",
        default=True,
    ) 
    
    decimal_rounding_outline : bpy.props.IntProperty(
        name="Decimals:",
        description="Rounding of vertex positions to specify which are the overlapping vertices",
        default=3,
    )

    angle_weighted : BoolProperty(
        name="Weight by angle",
        description="Optional: calculate angles to improve accuracy of outlines. Slow",
        default=False,
    )

    overlapping_faces : BoolProperty(
        name="Ignore overlapping faces",
        description="Detect and ignore overlapping/antiparallel faces to avoid buggy outlines",
        default=False,
    )

    detect_edges : BoolProperty(
        name="Calculate edges",
        description="Calculate for disconnected edges when rounding, closing holes in the edge outline",
        default=False,
    )

    calculate_all_faces : BoolProperty(
        name="Calculate outline for all faces",
        description="Calculate outline for all faces, which is especially useful if you have any flat shaded non-edge faces. Slow",
        default=False,
    )

    nearest_edge_distance : bpy.props.FloatProperty(
        name="Distance:",
        description="Expand grouping for edge vertices within this radial distance to close holes in the edge outline. Requires rounding",
        default=0.001,
        soft_min=0,
    )
    game: bpy.props.EnumProperty(
    name="Game to mod",
    description="Select the game you are modding to optimize the mod for that game",
    items=game_enums,
    )    
    apply_modifiers_and_shapekeys: bpy.props.BoolProperty(
        name="Apply modifiers and shapekeys",
        description="Applies shapekeys and modifiers(unless marked MASK); then joins meshes to a single object. The criteria to join is as follows, the objects imported from dump are considered containers; collections starting with their same name are going to be joint into said containers",
        default=False,
    )
    join_meshes: bpy.props.BoolProperty(
        name="Join meshes",
        description="Joins all meshes into a single object. Allows for versatile pre-baked animation mods and blender like masking for toggles.",
        default=False,
    )
    normalize_weights: bpy.props.BoolProperty(
        name="Normalize weights to format",
        description="Limits weights to match export format. Also normalizes the remaining weights",
        default=False,
    )
    export_shapekeys: bpy.props.BoolProperty(
        name="Export shape keys",
        description="Exports marked shape keys for the selected object. Also generates the necessary sections in ini file",
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.prop(self, 'game')
        col.prop(self, 'use_foldername')
        col.prop(self, 'ignore_hidden')
        col.prop(self, 'only_selected')
        col.prop(self, 'no_ramps')
        col.prop(self, 'delete_intermediate')
        col.prop(self, 'copy_textures')
        col.prop(self, 'apply_modifiers_and_shapekeys')
        col.prop(self, 'join_meshes')
        col.prop(self, 'normalize_weights')
        # col.prop(self, 'export_shapekeys')
        layout.separator()
        col.prop(self, 'outline_optimization')
        
        if self.outline_optimization:
            col.prop(self, 'toggle_rounding_outline', text='Vertex Position Rounding', toggle=True, icon="SHADING_WIRE")
            col.prop(self, 'decimal_rounding_outline')
            if self.toggle_rounding_outline:
                col.prop(self, 'detect_edges')
            if self.detect_edges and self.toggle_rounding_outline:
                col.prop(self, 'nearest_edge_distance')
            col.prop(self, 'overlapping_faces')
            col.prop(self, 'angle_weighted')
            col.prop(self, 'calculate_all_faces')
        layout.separator()
        
        col.prop(self, 'credit')
    def invoke(self, context, event):
        obj = context.object
        if obj is None:
            try:
                obj = [obj for obj in bpy.data.objects if obj.type == 'MESH' and obj.visible_get()][0]
            except IndexError:
                return ExportHelper.invoke(self, context, event)
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        try:
            vb_path = self.filepath
            ib_path = os.path.splitext(vb_path)[0] + '.ib'
            fmt_path = os.path.splitext(vb_path)[0] + '.fmt'
            object_name = os.path.splitext(os.path.basename(self.filepath))[0]

            # FIXME: ExportHelper will check for overwriting vb_path, but not ib_path
            outline_properties = (self.outline_optimization, self.toggle_rounding_outline, self.decimal_rounding_outline, self.angle_weighted, self.overlapping_faces, self.detect_edges, self.calculate_all_faces, self.nearest_edge_distance)
            game = silly_lookup(self.game)
            export_3dmigoto_xxmi(self, context, object_name, vb_path, ib_path, fmt_path, self.use_foldername, self.ignore_hidden, self.only_selected, self.no_ramps, self.delete_intermediate, self.credit, self.copy_textures, outline_properties, game)
            self.report({'INFO'}, "Export completed")
        except Fatal as e:
            self.report({'ERROR'}, str(e))
        return {'FINISHED'}
class XXMIProperties(bpy.types.PropertyGroup):
    '''Properties for XXMITools'''
    destination_path: bpy.props.StringProperty(name="Output Folder", description="Output Folder:", default="", maxlen=1024,)
    dump_path: bpy.props.StringProperty(name="Dump Folder", description="Dump Folder:", default="", maxlen=1024,)
    filter_glob: StringProperty(
            default='*.vb*',
            options={'HIDDEN'},
            )

    flip_winding: BoolProperty(
            name="Flip Winding Order",
            description="Flip winding order during export (automatically set to match the import option)",
            default=False,
            )

    use_foldername : BoolProperty(
        name="Use foldername when exporting",
        description="Sets the export name equal to the foldername you are exporting to. Keep true unless you have changed the names",
        default=True,
    )

    ignore_hidden : BoolProperty(
        name="Ignore hidden objects",
        description="Does not use objects in the Blender window that are hidden while exporting mods",
        default=True,
    )

    only_selected : BoolProperty(
        name="Only export selected",
        description="Uses only the selected objects when deciding which meshes to export",
        default=False,
    )

    no_ramps : BoolProperty(
        name="Ignore shadow ramps/metal maps/diffuse guide",
        description="Skips exporting shadow ramps, metal maps and diffuse guides",
        default=True,
    )

    delete_intermediate : BoolProperty(
        name="Delete intermediate files",
        description="Deletes the intermediate vb/ib files after a successful export to reduce clutter",
        default=True,
    )

    copy_textures : BoolProperty(
        name="Copy textures",
        description="Copies the texture files to the mod folder, useful for the initial export but might be redundant afterwards",
        default=True,
    )

    credit : StringProperty(
        name="Credit",
        description="Name that pops up on screen when mod is loaded. If left blank, will result in no pop up",
        default='',
    )
    
    outline_optimization : BoolProperty(
        name="Outline Optimization",
        description="Recalculate outlines. Recommended for final export. Check more options below to improve quality",
        default=False,
    )
    
    toggle_rounding_outline : BoolProperty(
        name="Round vertex positions",
        description="Rounding of vertex positions to specify which are the overlapping vertices",
        default=True,
    ) 
    
    decimal_rounding_outline : bpy.props.IntProperty(
        name="Decimals:",
        description="Rounding of vertex positions to specify which are the overlapping vertices",
        default=3,
    )

    angle_weighted : BoolProperty(
        name="Weight by angle",
        description="Optional: calculate angles to improve accuracy of outlines. Slow",
        default=False,
    )

    overlapping_faces : BoolProperty(
        name="Ignore overlapping faces",
        description="Detect and ignore overlapping/antiparallel faces to avoid buggy outlines",
        default=False,
    )

    detect_edges : BoolProperty(
        name="Calculate edges",
        description="Calculate for disconnected edges when rounding, closing holes in the edge outline",
        default=False,
    )

    calculate_all_faces : BoolProperty(
        name="Calculate outline for all faces",
        description="Calculate outline for all faces, which is especially useful if you have any flat shaded non-edge faces. Slow",
        default=False,
    )

    nearest_edge_distance : bpy.props.FloatProperty(
        name="Distance:",
        description="Expand grouping for edge vertices within this radial distance to close holes in the edge outline. Requires rounding",
        default=0.001,
        soft_min=0,
    )
    game: bpy.props.EnumProperty(
    name="Game to mod",
    description="Select the game you are modding to optimize the mod for that game",
    items=game_enums,
    )
    apply_modifiers_and_shapekeys: bpy.props.BoolProperty(
        name="Apply modifiers and shapekeys",
        description="Applies shapekeys and modifiers(unless marked MASK); then joins meshes to a single object. The criteria to join is as follows, the objects imported from dump are considered containers; collections starting with their same name are going to be joint into said containers",
        default=False,
    )
    join_meshes: bpy.props.BoolProperty(
        name="Join meshes",
        description="Joins all meshes into a single object. Allows for versatile pre-baked animation mods and blender like masking for toggles.",
        default=False,
    )
    normalize_weights: bpy.props.BoolProperty(
        name="Normalize weights to format",
        description="Limits weights to match export format. Also normalizes the remaining weights",
        default=False,
    )
    export_shapekeys: bpy.props.BoolProperty(
        name="Export shape keys",
        description="Exports marked shape keys for the selected object. Also generates the necessary sections in ini file",
        default=False,
    )
    batch_pattern: bpy.props.StringProperty(
        name="Batch pattern",
        description="Pattern to name export folders. Example: name_###",
        default="",
    )
class DestinationSelector(bpy.types.Operator, ExportHelper):
    """Export single mod based on current frame"""
    bl_idname = "destination.selector"
    bl_label = "Destination"
    filename_ext = "."
    use_filter_folder = True
    filter_glob : bpy.props.StringProperty(default='.', options={'HIDDEN'},)

    def execute(self, context):
        userpath = self.properties.filepath
        if not os.path.isdir(userpath):
            userpath = os.path.dirname(userpath)
            self.properties.filepath = userpath
            if not os.path.isdir(userpath):
                msg = "Please select a directory not a file\n" + userpath
                self.report({'ERROR'}, msg)
                return {'CANCELLED'}
        context.scene.xxmi.destination_path = self.properties.filepath
        bpy.ops.ed.undo_push(message="XXMI Tools: destination selected")
        return{'FINISHED'}
class DumpSelector(bpy.types.Operator, ExportHelper):
    """Export single mod based on current frame"""
    bl_idname = "dump.selector"
    bl_label = "Dump folder selector"
    filename_ext = "."
    use_filter_folder = True
    filter_glob : bpy.props.StringProperty(default='.', options={'HIDDEN'},)

    def execute(self, context):
        userpath = self.properties.filepath
        if not os.path.isdir(userpath):
            userpath = os.path.dirname(userpath)
            self.properties.filepath = userpath
            if not os.path.isdir(userpath):
                msg = "Please select a directory not a file\n" + userpath
                self.report({'ERROR'}, msg)
                return {'CANCELLED'}
        context.scene.xxmi.dump_path = userpath
        bpy.ops.ed.undo_push(message="XXMI Tools: dump path selected")
        return{'FINISHED'}
class ExportAdvancedOperator(bpy.types.Operator):
    """Export operation base class"""
    bl_idname = "xxmi.exportadvanced"
    bl_label = "Export Mod"
    bl_description = "Export mod"
    bl_options = {'REGISTER'}
    operations = []
    def execute(self, context):
        scene = bpy.context.scene
        xxmi = scene.xxmi
        if not xxmi.dump_path:
            self.report({'ERROR'}, "Dump path not set")
            return {'CANCELLED'}
        if not xxmi.destination_path:
            self.report({'ERROR'}, "Destination path not set")
            return {'CANCELLED'}
        if xxmi.destination_path == xxmi.dump_path:
            self.report({'ERROR'}, "Destination path can not be the same as Dump path")
            return {'CANCELLED'}
        self.apply_modifiers_and_shapekeys = xxmi.apply_modifiers_and_shapekeys
        self.join_meshes = xxmi.join_meshes
        self.normalize_weights = xxmi.normalize_weights
        self.export_shapekeys = xxmi.export_shapekeys
        try:
            vb_path = os.path.join(xxmi.dump_path, ".vb0")
            ib_path = os.path.splitext(vb_path)[0] + '.ib'
            fmt_path = os.path.splitext(vb_path)[0] + '.fmt'
            object_name = os.path.splitext(os.path.basename(xxmi.dump_path))[0]
            # FIXME: ExportHelper will check for overwriting vb_path, but not ib_path
            outline_properties = (xxmi.outline_optimization, xxmi.toggle_rounding_outline, xxmi.decimal_rounding_outline, xxmi.angle_weighted, xxmi.overlapping_faces, xxmi.detect_edges, xxmi.calculate_all_faces, xxmi.nearest_edge_distance)
            game = silly_lookup(xxmi.game)
            start = time.time()
            export_3dmigoto_xxmi(self, context, object_name, vb_path, ib_path, fmt_path, xxmi.use_foldername, xxmi.ignore_hidden, xxmi.only_selected, xxmi.no_ramps, xxmi.delete_intermediate, xxmi.credit, xxmi.copy_textures, outline_properties, game, xxmi.destination_path)
            print("Export took", time.time() - start, "seconds")
            self.report({'INFO'}, "Export completed")
        except Fatal as e:
            self.report({'ERROR'}, str(e))
        return {'FINISHED'}
class ExportAdvancedBatchedOperator(bpy.types.Operator):
    """Export operation base class"""
    bl_idname = "xxmi.exportadvancedbatched"
    bl_label = "Batch export"
    bl_description = "Exports 1 mod per frame of blender timeline as a single mod. Folder names follow the pattern specified in the batch pattern"
    bl_options = {'REGISTER'}
    operations = []
    def invoke(self, context, event):
        scene = bpy.context.scene
        if bpy.app.version < (4, 1, 0):
            return context.window_manager.invoke_confirm(operator = self, event = event)
        return context.window_manager.invoke_confirm(operator = self,
            event = event,
            message = f"Exporting {scene.frame_end + 1 - scene.frame_start} copies of the mod. This may take a while. Continue?",
            title = "Batch export",
            icon = 'WARNING',
            confirm_text = "Continue")

    def execute(self, context):
        scene = bpy.context.scene
        xxmi = scene.xxmi
        start_time = time.time()
        base_dir = xxmi.destination_path
        wildcards = ("#####", "####", "###", "##", "#")
        try:
            for frame in range(scene.frame_start, scene.frame_end + 1):
                context.scene.frame_set(frame)
                for w in wildcards:
                    frame_folder = xxmi.batch_pattern.replace(w, str(frame).zfill(len(w)))
                    if frame_folder != xxmi.batch_pattern:
                        break
                else:
                    self.report({'ERROR'}, "Batch pattern must contain any number of # wildcard characters for the frame number to be written into it. Example name_### -> name_001")
                    return False
                xxmi.destination_path = os.path.join(base_dir, frame_folder)
                bpy.ops.xxmi.exportadvanced()
                print(f"Exported frame {frame + 1 - scene.frame_start}/{scene.frame_end + 1 - scene.frame_start}")
            print(f"Batch export took {time.time() - start_time} seconds")
        except Fatal as e:
            self.report({'ERROR'}, str(e))
        xxmi.destination_path = base_dir
        return {'FINISHED'}

class Preferences(bpy.types.AddonPreferences):
    """Preferences updater"""
    bl_idname = package_name
    # Addon updater preferences.

    auto_check_update: BoolProperty(
        name="Auto-check for Update",
        description="If enabled, auto-check for updates using an interval",
        default=False)

    updater_interval_months: IntProperty(
        name='Months',
        description="Number of months between checking for updates",
        default=0,
        min=0)

    updater_interval_days: IntProperty(
        name='Days',
        description="Number of days between checking for updates",
        default=7,
        min=0,
        max=31)

    updater_interval_hours: IntProperty(
        name='Hours',
        description="Number of hours between checking for updates",
        default=0,
        min=0,
        max=23)

    updater_interval_minutes: IntProperty(
        name='Minutes',
        description="Number of minutes between checking for updates",
        default=0,
        min=0,
        max=59)

    def draw(self, context):
        layout = self.layout
        print(addon_updater_ops.get_user_preferences(context))
        # Works best if a column, or even just self.layout.
        mainrow = layout.row()
        col = mainrow.column()
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

def register():
    '''Register all classes'''
    bpy.types.Scene.xxmi = bpy.props.PointerProperty(type=XXMIProperties)

def unregister():
    '''Unregister all classes'''
    del bpy.types.Scene.xxmi
