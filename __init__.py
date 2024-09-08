#!/usr/bin/env python3
from . import auto_load

# Original plugin by DarkStarSword (https://github.com/DarkStarSword/3d-fixes/blob/master/blender_3dmigoto.py)
# Updated to support 3.0 by MicroKnightmare from the DOA modding discord

####### AGMG Discord Contributors #######

# Modified by SilentNightSound#7430 to add Genshin support and some more Genshin-specific features
# QOL feature (ignoring hidden meshes while exporting) added by HazrateGolabi#1364
# HummyR#8131
# merged several iterations of this plugin for other games back into a single one by LeoTorreZ


# TODO:
# - Option to reduce vertices on import to simplify mesh (can be noticeably lossy)
# - Option to untesselate triangles on import?
# - Operator to generate vertex group map
# - Generate bones, using vertex groups to approximate position
#   - And maybe orientation & magnitude, but I'll have to figure out some funky
#     maths to have it follow the mesh like a cylinder
# - Test in a wider variety of games
# - Handle TANGENT better on both import & export?
bl_info = {
    "name": "XXMI_Tools",
    "blender": (2, 93, 0),
    "author": "Ian Munsie (darkstarsword@gmail.com), SilentNightSound#7430, LeoTorreZ",
    "location": "File > Import-Export",
    "description": "Imports meshes dumped with 3DMigoto's frame analysis and exports meshes suitable for re-injection.",
    "category": "Import-Export",
    "tracker_url": "https://github.com/leotorrez/XXMI-Tools",
    "version" : (1, 3, 0),
}
auto_load.init()

def register():
    auto_load.register()

def unregister():
    auto_load.unregister()
