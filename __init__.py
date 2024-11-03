#!/usr/bin/env python3
from . import auto_load

# Original plugin by DarkStarSword (https://github.com/DarkStarSword/3d-fixes/blob/master/blender_3dmigoto.py)
# Updated to support 3.0 by MicroKnightmare from the DOA modding discord

####### AGMG Discord Contributors #######

# Modified by SilentNightSound#7430 to add Genshin support and some more Genshin-specific features
# QOL feature (ignoring hidden meshes while exporting) added by HazrateGolabi#1364
# HummyR#8131
# merged several iterations of this plugin for other games back into a single one by LeoTorreZ

bl_info = {
    "name": "XXMI_Tools",
    "blender": (2, 93, 0),
    "author": "Ian Munsie (darkstarsword@gmail.com), SilentNightSound#7430, LeoTorreZ",
    "location": "File > Import-Export",
    "description": "Imports meshes dumped with 3DMigoto's frame analysis and exports meshes suitable for re-injection.",
    "category": "Import-Export",
    "tracker_url": "https://github.com/leotorrez/XXMITools",
    "version" : (1, 3, 6),
}
auto_load.init()

def register():
    auto_load.register()

def unregister():
    auto_load.unregister()
