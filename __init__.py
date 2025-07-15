#!/usr/bin/env python3
import sys
from pathlib import Path
from . import auto_load

sys.path.insert(0, str(Path(__file__).parent / "libs"))
# Original plugin by DarkStarSword (https://github.com/DarkStarSword/3d-fixes/blob/master/blender_3dmigoto.py)
# Updated to support 3.0 by MicroKnightmare from the DOA modding discord

####### AGMG Discord Contributors #######
# Modified by SilentNightSound#7430 to add Genshin support and some more Genshin-specific features
# QOL feature (ignoring hidden meshes while exporting) added by HazrateGolabi#1364
# HummyR#8131 created optimized outline algorithm for Genshin meshes
# merged several iterations of this plugin for other games back into a single one by LeoTorreZ
# Testing and developing of modern features by SinsOfSeven
# Added support for WUWA and more formal implementation of its classes by SpectrumQT

bl_info = {
    "name": "XXMI_Tools",
    "blender": (2, 93, 0),
    "author": "LeoTorreZ",
    "location": "File > Import-Export",
    "description": "Imports meshes dumped with 3DMigoto's frame analysis and exports meshes suitable for re-injection.     Author of original plugin: DarkStarSword.    Contributors: SilentNightSound#7430, HazrateGolabi#1364, HummyR#8131, SinsOfSeven, SpectrumQT ",
    "category": "Import-Export",
    "tracker_url": "https://github.com/leotorrez/XXMITools",
    "version": (1, 5, 8),
}
auto_load.init()


def register():
    auto_load.register()


def unregister():
    auto_load.unregister()
