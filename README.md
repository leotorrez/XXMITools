To use XXMI Tools most stable version you can download the latest release from the [releases page](https://github.com/leotorrez/XXMI-Tools/releases)

Then head to Blender and go to `Edit > Preferences > Add-ons` and click on `Install...` and select the downloaded file.
Make sure it is enabled after installing it.
Done! You can now use XXMI Tools in Blender.

## DEVELOPMENT VERSION

To test the development version of this blender plugin you must clone this repository

```bash
cd %appdata%\Blender Foundation\Blender\4.2\scripts\addons
gh repo clone leotorrez/XXMI-Tools
```

Then you can enable the plugin in Blender by going to `Edit > Preferences > Add-ons` and searching for `XXMI Tools`

# Planned & necessary changes:
- Faster export Numpy
    - ~~Outfit compiler Features~~
        - ~~Apply join meshes per outfit~~
        - $variable in name(its the least of pains)
    - JoinMeshes features
        - ~~Make all objects single use to temporarily remove linked data~~
        - Apply all visible shapekeys (ignoring Marked $variable)
        - Export marked $variable as data shapekeys
        - ~~Apply all visible modifiers~~
        - ~~Remove the vertex groups that contain the word MASK in their name. To avoid DX11 hard limit~~
        - ~~Join all objects in the collection+$variable into a single "container" mesh~~
        - ~~conver to numpy arrays~~
        - ~~free bmesh/undo bs~~
        - ~~concat vbs and ib concat + offset~~
            - ~~store draw slices~~
        - ~~write ini + resources into destination folder~~
- update DUMP format to add split vbuffer
    - Adding support for several texture hashes in hash.json
    - 2.8 to 3.5 support
    - update asset repo to new DUMP format
    - Shapekey support for ZZZ and HSR maybe HI3
    - Support pipiline for implicit weights (aka fix scyll collect too)
        - Fix import weights
- partial exports(blend, pos, tex, ib)
- partial export per component(meaning all materials of a single component are needed but not all components to export)
- abstracting INI generation for easier adaptation to new games

# Bugs to fix:
- report triangulate issues to dss
- recalculate normals on triangulate
- fix export only selected and ignore hidden
- double check on copy(texture that dont exist, destination)

# Would be nice to have:
- Restore resources on INI generation (textures, meshes, etc)
- Visual aid empty parented to import objects
- Mirrored mesh Boolean on import (3dm Custom property)
- "copy textures" -> "dont overwrite textures" on export
- apply texture on import(if requirements met, prolly needs some dds importer)
- WWMI added to the list as well?

# Might be good but arguable:
- incorporate in-blender collect script
    - as well as texture selection and renaming with blender UI
- WWMI like toolbox
- GIMI Tools weight handling extra buttons on weight paint mode
- Gustav0 armature tools


# Already implemented:
- ~~mesh clean up on import( merging by distance, tri to quad and delete loose)~~
- ~~adding 3.6 to 4.2 compatibility.~~
- ~~merging multibuffer support from stock dss plugin~~
- ~~merging GIMI, SRMI, HIMI, ZZMI into a single plugin~~
- ~~turn script into a zip file for easier project management~~
- ~~organize said zip xD~~
- ~~Generate stuff on proper destination~~