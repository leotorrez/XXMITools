# XXMI Tools

A Blender add-on to import and export mod files for the supported games.

## Installation

Download the latest release from [releases](https://github.com/leotorrez/XXMI-Tools/releases), in Blender and go to `Edit > Preferences > Add-ons` and click on `Install...` and select the downloaded `.zip` file.
Make sure it is enabled after installing it. Disable old versions of the plugin and restart blender to ensure the add-on is loaded correctly

Done! You can now use XXMI Tools in Blender.

## Development Version

To test the development version of this blender plugin you must clone this repository

```bash
gh repo clone leotorrez/XXMI-Tools
```

Recommended VSCode plugin: <https://marketplace.visualstudio.com/items?itemName=JacquesLucke.blender-development>
Recommended NVIM plugin: <https://github.com/b0o/blender.nvim>

Also is recommended to use `uv` to manage your virtual environment, however other equivalent methods work as well.

In the case of nvim in particular you'll have to setup the dependencies as follows:

```bash
uv sync --extras nvim
# and every time you launch your editor as follows:
uv run nvim
```
