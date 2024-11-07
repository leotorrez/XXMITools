"""Generates an ini file from a template file using Jinja2."""
import os
import addon_utils
from ..wheels.jinja2 import Environment, FileSystemLoader, Template
from .. import bl_info

def generate_ini(character_name:str, char_hash: dict, offsets: list, texture_hashes_written: dict, credit: str,
                game, templates_paths: list[str] = None, template_name: str = "default.ini"):
    """Generates an ini file from a template file using Jinja2.
    Trailing spaces are removed from the template file.
    Args:
        version (tuple): _description_
        character_name (str): _description_
        char_hash (dict): _description_
        offsets (list): _description_
        strides (list): _description_
        texture_hashes_written (dict): _description_
        credit (str): _description_
        game (_type_): _description_
        path (str, optional): _description_. Defaults to None.
        template_name (str, optional): _description_. Defaults to "default.j2".

    Returns:
        _type_: _description_
    """
    addon_path = None
    for mod in addon_utils.modules():
        if mod.bl_info['name'] == 'XXMI_Tools':
            addon_path = os.path.dirname(mod.__file__)
            break
    main_path = os.path.join(addon_path, "templates")
    templates_paths.insert(0, main_path)

    env = Environment(loader=FileSystemLoader(searchpath=templates_paths), trim_blocks=True, lstrip_blocks=True)
    template = env.get_template(template_name)

    return template.render(
        version=bl_info['version'],
        char_hash=char_hash,
        offsets=offsets,
        texture_hashes_written=texture_hashes_written,
        credit=credit,
        game=game,
        character_name=character_name
        )
