import os
import addon_utils
from ..wheels.jinja2 import Environment, FileSystemLoader, Template

def generate_ini(charater_name:str, char_hash: dict, offsets: list,
                strides: list, texture_hashes_written: dict, credit: str, game, path: str, template: str):
    
    addon_path = None
    for mod in addon_utils.modules():
        if mod.bl_info['name'] == 'XXMI_Tools':
            addon_path = os.path.dirname(mod.__file__)
            break

    template_path = os.path.join(addon_path, 'templates', 'test.j2')
    with open(template_path, 'r') as r:
        template = Template(r.read())
    
    env = Environment(loader=FileSystemLoader(searchpath=addon_path)).get_template(template)
    

    return env.render(
        char_hash=char_hash,
        offsets=offsets,
        strides=strides,
        texture_hashes_written=texture_hashes_written,
        credit=credit,
        game=game,
        character_name=charater_name
        )
