"""Generates an ini file from a template file using Jinja2."""
import os
# import bpy
# from .. import __package__ as base_package

class Fatal(Exception): pass
from jinja2 import Environment, FileSystemLoader

def generate_ini(user_paths: list[str],
                template_name: str,
                **variables) -> str:
    """Generates an ini file from a template file using Jinja2."""
    # extension_directory = bpy.utils.extension_path(base_package, path="", create=False)
    curr_path = os.path.dirname(os.path.abspath(__file__))
    templates_paths = [os.path.join(curr_path, "..", "templates")]
    if user_paths:
        templates_paths.extend(user_paths)
    env = Environment(loader=FileSystemLoader(searchpath=templates_paths),
                        trim_blocks=True, lstrip_blocks=True)
    template = env.get_template(template_name)
    for k,v in variables.items():
        print(f"{k}: {v}")
    return template.render(**variables)
