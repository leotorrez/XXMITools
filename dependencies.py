'''Dependency manager for the addon. Handles checking and installing required Python packages.'''

import subprocess
import sys
import os
import importlib
import ensurepip
import bpy
import bpy.app

DEBUG = True  # Set to False in production

class DependencyManager:
    """Manages external Python package dependencies for the addon."""

    @classmethod
    def read_requirements(cls):
        """Read and parse requirements.txt file."""
        requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
        dependencies = {}
        try:
            with open(requirements_path, 'r', encoding="UTF-8") as f:
                for line in f:
                    # Skip empty lines and comments
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Handle lines with version specifiers
                        package = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                        # For import name, use package name but handle special cases
                        import_name = package.replace('-', '_').lower()
                        dependencies[package] = import_name
        except FileNotFoundError:
            print("Warning: requirements.txt not found")
            return {}
        return dependencies

    @classmethod
    def get_python_exe(cls):
        """Get the path to Blender's Python executable."""
        print(os.path.join(sys.prefix, 'bin', 'python.exe'))

        return os.path.join(sys.prefix, 'bin', 'python.exe')

    @classmethod
    def check_dependencies(cls):
        """Check for missing dependencies listed in requirements.txt.     list: List of missing package names"""
        missing = []
        dependencies = cls.read_requirements()
        for package, module in dependencies.items():
            try:
                importlib.import_module(module)
            except ImportError:
                missing.append(package)
        return missing

    @classmethod
    def ensure_pip(cls):
        """Ensure pip is installed and up to date."""
        try:
            import pip
        except ImportError:
            python_exe = cls.get_python_exe()
            ensurepip.bootstrap()
            subprocess.check_call([python_exe, '-m', 'pip', 'install', '--upgrade', 'pip'])
            return True
        return False

    @classmethod
    def install_dependencies(cls):
        """ Install all missing dependencies using pip. """
        cls.ensure_pip()
        missing_packages = cls.check_dependencies()
        python_exe = cls.get_python_exe()

        if missing_packages:
            requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
            try:
                subprocess.check_call([
                    python_exe,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    requirements_path
                ])
                return True
            except subprocess.CalledProcessError as e:
                print(f"Failed to install dependencies: {str(e)}")
                raise
        return False

    @classmethod
    def uninstall_dependencies(cls):
        """Uninstall all dependencies listed in requirements.txt."""
        dependencies = cls.read_requirements()
        python_exe = cls.get_python_exe()
        
        for package in dependencies.keys():
            try:
                subprocess.check_call([
                    python_exe,
                    "-m",
                    "pip",
                    "uninstall",
                    "-y",
                    package
                ])
            except subprocess.CalledProcessError as e:
                print(f"Failed to uninstall {package}: {str(e)}")
        return True

    @staticmethod
    def restart_blender():
        """Restart Blender with the same file and arguments."""
        bpy.ops.wm.quit_blender('INVOKE_DEFAULT')
        blender_exe = bpy.app.binary_path
        head, _ = os.path.split(blender_exe)
        blender_launcher = os.path.join(head,"blender-launcher.exe")
        subprocess.run([blender_launcher, "-con", "--python-expr", "import bpy; bpy.ops.wm.recover_last_session()"], check=False)
        bpy.ops.wm.quit_blender()

class XXMI_OT_check_dependencies(bpy.types.Operator): #pylint: disable=invalid-name
    """Operator for checking and installing required dependencies through the UI."""

    bl_idname = "xxmi.check_dependencies"
    bl_label = "Check/Install Dependencies"
    bl_description = "Check and install required dependencies"

    def execute(self, context):
        """Execute the dependency check and installation."""
        missing = DependencyManager.check_dependencies()

        if not missing:
            self.report({'INFO'}, "All dependencies are installed!")
            return {'FINISHED'}

        try:
            if DependencyManager.install_dependencies():
                bpy.ops.xxmi.show_restart_popup('INVOKE_DEFAULT') #pylint: disable=no-member
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to install dependencies: {str(e)}")
            return {'CANCELLED'}

class XXMI_OT_uninstall_dependencies(bpy.types.Operator): #pylint: disable=invalid-name
    """Debug operator for uninstalling addon dependencies."""

    bl_idname = "xxmi.uninstall_dependencies"
    bl_label = "Uninstall Dependencies"
    bl_description = "DEBUG: Uninstall all addon dependencies"

    def execute(self, context):
        try:
            if DependencyManager.uninstall_dependencies():
                bpy.ops.xxmi.show_restart_popup('INVOKE_DEFAULT')
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to uninstall dependencies: {str(e)}")
            return {'CANCELLED'}

class XXMI_OT_show_restart_popup(bpy.types.Operator): #pylint: disable=invalid-name
    """Operator to display a popup requesting the user to restart Blender."""

    bl_idname = "xxmi.show_restart_popup"
    bl_label = "Restart Required"
    bl_description = "Show restart popup"

    def execute(self, context):
        """Execute the popup display. """
        DependencyManager.restart_blender()
        return {'FINISHED'}

    def invoke(self, context, event):
        """ Invoke the popup dialog."""
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        """Draw the popup dialog UI."""
        layout = self.layout
        layout.label(text="Dependencies installed successfully!")
        layout.label(text="Blender needs to restart to complete installation.")
        layout.label(text="Save your work before continuing.", icon='ERROR')
        # The OK button will trigger execute() which restarts Blender

class XXMI_PT_dependency_panel(bpy.types.Panel):#pylint: disable=invalid-name
    """Panel for managing addon dependencies"""
    bl_idname = "XXMI_PT_dependency_panel"
    bl_label = "Dependency Manager"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"

    def draw(self, context):
        layout = self.layout
        layout.operator(XXMI_OT_check_dependencies.bl_idname)
        # Show current dependency status
        missing = DependencyManager.check_dependencies()
        if missing:
            layout.label(text="Missing Dependencies:", icon='ERROR')
            for pkg in missing:
                layout.label(text=f"• {pkg}")
        else:
            layout.label(text="All dependencies installed", icon='CHECKMARK')
            if DEBUG:
                layout.separator()
                layout.label(text="Debug Options:")
                layout.operator(XXMI_OT_uninstall_dependencies.bl_idname, icon='TRASH')
