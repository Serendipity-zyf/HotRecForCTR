"""
Shared type definitions for the utility modules.
"""

from colorama import Fore, Style


class ModuleCategory:
    """Module category configuration"""

    def __init__(self, name: str, directory: str, color: str = Fore.CYAN):
        self.name = name
        self.directory = directory
        self.modules = []
        self.color = color

    def __str__(self):
        return f"{self.color}{self.name}{Style.RESET_ALL} ({len(self.modules)} modules)"
