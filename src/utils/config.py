#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 30-Jan-2023
# 
# ---------------------------------------------------------------------------
# This file contains the base class for loading and saving experimental
# configurations.
# ---------------------------------------------------------------------------

import json

class Config(object):
    """Base class for experimental configurations
    """

    def __init__(self, settings):

        self.settings = settings

    def load_config(self, import_path):
        """Load configurations from json file

        Args:
            import_path (str): path of config.json
        """

        with open(import_path, 'r') as file:
            settings = json.load(file)

        for key, val in  settings.items():

            if isinstance(val, dict):
                for param in val.keys():
                    if isinstance(val[param], str):
                        val[param] = val[param].lower()

            elif isinstance(val, str):
                val = val.lower()

            self.settings[key] = val

    def save_config(self, export_path):
        """Save configurations to json file

        Args:
            export_path (str): path to export config.json
        """

        with open(export_path, 'w') as file:
            json.dump(self.settings, file)

    def formatted_config(self):

        formatted_json = dict()
        for key, val in  self.settings.items():
            if isinstance(val, dict):

                for param in val.keys():
                    if isinstance(val[param], dict):

                        for param1 in val[param].keys():
                            formatted_json[param1] = str(val[param][param1])

                    else:
                        formatted_json[param] = str(val[param])
            else:      
                formatted_json[key] = str(val)
        
        return formatted_json
