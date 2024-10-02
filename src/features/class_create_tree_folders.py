"""
Codigo para crear una lista de directorios en funcion de diccionaio de python
o un archivo Json
"""

import json
import os
from pathlib import Path
from typing import Tuple


class FolderArchitecture:
    """Modelo base para crear arquitecturas de directorios"""

    def __init__(self, arquitecture_tree: dict | str, root_dir: str):
        """
        __init__
        arquitecture_tree: Diccionario con la arquitectura de las carpetas
        Dtype: python Dict
        example

        "directorio_modelo_conteo":{
                    "01_Split_shape":"" ,
                    "02_Split_raster":"",
                    "03_Shapes_buffer":"",
                    '06_Temp_Images':"",
                    "07_Split_grid_raster":"",
                    "08_Split_grid_shape":"",
                    "09_Shape_grid_path":"",

                    }

        root_dir: Ruta del path donde se creara los archivos de salida
        Dtype: python.path

        Args:
            arquitecture_tree (_type_): _description_
            root_dir (_type_): _description_
        """
        self.arquitecture = arquitecture_tree
        self.arquitecture_root_dir = root_dir

    def load_tree_from_json(self, path_json_file: str):
        """load_tree_from_Json Cargar un archivo json y lo convierte en un diccionario

        Args:
            path_json_file (str): Ruta completa archivo json
        """

        if Path(path_json_file).suffix.lower() == ".json":
            with open(path_json_file, encoding="utf-8") as filejson:
                folder_arquitecture = json.loads(filejson.read())

        else:
            print(
                "[INFO] Revisar la extencion del archivo, es un diccionario o la ruta de un archivo Json"
            )

        return folder_arquitecture

    def dict_to_path(
        self, dictionary: dict, path="", listreturn=None, dict_return=None
    ) -> Tuple[list, dict]:
        """
        Funcion recursiva para convertir un diccionario en el una
        lista de rutas de carpetas apartir de las keys del diccionario
        """
        if dict_return is None:
            dict_return = {}

        if listreturn is None:
            listreturn = []

        if not isinstance(dictionary, dict):
            # print(f'{path}/{dictionary}')
            os.makedirs(os.path.join(path, dictionary), mode=0o777, exist_ok=True)
            listreturn.append(f"{path}/{dictionary}")

        else:
            for key, value in sorted(dictionary.items()):
                dict_return[key] = f"{path}/{key}"
                self.dict_to_path(
                    value,
                    (f"{path}/{key}"),
                    listreturn=listreturn,
                    dict_return=dict_return,
                )

        return listreturn, dict_return

    def create_tree_directory(self):
        """
        Función para determinar con que tipo de datos se esta
        trabajando si es un diccionario o un archivo Json
        """
        arquitectura_cargada = (
            self.arquitecture
            if isinstance(self.arquitecture, dict)
            else self.load_tree_from_json(self.arquitecture)
        )
        return self.dict_to_path(arquitectura_cargada, self.arquitecture_root_dir)


if __name__ == "__main__":
    # Funcionamiento si se ingresa  una ruta de un json
    # file_tree_arquitecture = os.path.join(
    #     "/home/bdebian/Documents/Projects/Paz_Flora/Rebase_code/architectureNA.json"
    # )
    # root_path = os.path.join('/home/bdebian/Documents/Projects/Stoke_prediccition')
    # FolderArchitecture(file_tree_arquitecture,
    #    root_path).create_tree_directory()

    # Funcionamientos si se ingresa un directorio
    path_system = {
        "references": "",
        "reports": "",
        "src": {
            "notebooks": "",
            "lib": "",
            "data": {"sql": "", "config": ""},
            "features": "",
            "models": "",
            "visualization": "",
        },
    }
    root_path = os.path.join(os.getcwd())
    print("Rutad de generacion de arquitectura de carpertas: ", root_path)
    capetas_creadas = FolderArchitecture(path_system, root_path).create_tree_directory()
