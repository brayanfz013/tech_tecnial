"""Clase recopilatoria encargar de cargar y guardar archivos en memorias"""

import json
import os
import pickle
from pathlib import Path
from typing import Union

import yaml  # type: ignore


class LoadFiles:
    """Clase recopilatoria de diferentes funciones para cargar archivos de carpetas y del sistema"""

    def read_names(self, path_dir):
        """
        Lee los archivos de una carpeta y los reportana ordenados

        @path_dir:
            Info:Especificacion de la ruta de un directorio para leer archivos
            Dtype:String

        """
        nombres = []
        for item in os.listdir(path_dir):
            nombres.append(os.path.join(path_dir, item))
        return sorted(nombres)

    def load_path_names(
        self, path_dir: str, extensions: list[str]
    ) -> tuple[list[str], list[str]]:
        """
        Lee los archivos de una carpeta y retorna una lista con la ruta completa del archivo
        y otra con solo el nombre del archivo

        @path_dir:
            Info:Especificacion de la ruta de un directorio para leer archivos
            Dtype:String

        @extenciones:
            Info: Lista de string separda por comas con las extenciones que se
            desean buscar en la carpeta
            Dtype:Lista Strings ['.tiff','.jpg','.jpge','.tif','.png]

        @sample:
            Info: Retorna una lista con la ruta completa del las imagenes
            Dtype:String

        @names:
            Info: Retorna una lista unicamente con los nombres de los archivos
            Dtype: String
        """

        extensions = ["." + i for i in extensions]
        extensions = extensions + [i.upper() for i in extensions]
        extensions = sorted(set(extensions))

        data_path = Path(path_dir)

        if data_path.is_dir():
            sample = []
            names = []

            for ext in extensions:
                t_file = "**/*" + ext
                if len(list(data_path.glob(t_file))) == 0:
                    continue
                else:
                    sample.extend(list(map(str, data_path.glob(t_file))))

            sample = sorted(set(sample))

            if not sample:
                print(
                    f"\n[INFO] No hay imagenes con estas extensiones:\n\t{extensions}"
                )
                print(f"\n\tEn la ruta:\n\t{path_dir}\n")
                return [], []

            else:
                if len(sample) != 1:
                    names = [Path(p).parts[-1] for p in sample]
                    return sorted(set(sample)), sorted(set(names))
                else:
                    return sorted({sample[0]}), sorted({Path(sample[0]).parts[-1]})

        else:
            print("[INFO], la ruta no existe, favor revisar el directorio")
            return [], []

    def search_load_files_extencion(self, path_search: str, ext: list) -> dict:
        """
        This Python function searches for files with specified extensions in a given directory
        and returns the results in a dictionary categorized by extension.

        Args:
          path_search (str): The `path_search` parameter is a string that represents the
        directory path where you want to search for files with specific extensions.
          ext (list): The `ext` parameter in the `search_load_files_extencion` function is
        a list of file extensions that you want to search for in the specified directory.
        For example, if you pass `['txt','csv']` as the `ext` parameter,the function will
        search for files with

        Returns:
          The function `search_load_files_extencion` returns a dictionary where the keys are file
          extensions provided in the `ext` list parameter, and the values are lists containing
          two elements each.
        """

        # Conversion de str a tipo Path
        data_path = Path(path_search)
        sample = None

        if data_path.is_dir():
            # conversion de las extenciones a diccionario para facilitar la busqueda
            search = {i: (f".{i.upper()}", f".{i.lower()}") for i in sorted(set(ext))}

            # Diccionario sobre el cual se retornan los tipos
            sample = {i: [] for i in sorted(set(ext))}

            for key, values in search.items():
                names_file = []
                file_path = []
                for ext_ in values:
                    t_file = "**/*" + ext_
                    search_file = list(data_path.glob(t_file))
                    if search_file:
                        for found_file in search_file:
                            file_path.append(found_file.as_posix())
                            names_file.append(found_file.name.split(".")[0])
                sample[key].append(names_file)
                sample[key].append(file_path)

        else:
            print("[INFO], la ruta no existe, favor revisar el directorio")

        return sample

    def save_dic_to_txt(
        self, dict_data: dict, path_to_save: str, file_name: str
    ) -> None:
        """save_dic_to_txt Guardar un diccionario en un archivos txt

        Args:
            dict_data (dict): Diccionario el cual se desea almacenar
            path_to_save (str): Ruta en memoria donde se almacena el archivo
            file_name (str): Nombre del archivos
        """
        path_file = Path(os.path.join(path_to_save, file_name)).with_suffix(".txt")

        with open(path_file, "w", encoding="utf-8") as savefile:
            savefile.write(str(dict_data))

    def save_dict_to_json(
        self, dict_data: dict, path_to_save: str, file_name: str
    ) -> None:
        """save_dict_to_json _summary_

        Args:
            dict_data (dict):  Diccionario el cual se desea almacenar
            path_to_save (str): Ruta en memoria donde se almacena el archivo
            file_name (str): Nombre del archivos
        """

        path_file = Path(os.path.join(path_to_save, file_name)).with_suffix(".json")

        with open(path_file, "w", encoding="utf-8") as file_save:
            json.dump(dict_data, file_save)

    def is_empty(self, path):
        """
        Funcion  para determinar si una carpeta se encuentra vacia

        Si no esta vacio retorna 1, Si esta vacio retorna 0

        @path: ruta de la carpeta que se quiere comprobar si esta vacia o no

        """

        if os.path.exists(path) and not os.path.isfile(path):
            # Checking if the directory is empty or not

            if not os.listdir(path):
                print("Empty directory")
                value = 0

            else:
                print("Not empty directory")
                value = 1
        else:
            print("The path is either for a file or not valid")
            value = 0

        return value

    def load_json(self, json_file_name: str, path_to_read: str) -> dict:
        """
        The function `load_json` reads and loads a JSON file from a specified path and returns the
        data as a dictionary.

        Args:
          json_file_name (str): The `json_file_name` parameter is a string that represents
          the name of the JSON file that you want to load.
          path_to_read (str): The `path_to_read` parameter in the `load_json` function represents
          the directory path where the JSON file is located. When calling this function, you need
          to provide the path to the directory where the JSON file is stored so that the function
          can locate and load the JSON file correctly.

        Returns:
          The function `load_json` returns a dictionary containing the data loaded from the
          specified JSON file.
        """
        json_file = Path(os.path.join(path_to_read, json_file_name)).with_suffix(
            ".json"
        )
        with open(json_file, encoding="utf-8") as file:
            data: dict = json.load(file)

        return data

    def json_to_dict(self, json_file: str) -> tuple[dict, list]:
        """
        Lee los archivos de una carpeta y retorna una lista con la ruta completa del archivo
        y otra con solo el nombre del archivo

        @json:
            Info:Especificacion de la ruta de un directorio para leer archivos
            Dtype:String

        @data:
            Info: Retorna una lista con la ruta completa del las imagenes
            Dtype:String

        @nKeys_dictames:
            Info: Retorna una lista con las Keys del diccionario
            Dtype: List

        """
        # path = os.path.join(json_file)
        with open(json_file, encoding="utf-8") as file:
            data = json.load(file)

        return data, list(data.keys())

    def delete_files_in_folder(self, path_dir: Union[str, os.PathLike]) -> None:
        """delete_files_in_folder Funcion para una carpeta dada eliminar su contenido
        sin importar lo que tenga adentro

        Args:
            path_dir (Union[str,os.PathLike]): Ruta de la carpeta a la cual se le desea
            eliminar el contenido
        """
        for item in os.listdir(path_dir):
            os.remove(os.path.join(path_dir, item))

    def save_scaler(self, scaler, path):
        """Metodo para guardar el escalador para hacer predicciones futuras"""
        with open(path, "wb") as f:
            pickle.dump(scaler, f)

    def load_scaler(self, path):
        """Metodo para cargar los datos serializado del scaler"""
        with open(path, "rb") as f:
            scaler = pickle.load(f)
        return scaler

    def save_strategy(self, pickle_strategy: str, pathfile: dict):
        """save_strategy Metodo para guardar diccionario de funciones en
        pytho en este caso un diccionario con metodo para depurar datos

        Args:
            pickle_strategy (str): Diccionario con valores que son funciones
            replace = {
                        int:lambda x: int(float(x.replace(',',''))),
                        float:lambda x: float(x.replace(',',''))
                    }

            pathfile (dict): Ruta donde se guarda los datos del diccionario
        """
        # Ahora los datos están guardados en 'strategy.pkl'
        with open(pickle_strategy, "wb") as file:
            pickle.dump(pathfile, file)

    def load_strategy(self, pickle_strategy):
        """load_strategy Metodo para cargar informacion de funciones de un diccionario

        Args:
            pickle_strategy (_type_): Rutan donde se encuentra guardada la informacion

        Returns:
            _type_: archivos cargados como diccionario
        """
        # Podemos cargar los datos del archivo de esta manera
        with open(pickle_strategy, "rb") as file:
            loaded_replace = pickle.load(file)
        return loaded_replace

    def save_yaml(self, datafile: dict, savepath: str):
        """save_yaml Metodo para guardar un archivo yaml en memoria

        Args:
            datafile (dict): Datos del archivo yaml
            savepath (str): Ruta donde se guardara el archivo
        """

        with open(savepath, mode="w", encoding="utf-8") as file:
            yaml.dump(datafile, file)
