import sys
from pathlib import Path
import os
from src.utils.local_functs import get_all_subfolders
from src.utils.cnn import load_data
from config.variables import Batch_size, Images_size

# Obtener la ruta absoluta de la carpeta que contiene el módulo
root_dir = Path(__file__).resolve().parent.parent.parent.parent.parent


# Agregar la ruta de la carpeta al sys.path
sys.path.append(root_dir)
data_dir=root_dir/'data'

train_dir=data_dir/'Training'
valid_dir=data_dir/'validation'
test_dir=data_dir/'Test'



train_loader, valid_loader,test_loader, num_classes = load_data(train_dir, 
                                                    valid_dir,test_dir, 
                                                    batch_size=Batch_size, 
                                                    img_size=Images_size) 

#classnames = train_loader.datasets.classes