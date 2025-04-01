import pickle
import os

def save_object(file_path, obj):
    """
    Save an object (like a preprocessor or model) to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)  

        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)

    except Exception as e:
        raise Exception(f"Error saving object: {str(e)}")

def load_object(file_path):
    """
    Load a pickled object (like a preprocessor or model) from a file.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        return obj

    except Exception as e:
        raise Exception(f"Error loading object: {str(e)}")
