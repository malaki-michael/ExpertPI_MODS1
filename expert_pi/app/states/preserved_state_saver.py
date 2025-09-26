import json

actual_state = {"last_save_directory": "."}


def load(data_folder: str):
    try:
        with open(data_folder + "/state.json") as f:
            data = json.load(f)
        actual_state.update(data)
    except:
        pass


def save(data_folder: str):
    with open(data_folder + "/state.json", "w") as f:
        json.dump(actual_state, f)
