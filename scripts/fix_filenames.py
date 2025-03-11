import os
import pathlib as pl

def get_new_name(name, name_dict):
    new_name = name
    for key in list(name_dict.keys()):
        new_name = new_name.replace(key, name_dict[key])
    return new_name


if __name__ == '__main__':
    invalid_chars_map = {
        "<": "less_than",
        ">": "greater_than",
        ":": "colon",
        '"': "double_quote",
        "/": "forward_slash",
        "\\": "backslash",
        "|": "vertical_bar",
        "?": "question_mark",
        "*": "asterisk"
        }
    dir = pl.Path('/mnt/nas6/data/Target/mrct1000_nobatch')
    patients = [elem for elem in os.listdir(dir) if (dir/elem).is_dir() and elem.startswith('sub-')]
    for pat in patients:
        sessions = [elem for elem in os.listdir(dir/pat) if (dir/pat/elem).is_dir() and elem.startswith('ses-')]
        for ses in sessions:
            filenames = [elem for elem in os.listdir(dir/pat/ses) if (dir/pat/ses/elem).is_dir()]
            for name in filenames:
                new_name = get_new_name(name, invalid_chars_map)
                if new_name != name:
                    os.rename(dir/pat/ses/name, dir/pat/ses/new_name)
                    print(f"Renamed: {dir/pat/ses/name} → {dir/pat/ses/new_name}")