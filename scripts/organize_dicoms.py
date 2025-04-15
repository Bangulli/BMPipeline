import pydicom
import os
import pathlib as pl
import re

# Helper function to extract UID from the filename
def extract_uid_from_filename(filename):
    match = re.search(r'([A-Z]{2}(?:\.\d+){8})', filename)
    if match:
        return match.group(1), filename.split('.')[0]
    match = re.search(r'([A-Z]{2}(?:\.\d+){6})', filename)
    if match:
        return match.group(1), filename.split('.')[0]
    return None, None

def filter_invalid_chars(description, invalid_chars):
    for char in list(invalid_chars.keys()):
        description = description.replace(char, invalid_chars[char])
    return description

dir = pl.Path('/mnt/nas6/data/Target/batch_copy/organize_dicoms_test/disorganized')
invalid_chars = {':': '',  # dict of chars and their replacement.
                 ';': '',  # if there are forbidden chars in the series description they are searched and replaced by this
                 '\\': '', # expand and adjust as needed
                 '/': '',
                 '*': ''}

subjects = [d for d in os.listdir(dir) if (dir/d).is_dir()]
for sub in subjects:
    print(f'= Organizing subject {sub}')
    files = [f for f in os.listdir(dir/sub) if (dir/sub/f).is_file()]
    print(f'= found {len(files)} files')

    slice_info = {}
    for f in files:
        current_uid, t = extract_uid_from_filename(f) # extract
        dcmf = pydicom.read_file(dir/sub/f)
        slice_info[f] = [dcmf['SeriesDescription'].value, dcmf['StudyDate'].value, dcmf['StudyTime'].value]

    while slice_info:
        cur_keys = None
        cur_ref = None
        for key, value in slice_info.items():
            if cur_keys == None:
                cur_keys = [key]
                cur_ref = value
            elif value == cur_ref:
                cur_keys.append(key)

        t = cur_keys[0].split('.')[0]
        print(f'''== organizing file of type: {t}''')
        series_name = filter_invalid_chars(cur_ref[0], invalid_chars)
        study_name = f"ses-{cur_ref[1]}{cur_ref[2]}"
        print(f'=== Creating Series {series_name} on {study_name}')
        os.makedirs(dir/sub/study_name/series_name)
        print(f'==== Moving {len(cur_keys)} files')
        for associated_file in cur_keys:
            del slice_info[associated_file]
            os.rename(dir/sub/associated_file, dir/sub/study_name/series_name/associated_file)
        print(f'==== DONE!')



# for sub in subjects:
#     print(f'= Organizing subject {sub}')
#     files = [f for f in os.listdir(dir/sub) if (dir/sub/f).is_file()]
#     print(f'= found {len(files)} files')

#     processed_uids = []
#     slice_info = {}
#     for f in files:
#         current_uid, t = extract_uid_from_filename(f) # extract
#         # dcmf = pydicom.read_file(dir/sub/f)
#         # slice_info[f] = [dcmf['SeriesDescription'].value, dcmf['StudyDate'].value, dcmf['StudyTime'].value]


#         if current_uid in processed_uids: # skip if processed
#             continue
#         if current_uid: # process
#             print(current_uid)
#             print(f'''== organizing file of type: {t}''')
#             uid_associated = [f for f in files if current_uid in f]
#             ref_dcm = pydicom.read_file(dir/sub/uid_associated[0])
#             description = ref_dcm['SeriesDescription'].value
#             study_date = ref_dcm['StudyDate'].value
#             study_time = ref_dcm['StudyTime'].value
#             series_name = filter_invalid_chars(description, invalid_chars)
#             study_name = f"ses-{study_date}{study_time}"
#             print(f'=== Creating Series {series_name} on {study_name}')
#             #os.makedirs(dir/sub/study_name/series_name)
#             print(f'==== Moving {len(uid_associated)} files')
#             for associated_file in uid_associated:
#                 continue
#                 os.rename(dir/sub/associated_file, dir/sub/study_name/series_name/associated_file)
#             print(f'==== DONE!')
#             processed_uids.append(current_uid)