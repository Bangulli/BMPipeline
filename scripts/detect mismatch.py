import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np

def extract_values(start, end, df):
    pat = df.iloc[start:end, :] # get subframe for patient
    rs, cs = pat.shape

    try: #bc of inconsistent strcture of the excel
        pid = int(pat.iloc[0, 0]) # get patient id
    except:
        try:
            pid = int(pat.iloc[1, 0]) # get patient id
        except:
            print('== Skip occured at:', start, ':', end, 'interval')
            return pd.DataFrame(columns=['Patient', 'Study', 'Series']), pd.DataFrame(columns=['Patient', 'Study', 'Series']), end

    if pid < 10:
        pid = 'sub-PAT00'+str(pid)
    elif pid < 100:
        pid = 'sub-PAT0'+str(pid)
    else:
        pid = 'sub-PAT'+str(pid)

    study = None
    t1 = pd.DataFrame(columns=['Patient', 'Study', 'Series'])
    t2 = pd.DataFrame(columns=['Patient', 'Study', 'Series'])

    for r in range(rs):
        if not pd.isna(pat.iloc[r, 1]): # filter for date
            study = pat.iloc[r, 1]

        if not pd.isna(pat.iloc[r, 12]) and not pat.iloc[r, 12] == 'Pas de MPRAGE' : # filter for t1
            t1 = t1.append({'Patient': pid, 'Study': 'ses-'+study.strftime('%Y%m%d'), 'Series': pat.iloc[r, 12]}, ignore_index=True)

        if not pd.isna(pat.iloc[r, 13]) and not pat.iloc[r, 13] == 'Pas de T2': # filter for t2
            t2 = t2.append({'Patient': pid, 'Study': 'ses-'+study.strftime('%Y%m%d'), 'Series': pat.iloc[r, 13]}, ignore_index=True)

    return t1, t2, end

def check_correspondence(a, b):
    """
    Checks if rows in a are present in b, if not append to result

    returns the rows in a that are missing in b
    
    """
    a = a.applymap(str.lower)
    b = b.applymap(str.lower)
    a = a.replace(r'[\/\\\s\-_,.]', '', regex=True)
    b = b.replace(r'[\/\\\s\-_,.]', '', regex=True)
    mismatches = []
    matches = []
    for i, row1 in a.iterrows():
        found = False
        for j, row2 in b.iterrows():
            if row1.equals(row2):
                found = True
                break
        if not found:
            mismatches.append(row1)
        else:
            matches.append(row1)
    return pd.DataFrame(mismatches), pd.DataFrame(matches)


            

if __name__ == '__main__':

    annotations = pd.read_excel('/home/lorenz/data/Annotations_Matthieu_3_31.01.23.xlsx')

    # bidsmap = pd.read_csv('/home/lorenz/data/90 bidsmap results/mapping.csv')
    # bidsmap['Study'] = bidsmap['Study'].apply(lambda x: x[:12])
    # print(bidsmap)

    bidsmap = pd.read_csv('/home/lorenz/data/49 subset/real data mapping.csv')
    bidsmap = bidsmap.drop(['JSON File', 'SourceFile'], axis='columns')
    bidsmap['Study'] = bidsmap['Study'].apply(lambda x: x[:12])
    bidsmap['Series'] = bidsmap['Series'].apply(lambda x: x.split('/')[-1])
    print('Data in parsed results')
    print(bidsmap)

    # not elegant but idc
    rows, cols = annotations.shape

    onset = 0
    ofset = 0
    t1 = pd.DataFrame(columns=['Patient', 'Study', 'Series'])
    t2 = pd.DataFrame(columns=['Patient', 'Study', 'Series'])
    for row in range(rows):
        if not pd.isna(annotations.iloc[row, 5]) and (onset == ofset):
            ofset = row+1 # incremented because indexing is exclusive with the upper border 
            t1_pat, t2_pat, onset = extract_values(onset, ofset, annotations)
            t1 = pd.concat([t1, t1_pat], axis=0, ignore_index=True)
            t2 = pd.concat([t2, t2_pat], axis=0, ignore_index=True)
    t1['Series'] = t1['Series'].apply(lambda x: str(x)[6:])
    print('T1 data in annotations:')
    print(t1)
    t2['Series'] = t2['Series'].apply(lambda x: str(x)[6:])
    print('T2 data in annotations:')
    print(t2)

    missing_t1, matching_t1 = check_correspondence(t1, bidsmap)#, bidsmap) # check what is missing from the annotations
    missing_t1.to_csv('/home/lorenz/data/49 subset/t1_imgs_missing.csv', index=False)
    matching_t1.to_csv('/home/lorenz/data/49 subset/t1_imgs_matching.csv', index=False)

    missing_t2, matching_t2 = check_correspondence(t2, bidsmap) # check what is missing from the annotations
    missing_t2.to_csv('/home/lorenz/data/49 subset/t2_imgs_missing.csv', index=False)
    matching_t2.to_csv('/home/lorenz/data/49 subset/t2_imgs_matching.csv', index=False)