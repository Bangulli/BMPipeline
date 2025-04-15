import os
import pathlib as pl
import numpy as np

def compute_sample_size(n_patients, confidence=0.80, margin_of_error=0.1, standard_deviation=0.5):
    z_values = {
        0.80: 1.28,
        0.85: 1.44,
        0.90: 1.65,
        0.95: 1.96,
        0.99: 2.58
    }
    Z = z_values[confidence]
    p=standard_deviation
    e=margin_of_error
    print(n_patients, Z, p, e)
    term1 = (Z**2*p*(1-p))/e**2
    term2 = 1+(Z**2*p*(1-p)) / (e**2*n_patients)
    return round(term1 / term2)
    

def get_samples(path):
    population = [pat for pat in os.listdir(path) if pat.startswith('sub-PAT') and (path/pat).is_dir()]
    sample_size = compute_sample_size(len(population))
    samples = np.random.choice(np.asarray(population), sample_size, replace=False)
    return samples



if __name__ == '__main__':
    conf = 0.80

    bids = pl.Path('/mnt/nas6/data/Target/BIDS_mrct1000')
    with open('sample_bids.txt', 'w') as file:
        file.write(f"QUALITYCHECK - BIDS {bids}\n")
        file.write(f"TASKS:\n")
        file.write(f"   - Check if images are valid and all converted\n")
        file.write(f"   - Check if all RTs with a valid CT link have been converted\n")
        samples = get_samples(bids)
        file.write(f"SAMPLES: {len(samples)} for a population of {len([pat for pat in os.listdir(bids) if pat.startswith('sub-PAT') and (bids/pat).is_dir()])} with a confidence of {conf}\n")
        for s in samples:
            file.write(f"   - {s}\n")


    processed = pl.Path('/mnt/nas6/data/Target/PROCESSED_mrct1000_nobatch')
    with open('sample_processed.txt', 'w') as file:
        file.write(f"QUALITYCHECK - PROCESSED {processed}\n")
        file.write(f"TASKS:\n")
        file.write(f"   - Check correct rt date to anat date matching")
        file.write(f"   - Check correct t0")
        file.write(f"   - Check mrmr registrations: qualitatively - load to itk and see if they move a lot\n")
        file.write(f"   - Check ctmr registrations: qualitatively - object matches structure in image\n")
        file.write(f"   - Check Lesion resegmentation\n")
        file.write(f"   - Check Lesion tracking\n")
        samples = get_samples(processed)
        file.write(f"SAMPLES: {len(samples)} for a population of {len([pat for pat in os.listdir(processed) if pat.startswith('sub-PAT') and (processed/pat).is_dir()])} with a confidence of {conf}\n")
        for s in samples:
            file.write(f"   - {s}\n")
