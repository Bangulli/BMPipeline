import os
import json
import csv

def get_bids(bids_output_dir):
    # List to store extracted metadata
    data = []

    # Walk through BIDS directory
    for subject in sorted(os.listdir(bids_output_dir)):
        subject_path = os.path.join(bids_output_dir, subject)
        if not os.path.isdir(subject_path) or not subject.startswith("sub-"):
            continue  # Skip non-subject directories

        for session in sorted(os.listdir(subject_path)):
            session_path = os.path.join(subject_path, session)
            if not os.path.isdir(session_path) or not session.startswith("ses-"):
                continue  # Skip non-session directories

            anat_path = os.path.join(session_path, "anat")
            if not os.path.exists(anat_path):
                continue  # Skip if no anat folder exists

            for file in sorted(os.listdir(anat_path)):
                if file.endswith(".json"):  # Only process JSON files
                    json_path = os.path.join(anat_path, file)

                    try:
                        # Read JSON file safely
                        with open(json_path, "r") as f:
                            metadata = json.load(f)
                        
                        # Ensure metadata is a dictionary
                        if not isinstance(metadata, dict):
                            print(f"Skipping {json_path} - Unexpected JSON format")
                            continue

                        # Extract relevant metadata
                        series_desc = metadata.get("SeriesDescription", "Unknown")
                        provenance = metadata.get("Provenance")

                        # Store extracted info
                        data.append([subject, session, file, series_desc, provenance])

                    except json.JSONDecodeError:
                        print(f"Skipping {json_path} - Corrupt or invalid JSON")
                    except Exception as e:
                        print(f"Error reading {json_path}: {e}")
    return data

if __name__ == '__main__':

    # Define the BIDS output directory
    bids_output_dir = "/mnt/nas6/data/Target/BIDS/filtered_subset"  # Adjust as needed

    # Output CSV file
    output_csv = "/home/lorenz/data/49 subset/real data mapping.csv"

    data = get_bids(bids_output_dir)

    # Save extracted data to a CSV file
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Patient", "Study", "JSON File", "Series", "SourceFile"])
        writer.writerows(data)

    print(f"Extraction complete! Conversion results saved to {output_csv}")
