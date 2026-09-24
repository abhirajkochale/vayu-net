"""
Update gridsat_required_file_manifest.csv with download and validation statuses.
"""
import os
import pandas as pd

MANIFEST = "data/manifests/gridsat_required_file_manifest.csv"
RAW_DIR = "data/raw/gridsat"
INTERIM_DIR = "data/interim/gridsat"

def main():
    df = pd.read_csv(MANIFEST)
    
    download_statuses = []
    file_sizes = []
    validation_statuses = []
    validation_messages = []
    
    for _, row in df.iterrows():
        yr = row['year']
        fname = row['source_filename']
        raw_p = os.path.join(RAW_DIR, str(yr), fname)
        base = fname.replace(".v02r01.nc", "").replace("GRIDSAT-B1.", "gridsat_")
        nc_p = os.path.join(INTERIM_DIR, str(yr), f"{base}.nc")
        npz_p = os.path.join(INTERIM_DIR, str(yr), f"{base}.npz")
        
        if os.path.exists(raw_p) and os.path.getsize(raw_p) > 25 * 1024 * 1024:
            download_statuses.append("DOWNLOADED")
            file_sizes.append(os.path.getsize(raw_p))
            if os.path.exists(nc_p) and os.path.exists(npz_p):
                validation_statuses.append("PASSED")
                validation_messages.append("VALIDATED_AND_CROPPED")
            else:
                validation_statuses.append("PENDING_PROCESSING")
                validation_messages.append("RAW_DOWNLOADED")
        elif os.path.exists(nc_p) and os.path.exists(npz_p):
            download_statuses.append("STREAM_CROPPED")
            file_sizes.append(os.path.getsize(nc_p))
            validation_statuses.append("PASSED")
            validation_messages.append("STREAM_CROPPED_AND_VALIDATED")
        else:
            download_statuses.append("PENDING")
            file_sizes.append(0)
            validation_statuses.append("PENDING")
            validation_messages.append("QUEUED_FOR_ACQUISITION")
            
    df['download_status'] = download_statuses
    df['file_size_bytes'] = file_sizes
    df['validation_status'] = validation_statuses
    df['validation_message'] = validation_messages
    
    df.to_csv(MANIFEST, index=False)
    print(f"Updated {MANIFEST}:")
    print(df['download_status'].value_counts())

if __name__ == "__main__":
    main()
