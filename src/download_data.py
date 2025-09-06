import os
import requests
import pandas as pd
from tqdm import tqdm
import gzip
import shutil

"""
Download and organize GeoLocator-DP Zenodo records.

- Downloads measurements.csv(.gz) and tags.csv(.gz) for each record in the geolocator-dp Zenodo community.
- Organizes files in data/{conceptrecid}/{id}/ for each record and version.
- Only downloads new or updated versions, removing old ones.
- Filters measurements to keep only 'pressure' and 'activity' sensors.
- Shows progress bars for downloads.

Usage:
    python download_data.py

Requirements:
    - requests
    - pandas
    - tqdm
    - gzip, shutil (standard library)

See also:
    https://raphaelnussbaumer.com/GeoLocator-DP/core/measurements/
    https://zenodo.org/communities/geolocator-dp/records
"""

ZENODO_API = "https://zenodo.org/api/records/?communities=geolocator-dp"

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

dtype_dict = {
    "tag_id": str,
    "sensor": str,
    "datetime": str,
    "value": float,
    "label": str,
}


def download_file(url, dest):
    """Download a file from a URL to a destination with a progress bar."""
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        total_size = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {os.path.basename(dest)}",
                dynamic_ncols=True,
                leave=False,
            ) as pbar:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    return False


def extract_gz(src, dest):
    """Extract a .gz compressed file to a destination."""
    with gzip.open(src, "rb") as f_in:
        with open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def find_file(files, base):
    """Find a file in the Zenodo record file list matching base name (csv or csv.gz)."""
    for f in files:
        if f["key"].endswith(f"{base}.csv") or f["key"].endswith(f"{base}.csv.gz"):
            return f
    return None


def prepare_version_folder(conceptrecid, id):
    """Create and clean the version directory for a record, removing old versions."""
    rec_dir = os.path.join(DATA_DIR, str(conceptrecid))
    version_dir = os.path.join(rec_dir, str(id))
    # Remove all other version folders for this conceptrecid
    if os.path.exists(rec_dir):
        for v in os.listdir(rec_dir):
            v_path = os.path.join(rec_dir, v)
            if v != str(id) and os.path.isdir(v_path):
                shutil.rmtree(v_path)
    os.makedirs(version_dir, exist_ok=True)
    return version_dir


def process_measurements(mfile, version_dir):
    """Download, extract, filter, and save measurements.csv(.gz) for a record version."""
    fname = mfile["key"]
    url = mfile["links"]["self"]
    out_path = os.path.join(version_dir, fname)
    filtered_path = out_path.replace(".gz", "").replace(".csv", "_filtered.csv")
    # Only download if not present
    if os.path.exists(filtered_path):
        print(f"Already have latest measurements for: {out_path}")
        return
    file_type = "csv.gz" if fname.endswith(".gz") else "csv"
    file_size = mfile.get("size", None)
    size_str = f"{file_size/1024/1024:.2f} MB" if file_size else "Unknown size"
    print(f"Processing measurements file: {fname} ({file_type}, {size_str})")

    if fname.endswith(".gz"):
        gz_path = out_path
        csv_path = out_path[:-3]
        if not os.path.exists(csv_path):
            if download_file(url, gz_path):
                extract_gz(gz_path, csv_path)
                os.remove(gz_path)
        df = pd.read_csv(csv_path, dtype=dtype_dict, low_memory=False)
    else:
        if not os.path.exists(out_path):
            download_file(url, out_path)
        df = pd.read_csv(out_path, dtype=dtype_dict, low_memory=False)
    df = df[df["sensor"].isin(["pressure", "activity"])]
    df.to_csv(filtered_path, index=False)


def process_tags(tfile, version_dir):
    """Download and extract tags.csv(.gz) for a record version if available."""
    fname = tfile["key"]
    url = tfile["links"]["self"]
    out_path = os.path.join(version_dir, fname)
    # Only download if not present
    if os.path.exists(out_path):
        print(f"Already have latest tags for: {out_path}")
        return
    file_type = "csv.gz" if fname.endswith(".gz") else "csv"
    file_size = tfile.get("size", None)
    size_str = f"{file_size/1024/1024:.2f} MB" if file_size else "Unknown size"
    print(f"Processing tags file: {fname} ({file_type}, {size_str})")
    if fname.endswith(".gz"):
        gz_path = out_path
        csv_path = out_path[:-3]
        if not os.path.exists(csv_path):
            if download_file(url, gz_path):
                extract_gz(gz_path, csv_path)
                os.remove(gz_path)
    else:
        if not os.path.exists(out_path):
            download_file(url, out_path)


def main():
    """Main routine: iterate over Zenodo records, process measurements and tags for each."""
    resp = requests.get(ZENODO_API).json()
    hits = resp.get("hits", {}).get("hits", [])
    for rec in hits:
        rec_title = rec.get("metadata", {}).get("title", "Unknown Record")
        conceptrecid = rec.get("conceptrecid", None)
        id = rec.get("id", None)
        if not conceptrecid or not id:
            print(f"Missing conceptrecid or id for record: {rec_title}")
            continue
        files = rec.get("files", [])
        mfile = find_file(files, "measurements")
        tfile = find_file(files, "tags")
        if not mfile:
            print(f"No measurements file found for record: {rec_title}")
            continue
        version_dir = prepare_version_folder(conceptrecid, id)
        process_measurements(mfile, version_dir)
        if tfile:
            process_tags(tfile, version_dir)
        else:
            print(f"No tags file found for record: {rec_title}")


if __name__ == "__main__":
    main()
