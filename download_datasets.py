import os
import urllib.request

def download_file(url, filename):
    print(f"Downloading {url} to {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Successfully downloaded {filename} ({os.path.getsize(filename)} bytes).")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

if __name__ == "__main__":
    # URLs
    rhc_url = "https://hbiostat.org/data/repo/rhc.csv"
    ihdp_url = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv"
    
    # Save filenames
    os.makedirs("data", exist_ok=True)
    
    download_file(rhc_url, os.path.join("data", "rhc.csv"))
    download_file(ihdp_url, os.path.join("data", "ihdp.csv"))
