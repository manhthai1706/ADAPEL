import os
import ssl
import urllib.request

BASE = "data"

FILES = [
    # (subfolder, filename, url)
    ("ihdp", "ihdp.csv",
     "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv"),
    ("ihdp", "train.npz",
     "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"),
    ("ihdp", "test.npz",
     "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz"),
    ("rhc", "rhc.csv",
     "https://hbiostat.org/data/repo/rhc.csv"),
    ("lalonde", "lalonde.csv",
     "https://raw.githubusercontent.com/robjellis/lalonde/master/lalonde_data.csv"),
    ("hillstrom", "hillstrom.csv",
     "https://raw.githubusercontent.com/W-Tran/uplift-modelling/master/data/hillstrom/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"),
    ("twins", "X.csv",
     "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_X_3years_samesex.csv"),
    ("twins", "T.csv",
     "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_T_3years_samesex.csv"),
    ("twins", "Y.csv",
     "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_Y_3years_samesex.csv"),
    ("acic2016", "x.csv",
     "https://raw.githubusercontent.com/BiomedSciAI/causallib/master/causallib/datasets/data/acic_challenge_2016/x.csv"),
]

# ACIC 2016 zymu files (10 settings)
for i in range(1, 11):
    FILES.append(("acic2016", f"zymu_{i}.csv",
                  f"https://raw.githubusercontent.com/BiomedSciAI/causallib/master/"
                  f"causallib/datasets/data/acic_challenge_2016/zymu_{i}.csv"))


def download(url, path):
    print(f"  Downloading {url.split('/')[-1]}...", end=" ")
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        urllib.request.urlretrieve(url, path)
        print(f"OK ({os.path.getsize(path):,} bytes)")
    except Exception as e:
        print(f"FAILED: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print(" ADAPEL Dataset Downloader")
    print("=" * 60)
    for folder, name, url in FILES:
        dest = os.path.join(BASE, folder, name)
        if os.path.exists(dest):
            print(f"  [SKIP] {folder}/{name}")
        else:
            os.makedirs(os.path.join(BASE, folder), exist_ok=True)
            download(url, dest)
    print("=" * 60)
    print("Done.")
