# Capstone-AML

**Setup for collaborators**

Brief: collaborators must have Git and Git LFS installed to receive the real dataset file `Elliptic Dataset/elliptic_txs_features.csv`. Without LFS they will see a small pointer file.

**Windows (PowerShell)**

```powershell
# Install (Chocolatey) or download installers from upstream sites
choco install git
choco install git-lfs

# Enable LFS and clone
git lfs install
git clone https://github.com/mahendra-kausik/Capstone-AML.git
cd Capstone-AML
git lfs pull

# If you already cloned and see a pointer file:
Get-Content "Elliptic Dataset/elliptic_txs_features.csv" -TotalCount 1
# If it starts with "version https://git-lfs.github.com/spec/v1":
git lfs fetch --all
git lfs checkout

# Python env
python -m venv .venv
.venv\Scripts\Activate
pip install -r requirements.txt
```

**macOS (Terminal)**

```bash
# Install Homebrew if needed, then Git + Git LFS
#/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install git
brew install git-lfs

git lfs install
git clone https://github.com/mahendra-kausik/Capstone-AML.git
cd Capstone-AML
git lfs pull

# If you already cloned and see a pointer file:
head -n 1 "Elliptic Dataset/elliptic_txs_features.csv"
# If it starts with "version https://git-lfs.github.com/spec/v1":
git lfs fetch --all
git lfs checkout

# Python env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Verify & troubleshooting**

- Check LFS environment: `git lfs env`
- List LFS-managed files: `git lfs ls-files`
- Detect pointer file: file starts with `version https://git-lfs.github.com/spec/v1`
- Quota: GitHub LFS storage/bandwidth is billed to the repo owner — downloads/pushes may fail if quota is exceeded.
- If LFS download fails or collaborators cannot use LFS, provide a dataset download (GitHub Release, Google Drive, or S3) and add the link here.

If you need, I can add a Release with the dataset or upload it to a cloud link and update this README.
