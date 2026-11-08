# MIDOG 2025

Reproducible data plumbing, validation, exploratory analysis, and modeling
scaffolding for the two tracks of the **Mitosis Domain Generalization Challenge
2025**:

- **Track 1:** robust mitotic figure detection across clean, random, and adverse
  tissue regions.
- **Track 2:** binary classification of normal (NMF) versus atypical (AMF)
  mitotic figures across tumor, species, laboratory, stain, and scanner domains.

The repository starts with data quality because domain leakage, inconsistent
pixel spacing, and patch-level splits can dominate apparent model performance.

## Resolution: do not hard-code 0.25 µm/px

The test images are at similar magnification, but **they do not all have exactly
the same native resolution**. Official challenge documentation lists scanners at
0.22, 0.23, and 0.25 µm/px. The published MIDOG 2025 challenge analysis also
reports two final-test domains at 0.12 µm/px. Read TIFF spacing/DPI metadata per
image and explicitly resample to a model target resolution.

The reference Track 1 container converts pixel coordinates to millimetres when
resolution metadata is present. Preserve that metadata through preprocessing.

Sources: [challenge specification](https://zenodo.org/records/15077361),
[challenge paper](https://arxiv.org/abs/2606.07368), and the
[official Track 1 container](https://github.com/DeepMicroscopy/MIDOG25_T1_reference_docker).

## Repository structure

```text
.
├── configs/                 Dataset source registry
├── data/                    Ignored raw/interim/processed data
├── models/                  Ignored weights and checkpoints
├── notebooks/               Thin experiment notebooks only
├── reports/generated/       Audit HTML, plots, CSVs, JSON
├── src/midog2025/
│   ├── audit.py             Integrity, leakage, label, domain, image audit
│   ├── download.py          Resumable and checksum-aware dataset wiring
│   └── paths.py             Portable path configuration
├── tests/
├── Makefile
└── pyproject.toml
```

## Quick start

Python 3.10+ is supported.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

python -m pip install -e ".[dev]"
```

Download Track 2 (corrected CSV + 426 MB patch archive) and Track 1 metadata:

```bash
midog-download --track all-metadata
```

Download the full Track 1 MIDOG++ image set. This is approximately **65 GB**
and is deliberately opt-in:

```bash
midog-download --track track1-images --i-understand-size --workers 8
```

Downloads use `.part` files and resume when the upstream server supports byte
ranges. Existing zero-byte files are rejected instead of being mistaken for
completed images. Track 2 files are checked against the upstream MD5 digests.

Generate the full audit:

```bash
midog-audit --workers 8
```

Run the resource-bounded candidate-level pilot (32 GB RAM, CPU-safe defaults):

```bash
midog-pilot
```

This trains a lightweight verifier and normal/atypical classifier, selects
thresholds on specimen-disjoint validation data, and evaluates the hierarchical
background/normal/atypical path on held-out specimens. It opens Track 1 TIFFs
sequentially and caps all sample counts in `configs/pilot.toml`. The pilot uses
annotation coordinates as proposals, so it validates data flow and modeling
sanity—not whole-slide proposal detection or final challenge performance.

Open `reports/generated/data_audit_report.html`. Machine-readable companion
files include:

- `audit_summary.json`
- `data_quality_issues.csv`
- `track1_image_inventory.csv`
- `track1_tiff_inventory.csv` (when TIFFs are available)
- `track2_image_inventory.csv`
- `track2_duplicate_image_ids.csv`
- `track2_recommended_split.csv` (deterministic, WSI-grouped, domain-stratified)

## Audit coverage

The audit checks:

- CSV schema, missingness, invalid labels, and CSV ↔ file membership;
- duplicate IDs, duplicate source coordinates, exact SHA-256 duplicates, and
  average-hash collision candidates;
- PNG/TIFF decode integrity, dimensions, mode, file size, RGB/tissue summaries,
  and TIFF resolution tags;
- strict majority consistency and pairwise expert agreement/Cohen's κ;
- class imbalance across tumor, scanner, laboratory, and species domains;
- COCO referential integrity, category validity, annotation vote coverage, and
  bounding-box geometry/bounds;
- cross-track coordinate overlap and WSI-level leakage risk.

Important: `MIDOGpp.json` stores boxes as `[x0, y0, x1, y1]`, although the field
is called `bbox`. It is not the standard COCO `[x, y, width, height]` convention.

## Split policy

Never make a random patch split. At minimum, group by the source TIFF
`filename`; for domain-generalization experiments, hold out complete domains.
Use one shared split manifest for both tracks because Track 2 patches are
derived from Track 1 mitosis coordinates.

A recommended experiment ladder is:

1. grouped WSI-level cross-validation;
2. leave-one-domain-out validation;
3. calibration/threshold selection on a group-disjoint validation set;
4. a final frozen Docker inference path matching the Grand Challenge contract.

For Track 2, report balanced accuracy, AUROC, sensitivity, specificity, F1, and
calibration—not accuracy alone. For Track 1, report F1 at the challenge matching
radius plus average precision, both overall and per domain/tissue-region type.

## Submission contracts

The official process packages inference in Docker and uploads a `tar.gz` image
to Grand Challenge. Start from the organizer templates rather than inventing the
I/O contract:

- [Track 1 reference Docker](https://github.com/DeepMicroscopy/MIDOG25_T1_reference_docker)
- [Track 2 reference Docker](https://github.com/DeepMicroscopy/MIDOG25_T2_reference_docker)
- [Submission guide](https://midog2025.deepmicroscopy.org/submitting/)

Track 1 writes `/output/mitotic-figures.json`; coordinates are physical when
spacing metadata is available. Track 2 consumes stacked ROI cropouts. Keep model
code separate from the adapter that implements these paths and schemas.

## Data and citations

Raw data is not committed. Review the upstream licenses and cite each dataset
used. Primary resources:

- [MIDOG 2025 datasets](https://midog2025.deepmicroscopy.org/datasets/)
- [MIDOG++](https://github.com/DeepMicroscopy/MIDOGpp)
- [MIDOG 2025 challenge site](https://midog2025.grand-challenge.org/)
- [Official getting-started guide](https://github.com/DeepMicroscopy/MIDOG_2025_Guide)

The code in this repository is MIT licensed; dataset licenses remain upstream.
