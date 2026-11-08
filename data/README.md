# Data layout

Raw data is intentionally ignored by Git.

```text
data/
  raw/
    track1/
      MIDOGpp.json
      images/                       # optional; approximately 65 GB
    track2/
      MIDOG25_Atypical_Classification_Train_Set.csv
      MIDOG25_Binary_Classification_Train_Set/
  interim/
  processed/
```

Use `midog-download` (or `python -m midog2025.download`) to populate the raw
folders. The Track 1 image download is deliberately opt-in because MIDOG++ is
approximately 65 GB. Dataset licenses and citations remain those of the
upstream datasets; see the official dataset page before redistribution.
