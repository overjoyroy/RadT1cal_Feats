# RadT1cal_Feats

T1-weighted structural MRI preprocessing and radiomic feature extraction pipeline. Takes a raw T1w NIfTI, runs it through a Nipype workflow (FSL + ANTs), registers an anatomical atlas to patient space, and outputs per-ROI volume and radiomics CSVs.

## Key Files

- [t1_preproc.py](t1_preproc.py) — main entrypoint; builds and runs the Nipype workflow
- [radiomics_helper.py](radiomics_helper.py) — extracts per-ROI binary masks, volumes, and pyradiomics features
- [Template/](Template/) — bundled MNI152 template and AAL atlas files used as defaults
- [Dockerfile](Dockerfile) — builds from `jor115/neurodocker`; installs pyradiomics, sets FSL env vars

## Pipeline Steps

1. **Reorient** — FSL `Reorient2Std`
2. **Bias correction** — FSL `FAST`
3. **Skull strip** — FSL `BET` (frac=0.5, robust)
4. **Affine registration** — FSL `FLIRT` (template → patient space)
5. **Nonlinear registration** — ANTs `SyN` (Affine + SyN, Mattes metric)
6. **Atlas warp** — ANTs `ApplyTransforms` (nearest-neighbor, atlas follows template warp)
7. **Feature extraction** — per-ROI volume (mm³) + all pyradiomics `original_*` features → two CSVs

## Usage

### Direct (requires FSL, ANTs, nipype, pyradiomics installed locally)

```bash
python3 t1_preproc.py \
  -p /path/to/bids/dataset \
  -sid sub-001 \
  -o /path/to/output \
  [-ses_id ses-01] \
  [-tem /path/to/template.nii.gz] \
  [-seg /path/to/atlas.nii.gz] \
  [--testmode]
```

### Docker (recommended)

```bash
# Pull prebuilt image
docker pull jor115/RadT1cal_Feats

# Run
docker run --rm -u $UID:$UID \
  -v /path/to/bids:/data/my_data \
  -v /path/to/output:/data/output \
  jor115/RadT1cal_Feats \
  -p /data/my_data -sid sub-001 -o /data/output
```

If the output dir is inside the data dir (e.g. `[data]/derivatives`), only mount the data dir and use the relative internal path for `-o`.

## Inputs

| Flag | Required | Description |
|------|----------|-------------|
| `-p` | Yes | BIDS parent data directory |
| `-sid` | Yes | Subject ID (e.g. `sub-001`) |
| `-o` | Yes | Output directory |
| `-ses_id` | If sessions exist | Session ID (e.g. `ses-01`) |
| `-tem` | No | Custom template (default: MNI152lin_T1_2mm_brain.nii.gz) |
| `-seg` | No | Custom atlas (default: aal2.nii.gz) |
| `--testmode` | No | Reduces ANTs iterations for fast debugging |

Input T1 must be named with `_T1w.nii.gz` or `_T1w.nii` suffix (BIDS convention). Located at `[parentDir]/[subject_id]/anat/` or `[parentDir]/[subject_id]/[session_id]/anat/`.

## Outputs

Written to `[outDir]/RadT1cal_Features/[subject_id]/anat/`:

- `*_reorient.nii.gz` — reoriented T1
- `*_nobias.nii.gz` — bias-corrected T1
- `*_brain.nii.gz` — skull-stripped T1
- `*_warpedTemplate.nii.gz` — MNI template in patient space
- `*_warpedAtlas.nii.gz` — AAL atlas in patient space
- `*_volumes.csv` — ROI volumes in mm³
- `*_radiomicsFeatures.csv` — all pyradiomics `original_*` features per ROI

## Dependencies

- Python: `nibabel`, `nipy`, `nipype`, `numpy`, `pandas`, `pyradiomics`
- External: FSL (reorient, FAST, BET, FLIRT), ANTs (Registration, ApplyTransforms)
- Docker base image: `jor115/neurodocker`

## Benchmarked Runtimes

Timed on a representative T1w scan (AALv2 atlas, 120 ROIs):

| Step | Tool | `--testmode` |
|------|------|-------------|
| Reorient | FSL Reorient2Std | ~1s |
| Bias correction | FSL FAST | ~325s |
| Skull strip | FSL BET | ~7s |
| Affine reg | FSL FLIRT | ~8s |
| Apply affine to atlas | FSL FLIRT | ~0.4s |
| Nonlinear reg | ANTs SyN | ~125s |
| Apply warp to atlas | ANTs ApplyTransforms | ~2s |
| ROI feature extraction (120 ROIs) | pyradiomics | ~185s |
| **Total** | | **~653s (~11 min)** |

`--testmode` caps ANTs at 5 iterations per level (vs. 1500/200 and 100/50/30 in full mode) — full-mode ANTs registration will be significantly longer. FAST bias correction and pyradiomics feature extraction dominate runtime regardless of mode.

## Notes

- Only the first `_T1w.nii.gz` found per subject/session is processed (multiple-run support not yet implemented)
- Default atlas is AALv2; AALv3 variants with combined thalami are included in [Template/](Template/) as alternatives
- `--testmode` reduces ANTs to 5 iterations per level — use only for debugging workflow logic
- Contact: pirc@chp.edu (Joy Roy, Rafael Ceschin — Dept. of Biomedical Informatics, Univ. of Pittsburgh / Children's Hospital of Pittsburgh)
