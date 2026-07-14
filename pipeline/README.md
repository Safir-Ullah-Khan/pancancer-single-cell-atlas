# Pipeline scripts

Data assembly scripts (run once to build the atlas from source accessions):

- `cancerscem_downloader_final.py` - download source datasets
- `convert_to_h5ad.py`            - convert per-sample matrices to H5AD
- `04_merge_h5ad.py`              - merge into the unified atlas

These are available in the project's compute environment and are listed in
run_all.py. Add them to this folder before publishing if you wish to include
the full assembly pipeline.
