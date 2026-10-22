# gMRI
Lightweight implementation of the MRI and gMRI algorithms

## Prerequisites
**gMRI** requires
* **numpy**
* **scipy**
* other standard Python3 modules (**copy**, **warnings**, ...)

## Installing
**Recommended (conda):** create the environment from `env.yml` (Python 3.12, numpy ≥1.23,<2.4, scipy ≥1.9, matplotlib), then install the package:
```bash
conda env create -f env.yml
conda activate gmri
pip install -e .   # -e: editable install, code changes take effect immediately
```

**Alternatively**, in an existing environment with numpy and scipy, from the main folder:
```bash
pip install .
```

## License
This project is licensed under the GNU GENERAL PUBLIC LICENSE license - see the [LICENSE](LICENSE) file for details.
