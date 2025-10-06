# Data Directory Structure

This directory contains all data files for the Tourist Trip Optimizer project. All large data files are tracked using Git Large File Storage (LFS) for efficient version control.

## Directory Structure

```
data/
├── raw/              # Original, immutable data files
├── interim/          # Intermediate data that has been transformed
├── processed/        # Final, canonical data sets for modeling
└── results/          # Output from models and analysis (reports, visualizations, etc.)
```

## Subdirectories

### raw/
Contains original, immutable source data files:
- `locations.csv` - Sri Lankan tourist attractions with coordinates, ratings, and categories

### interim/
Intermediate data files generated during processing:
- Distance matrices
- Preprocessed datasets
- Temporary calculation results

### processed/
Final processed datasets ready for analysis and modeling:
- Cleaned and validated attraction data
- Calculated distance/time matrices
- Feature-engineered datasets

### results/
Model outputs and analysis results:
- Genetic Algorithm solutions (`.npy` files)
- Mixed Integer Programming solutions
- Visualization outputs (`.html` maps)
- Performance reports (`.txt` files)
- Comparison summaries

## Git LFS Configuration

All data files are tracked with Git LFS to handle large file sizes efficiently. The following file types are automatically tracked:

- CSV files: `*.csv`
- NumPy arrays: `*.npy`
- HTML outputs: `*.html` (in data directory)
- Text reports: Large `.txt` files

To ensure Git LFS is working correctly, run:
```bash
git lfs ls-files
```

## Usage Guidelines

1. **Never modify files in `raw/`** - Keep original data immutable
2. Place intermediate processing outputs in `interim/`
3. Store final datasets in `processed/`
4. Save all model outputs and visualizations in `results/`
5. Large files (>100KB) will be automatically tracked by Git LFS

## Adding New Data

When adding new data files:

1. Place them in the appropriate subdirectory
2. Git LFS will automatically track supported file types
3. Commit and push as normal - LFS handles the rest

For more information about the project, see the main README.md in the project root.
