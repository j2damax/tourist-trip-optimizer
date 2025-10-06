# Processed Data

This directory contains preprocessed tourist attraction data ready for optimization algorithms.

## Files

### attractions.csv
Clean, augmented dataset with 953 Sri Lankan tourist attractions.

**Columns:**
- `poi_id`: Unique identifier (0-952)
- `name`: Attraction name
- `latitude`: Geographic latitude (5-10° range)
- `longitude`: Geographic longitude (79-82° range)
- `category`: Attraction category (Religious Site, Beach, Nature, Waterfall, Cultural, Hiking, General)
- `interest_score`: Synthetic interest/satisfaction score (45-95 on 100-point scale)
- `visit_duration`: Estimated visit time in hours (1.0-3.0)

**Statistics:**
- 953 tourist attractions (filtered from 2074 raw locations)
- Categories: 291 Religious Sites, 245 General, 230 Waterfalls, 79 Cultural, 53 Nature, 47 Beaches, 8 Hiking
- Interest scores: Mean 71.4, Range [45, 95]
- Visit durations: Mean 1.5 hours, Range [1.0, 3.0]

### distance_matrix.npy
Pre-calculated pairwise distances between all attractions using the Haversine formula.

**Format:** NumPy array (953×953)
**Unit:** Kilometers
**Statistics:**
- Range: 0.00 - 437.65 km
- Mean distance: 126.49 km

### travel_time_matrix.npy
Pre-calculated pairwise travel times between all attractions.

**Format:** NumPy array (953×953)
**Unit:** Hours
**Calculation:** `distance / 40` (assuming 40 km/h average speed as specified)
**Statistics:**
- Range: 0.00 - 10.94 hours
- Mean travel time: 3.16 hours

### preprocessing_summary.txt
Text summary of the preprocessing pipeline execution, including statistics and metadata.

## Data Preprocessing Pipeline

The data was generated using the three-phase preprocessing pipeline:

### Phase 1: Initial Data Cleaning and Structuring
1. **Filter tourist attractions** - Excluded mechanics, fuel stations, restaurants, and accommodations
2. **Select core columns** - Retained name, latitude, longitude
3. **Handle missing values** - Dropped rows with missing essential data
4. **Validate coordinates** - Verified Sri Lanka coordinate ranges (Lat: 5-10°, Lon: 79-82°)
5. **Remove duplicates** - Based on name and coordinates

### Phase 2: Data Augmentation (Synthetic Features)
1. **Category assignment** - Based on keyword matching in location names:
   - Religious sites: temple, vihara, kovil, church, mosque, shrine, stupa
   - Beaches: beach, bay, surf
   - Nature: national park, sanctuary, reserve, forest, wildlife
   - Waterfalls: waterfall, falls, ella
   - Cultural: fort, palace, museum, ruins, rock, fortress
   - Hiking: mountain, peak, hike, view
   - General: everything else

2. **Interest score generation** - Category-based with random variation:
   - Nature: 90 ± 5
   - Cultural: 85 ± 5
   - Religious: 80 ± 5
   - Hiking: 80 ± 5
   - Waterfall: 75 ± 5
   - Beach: 70 ± 5
   - General: 50 ± 5

3. **Visit duration generation** - Category-based:
   - Nature: 3.0 hours
   - Hiking: 3.0 hours
   - Cultural: 2.0 hours
   - Beach: 2.0 hours
   - Religious: 1.5 hours
   - Waterfall: 1.5 hours
   - General: 1.0 hour

### Phase 3: Final Preparation for Optimization
1. **Distance matrix calculation** - Using Haversine formula for geographic distances
2. **Travel time matrix conversion** - Distance / 40 km/h average speed

## Usage

### Python

```python
import pandas as pd
import numpy as np

# Load attractions
attractions = pd.read_csv('data/processed/attractions.csv')

# Load matrices
distance_matrix = np.load('data/processed/distance_matrix.npy')
travel_time_matrix = np.load('data/processed/travel_time_matrix.npy')

# Use with data_utils
from scripts.data_utils import load_attractions_data, prepare_data_for_optimization

data = load_attractions_data('data/processed/attractions.csv')
prepared = prepare_data_for_optimization(data)
```

### Jupyter Notebook

See `notebooks/00_Data_Preprocessing_Demo.ipynb` for an interactive demonstration.

## Regenerating Data

To regenerate the processed data from raw sources:

```bash
python scripts/preprocess_data.py
```

This will:
1. Load `data/raw/locations.csv`
2. Execute the three-phase preprocessing pipeline
3. Save outputs to `data/processed/`
4. Generate a summary report

## Notes

- **Random seed**: The preprocessing uses `np.random.seed(42)` for reproducible interest score generation
- **Average speed**: Travel time calculations use 40 km/h as specified in the problem requirements
- **Data source**: Raw data contains 2,074 locations; tourist attractions were filtered to 953 POIs
- **Coordinate validation**: All attractions fall within Sri Lanka's geographic boundaries

## Version

- **Generated**: 2025-10-06
- **Pipeline version**: 1.0
- **Raw data source**: `data/raw/locations.csv`
