# Tourist Trip Optimizer

## About the Project
This project solves the Tourist Trip Design Problem (TTDP) for Sri Lanka, generating optimal multi-day itineraries using real attraction data. It compares a fast Genetic Algorithm (GA) and an exact Mixed-Integer Programming (MIP) approach, providing practical tools and visualizations for itinerary planning.

## Project Structure
```
tourist-trip-optimizer/
├── data/            # Datasets (raw, processed)
├── notebooks/       # Jupyter notebooks (step-by-step workflow)
├── results/         # Output results (GA, MIP, summary)
├── src/             # Core Python modules (GA, MIP, utils)
├── requirements.txt # Dependencies
└── README.md        # Project info
```

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the demonstration notebook:
   - `notebooks/ttdp_demo.ipynb`

## Summary of Findings
- GA produces near-optimal itineraries in seconds, scalable to large datasets.
- MIP guarantees optimality but is practical only for small datasets (minutes).
- Both methods respect all constraints and use real Sri Lankan POI data.
- Visualizations and summary reports are generated in the `results/` folder.

## License
MIT
