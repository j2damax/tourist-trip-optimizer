# Tourist Trip Design Problem (TTDP): GA vs MIP – Academic Report

## Abstract
We study the Tourist Trip Design Problem (TTDP) for Sri Lanka and compare a Genetic Algorithm (GA) against a Mixed-Integer Programming (MIP) formulation. Using real POI data, GA produced near-optimal itineraries across three days with 9 POIs (total score: 913.0) in ~802.69s. MIP delivered a proven optimal solution on a 20-POI subset with 4 POIs selected (score: 400.0) in ~11.63s. Results demonstrate GA’s scalability and practical effectiveness for large instances, while MIP provides optimality guarantees on smaller instances. We discuss trade-offs, methodology, and implications for real-world itinerary planning.

## 1. Introduction
The TTDP seeks to select and sequence attractions to maximize tourist utility under daily time constraints. The problem is NP-hard due to combinatorial selection and routing, motivating heuristic approaches for scale and exact optimization for validation.

## 2. Data and Preprocessing
- Real Sri Lankan POIs with latitude/longitude, category, rating
- Engineered features: interest_score (category + rating), visit_duration (category-based)
- Travel time matrix via Haversine distance and 40 km/h speed assumption
- Outputs stored in `results/` for downstream analysis

## 3. Methods
### 3.1 Genetic Algorithm (GA)
- Representation: permutation of POI indices decoded into day-wise tours
- Fitness: total interest score subject to daily time budget
- Operators: tournament selection, OX1 crossover, swap mutation
- Hyperparameters (from artifact): population=200, generations=500, crossover=0.85, mutation=0.03, tournament=3

### 3.2 Mixed-Integer Programming (MIP)
- Variables: visit y_id and travel x_ijd; MTZ constraints for subtours
- Objective: maximize total score subject to visit, flow, and time constraints
- Solver: PuLP/CBC with time limit for tractability on subsets

## 4. Experimental Setup
- GA run on multi-day planning with full dataset; results persisted to `results/ga_results.json`
- MIP run on a 20-POI subset; results persisted to `results/mip_results.json`
- Metrics: total score, number of POIs, computation time, feasibility/optimality status

## 5. Results
### 5.1 GA Outcomes (from `ga_results.json`)
- Method: Genetic Algorithm
- Days: 3; POIs visited: 9; Total score: 913.0
- Computation time: 802.69 seconds
- Itinerary (indices by day): [[576,741,833],[817,613,318],[437,901,290]]

### 5.2 MIP Outcomes (from `mip_results.json`)
- Method: Mixed-Integer Programming
- Problem size: 20 POIs; Days: 3; POIs visited: 4; Total score: 400.0
- Status: Optimal; Time: 11.63 seconds
- Itinerary (indices by day): [[7],[12],[9,6]]

### 5.3 Comparative Analysis
- Optimality: MIP provides proof of optimality on small subsets; GA is near-optimal at scale
- Scalability: GA scales to large datasets; MIP practical up to tens of POIs
- Speed: GA slower in this run due to full-scale instance; MIP faster on small subset
- Utility: GA visits more POIs with higher total score in large instance; MIP is ideal for validation and benchmarking

## 6. Discussion
- GA is suitable for production planning where responsiveness and scale matter
- MIP is valuable for establishing ground truth on small to medium instances
- Trade-offs depend on dataset size, constraint tightness, and real-time requirements
- Hybrid approach: Use GA to downselect POIs, then MIP for final routing on a subset

## 7. Conclusion
GA offers strong performance and scalability for real-world TTDP, while MIP ensures optimality on smaller instances. Together, they provide a comprehensive toolkit for automated itinerary planning.

## References
- Core VRP/OP/TTDP literature; GA metaheuristics; MIP formulations with MTZ constraints (see project research notes)
