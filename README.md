# Infomap-Based Community Detection and Graph Neural Networks

## Course Information
This project was developed for the course **"INFORMATION THEORY AND INFERENCE" (2024–2025, SCQ0093479)**,  
part of the Master’s Degree in **Physics of Data** at the **University of Padova**.  
**Students involved:** Gabriele Poccianti and Alberto Schiavinato.  

---

## Project Overview
This project focuses on **community detection in complex networks** through the implementation of the **Infomap algorithm** and the **map equation framework**:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}. The main goals are:

1. **Implementation of Infomap**  
   - We develop our own implementation of the Infomap algorithm, which uses random walks and information-theoretic compression to uncover community structures in networks.  
   - Our method is benchmarked against the official Python implementation to evaluate accuracy, scalability, and efficiency.

2. **Experiments on Synthetic and Real Networks**  
   - We test both implementations on a series of **synthetic networks** with known community structures.  
   - We also analyze real-world data from the [Social Connectivity Index (SCI)](https://data.humdata.org/dataset/social-connectedness-index), focusing on networks of **U.S. states divided by counties**, as provided by Meta.  
   - This allows us to investigate how community detection captures social and geographic clustering patterns.

3. **Graph Neural Network (GNN) for Infomap Clustering**  
   - Beyond direct algorithmic detection, we design and train a **Graph Neural Network (GNN) architecture** to learn and reproduce the community structures identified by Infomap.  
   - This approach bridges classical community detection and deep learning on graphs, opening avenues for scalable and adaptive clustering in large-scale networks.

---

## Key Contributions
- A **from-scratch implementation** of Infomap using the map equation.  
- A **comparative study** with the existing Python Infomap library on both synthetic benchmarks and SCI data.  
- Development of a **GNN-based clustering model** that learns to approximate Infomap partitions.

---

## References
- Rosvall, M., & Bergstrom, C. T. (2008). *Maps of random walks on complex networks reveal community structure.* PNAS, 105(4), 1118–1123:contentReference[oaicite:2]{index=2}.  
- Rosvall, M., & Bergstrom, C. T. (2008). *Supporting Information: The map equation and algorithmic implementation.*:contentReference[oaicite:3]{index=3}  
