#  CushionCatch: A Compliant Catching Mechanism for Mobile Manipulators via Combined Optimization and Learning  2025 IROS

The paper has been accepted for the 2025 IROS.

Project Page: https://cushion-catch.github.io/

arXiv: https://arxiv.org/pdf/2409.14754

# Installation

1. Create environment 

   ```bash
   conda create -n cushion_catch python=3.9
   conda activate cushion_catch
   ```

2. Install swift by https://github.com/jhavl/swift.git

   ```bash
   cd swift
   pip install -e .
   ```

3. Clone this codebase

   ```bash
   cd CushionCatch
   pip install -r requirements.txt
   ```

# Running
  ```bash
  python robot_catch.py
  ```

# Code Structure
```
|-- Cushion-Catch
    |-- Compliance_Learner  # PE-LSTM network for learning compliant trajectories from human demonstrations
    |-- Env                 # Robot and simulation environment definitions
    |-- Plot                # Scripts for plotting catching results
    |-- POC                 # Code for the POC planner
    |-- PRC                 # Code for the PRC planner
```
