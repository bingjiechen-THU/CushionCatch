#  CushionCatch: A Compliant Catching Mechanism for Mobile Manipulators via Combined Optimization and Learning 

The paper has been accepted for the 2025 IROS.

Project Page: https://cushion-catch.github.io/

arXiv: https://arxiv.org/pdf/2409.14754

<!-- Simulation Results -->
<h2>Simulation Results</h2>
<table>
  <tr>
    <td><img src="gifs/sim1.gif" width="400" alt="Simulation result 1"/></td>
    <td><img src="gifs/sim2.gif" width="400" alt="Simulation result 2"/></td>
    <td><img src="gifs/sim3.gif" width="400" alt="Simulation result 3"/></td>
    <td><img src="gifs/sim4.gif" width="400" alt="Simulation result 4"/></td>
  </tr>
</table>

<!-- Real-World Experiments -->
<h2>Real-World Results</h2>
<table>
  <tr>
    <td><img src="gifs/phy1.gif" width="400" alt="Physical demo 1"/></td>
    <td><img src="gifs/phy2.gif" width="400" alt="Physical demo 2"/></td>
    <td><img src="gifs/phy3.gif" width="400" alt="Physical demo 3"/></td>
    <td><img src="gifs/phy4.gif" width="400" alt="Physical demo 4"/></td>
  </tr>
</table>

<!-- Collision Case Study -->
<h2>Collision Case Study</h2>
<table style="width:100%; table-layout:fixed;">
  <tr>
    <td style="width:33.33%; text-align:center; vertical-align:top;">
      <img src="gifs/no_collision.gif"
           alt="No collision"
           style="width:320px; max-width:100%; height:auto; display:block; margin:0 auto;" />
      <div><strong>No Collision</strong><br/><small>all constraints enforced</small></div>
    </td>
    <td style="width:33.33%; text-align:center; vertical-align:top;">
      <img src="gifs/self_collision.gif"
           alt="Self collision"
           style="width:320px; max-width:100%; height:auto; display:block; margin:0 auto;" />
      <div><strong>Self-Collision</strong><br/><small>self-collision constraint not enforced</small></div>
    </td>
    <td style="width:33.33%; text-align:center; vertical-align:top;">
      <img src="gifs/ground_collision.gif"
           alt="Ground collision"
           style="width:320px; max-width:100%; height:auto; display:block; margin:0 auto;" />
      <div><strong>Ground Collision</strong><br/><small>ground-collision constraint not enforced</small></div>
    </td>
  </tr>
</table>




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
Default (with safety constraints enabled)
  ```bash
  python robot_catch.py
  ```
Disable self-collision constraint only
  ```bash
  python robot_catch.py --no-self-collision-cons
  ```
Disable ground-collision constraint only
  ```bash
  python robot_catch.py --no-ground-collision-cons
  ```

# Code Structure
```
|-- Cushion-Catch
    |-- Compliance_Learner  # PE-LSTM network for learning compliant trajectories from human demonstrations
    |-- Env                 # Robot and simulation environment definitions
    |-- Plot                # Scripts for plotting catching results
    |-- POC                 # Code for the POC planner
    |-- PRC                 # Code for the PRC planner
    |-- gifs                # gifs for readme
```
