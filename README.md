# Gurobi with Parameter Tuning in MIP Problems

The primary goal is to see if the gurobi default parameter settings are good enough for solving MIP problems, using MIPLIB Benchmark.

03/05:
Parameter Guidelines: https://docs.gurobi.com/projects/optimizer/en/current/concepts/parameters/guidelines.html
Reference: https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html

By referring to this page, I decided to focus on three different params that might affect MIP solving: 
1. MIPFocus:[1,2,3]
2. Cutting Planes:[0,-1,1,2]
3. Presolve:[0,-1,1,2]
The reasons are: they are all mentioned specificly in part of MIP Problems of the page "Parameter Guidelines", and they seems to affect the solving results significantly.
Also here I would like to use MIPGap as Termination control. 