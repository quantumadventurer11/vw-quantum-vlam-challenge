"""MuJoCo bridge: maps compressed OpenVLA-7B action outputs to humanoid joints.

Loads dm_control humanoid, takes 7-DoF end-effector delta actions from OpenVLA,
maps them to the right arm DoF subset, and runs physics rollouts.
"""
# Implementation added in Phase 4.
