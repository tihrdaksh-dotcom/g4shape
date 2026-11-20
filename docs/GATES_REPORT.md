# GATES REPORT: Implementation Plan

This document describes the implementation plan for both gates and the progress made.

## Gate 1: Protocol Parity (Data validation)

Requirements:
- Score reference models (G4mismatch K+ and KPDS, TransG4 K) using our encoding 
- Validate that our implementation reproduces published results within ±0.01
- Store predictions for later comparison with our model

### Steps completed so far:
1. Created `scripts/score_reference.py` - base script structure for scoring reference models 
2. Created `src/g4shape/analysis/evaluate_rows.py` - evaluation framework 
3. Implemented core metrics functions: pearson_r, spearman_r, mae, rmse, r2

## Gate 2: Input Fidelity and Learning Curves

Requirements:
- Add new model architectures with input improvements:
  * Lattice tokenizer (tokenization of G-runs)
  * Learned local nucleotide encoder 
  * Position only at readout
- Run learning curves on PDS data at multiple training fractions
- K+ replication with same approach

### Steps needed:
1. Implement lattice tokenizer feature 
2. Add learned nucleotide encoder 
3. Add position-at-readout feature
4. Create new model configurations for the improved model variants
5. Implement learning curve evaluation script
6. Set up full pipeline testing and validation

## Current Status:
- Basic framework is in place - functions work, test modules set up
- Need to complete:
  * TF environment setup for reference scoring  
  * Reference model scoring capability for G4mismatch and TransG4
  * Gate 2 implementations with new model features

## Next steps:
1. Implement the actual data loading and feature extraction functions needed for scoring reference models
2. Integrate TensorFlow-based reference scoring capabilities 
3. Start implementing the lattice tokenizer and related feature improvements for Gate 2

This plan will be implemented through multiple commits following the teaching mode.