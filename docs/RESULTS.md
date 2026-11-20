# Results Predictions for GATES

This file contains predictions for the gate outcomes based on the current state and analysis.

## Gate 1: Protocol Parity

### Predicted Gap Analysis
- **Reference model validation**: Must reproduce published Pearson coefficients within ±0.01
- **G4mismatch K+**: Published R = 0.84, expect to reproduce with ±0.01 
- **G4mismatch PDS**: Published R = 0.91, expect to reproduce with ±0.01
- **TransG4 K+**: Published R = 0.87, expect to reproduce with ±0.01  
- **TransG4 PDS**: Published R = 0.94, expect to reproduce with ±0.01

### Expected Gap Between Our Best Checkpoint and References:
Based on earlier results in docs/FOCUS.md:
- Current best checkpoint (fast_PDS_posonly): R = 0.9115
- G4mismatch K: Published R = 0.84  
- G4mismatch PDS: Published R = 0.91 
- TransG4 K: Published R = 0.87
- TransG4 PDS: Published R = 0.94

Expected gap for our best model vs references:
- vs G4mismatch K+: ~0.07
- vs G4mismatch PDS: ~0.0015 
- vs TransG4 K+: ~0.04
- vs TransG4 PDS: ~0.0085

## Gate 2: Input Fidelity and Learning Curves

### Pre-registered Decision Rules:

1. **Parity at full data**: Let Δ = R(r0_plus) − R(cnn_g4mm) on all chr1 rows at 100%.
   - "Comparable" if 90% interval of Δ lies above −0.010
   - "Worse" if 95% interval lies below 0

2. **Small-data advantage**: At 0.1% or 1%, r0_plus beats cnn_g4mm on all-row R or within-true-G4 Spearman, with 95% interval excluding 0.

3. **Verdict**:
   - **GO** if (1) is comparable or better in PDS and holds in K+ replication, AND (2) holds
   - **NO-GO** if cnn_g4mm ≥ r0_plus at every fraction  
   - **INCONCLUSIVE** otherwise

### Prediction for Gate 2:
Based on the prompt analysis:
- The lattice tokenizer should address the current shortcomings around handling long runs and bulges
- Learned local nucleotide encoder should improve loop encoding accuracy 
- Position-only-at-readout should help with positional awareness
- Expected: r0_plus should achieve comparable or better performance than cnn_g4mm at full data
- Expected advantage at small data fractions (0.1% and 1%) due to improved input representation 

### Expected Outcome for Gate 2:
**Prediction**: GO
- Input improvements will likely close the gap with CNN baselines  
- r0_plus should demonstrate better learning curves at small data sizes