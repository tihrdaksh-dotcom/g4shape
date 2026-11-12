The two core ideas, as working definitions

Idea 1: sequence shape. A sequence is described by its tokens (short units such as G-tracts) and how they are arranged: which tokens, in what order, which are neighbours, and at what coarse spacing. "Topology" here means the description changes only a little under small defects: a bulge or mismatch inside a token, or a loop that is a few nucleotides longer or shorter. It should change a lot when tokens are reordered or moved to a different spacing class. Tolerance is graded, not total: a defect still costs something, and the model learns how much.

Idea 2: representation. The model's internal description should be small, readable (each part can be checked against known G4 biophysics), and reusable: the same embedding, frozen, should predict other G4 properties (melting temperature, fold class).

Each component of the project is kept only if it serves one of these by a stated mechanism.
