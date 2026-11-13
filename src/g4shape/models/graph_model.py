"""The network: G-run graph -> graph encoder -> LayerNorm -> head.

    x_graph = graph_encoder(x, edge_index, edge_attr, batch)   # [B, H] weighted-sum readout
    output  = classifier(LayerNorm(x_graph))                   # [B, 1] standardised score

``classifier`` is the head (the stage name used by the config, the factory and the
checkpoints). Both stages subclass their contract in ``src.models.stages``; the widths are
checked here, at construction.
"""

from __future__ import annotations

import torch.nn as nn

from g4shape.models.stages import GraphEncoder, Head

__all__ = ["GraphModel"]


class GraphModel(nn.Module):
    def __init__(self, graph_encoder: GraphEncoder, classifier: Head):
        super().__init__()
        for name, module, base in (("graph_encoder", graph_encoder, GraphEncoder),
                                   ("classifier", classifier, Head)):
            if not isinstance(module, base):
                raise TypeError(f"{name} must be a {base.__module__}.{base.__name__} subclass, "
                                f"got {type(module).__name__}")
        if graph_encoder.out_dim != classifier.in_dim:
            raise ValueError(f"stage widths do not fit: graph_encoder.out_dim="
                             f"{graph_encoder.out_dim} != classifier.in_dim={classifier.in_dim}")
        self.graph_encoder = graph_encoder
        self.norm = nn.LayerNorm(graph_encoder.out_dim)
        self.classifier = classifier

    def forward(self, data, return_details: bool = False):
        """Head output [B, 1]. With ``return_details=True`` returns ``(output, details)``:

        * ``"x_graph"`` [B, H]: the graph embedding (encoder output, before the LayerNorm);
        * ``"penultimate"`` [B, H_head]: last hidden layer of the head;
        * ``"graph"``: encoder details (per-edge attention, diffusion coefficients, node
          embeddings, readout weights; not per sample), ``"head"``: remaining head extras.
        """
        x_graph = self.graph_encoder(data.x, data.edge_index, data.edge_attr, data.batch,
                                     num_graphs=data.num_graphs, return_details=return_details)
        if not return_details:
            return self.classifier(self.norm(x_graph))
        x_graph, graph_det = x_graph
        output, head_det = self.classifier(self.norm(x_graph), return_details=True)
        head_det = dict(head_det)
        return output, {"graph": graph_det, "x_graph": x_graph,
                        "penultimate": head_det.pop("penultimate", None), "head": head_det}
