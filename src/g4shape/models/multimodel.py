import torch.nn as nn

from g4shape.models.stages import Fusion, GlobalEncoder, GraphEncoder, Head

__all__ = ["MultiModalModel"]

_STAGES = (("graph_encoder", GraphEncoder), ("global_encoder", GlobalEncoder),
           ("fusion", Fusion), ("classifier", Head))


class MultiModalModel(nn.Module):
    """``graph_encoder`` + ``global_encoder`` -> ``fusion`` -> ``classifier`` (the head).

    Each stage must subclass its contract in ``src.models.stages`` (same ``forward``
    signature for every class of a stage, declared widths); the widths are checked here,
    at construction.
    """

    def __init__(self, graph_encoder, global_encoder, fusion, classifier):
        super().__init__()
        for (name, base), module in zip(_STAGES, (graph_encoder, global_encoder, fusion, classifier)):
            if not isinstance(module, base):
                raise TypeError(f"{name} must be a {base.__module__}.{base.__name__} subclass, "
                                f"got {type(module).__name__}")
        widths = [("graph_encoder.out_dim", graph_encoder.out_dim, "fusion.graph_dim", fusion.graph_dim),
                  ("global_encoder.out_dim", global_encoder.out_dim, "fusion.global_dim", fusion.global_dim),
                  ("fusion.out_dim", fusion.out_dim, "classifier.in_dim", classifier.in_dim)]
        bad = [f"{a}={va} != {b}={vb}" for a, va, b, vb in widths if va != vb]
        if bad:
            raise ValueError("stage widths do not fit: " + "; ".join(bad))
        self.graph_encoder = graph_encoder
        self.global_encoder = global_encoder
        self.fusion = fusion
        self.classifier = classifier

    def forward(self, data, return_details: bool = False):
        """
        Returns the head output [B, 1] (logit, or standardised score). With return_details=True
        returns (output, details), where details holds, per sample unless noted:

        * "x_graph" [B, H], "x_global" [B, H]: branch embeddings;
        * "gate" [B, H] (fusion weight of the topology branch per dimension), "fused" [B, H];
        * "penultimate" [B, H_head]: last hidden layer of the head;
        * "graph": graph-encoder details (per-edge attention, diffusion coefficients, node
          embeddings, readout weights; not per sample), "global": global-encoder details
          (topo_curve: per-cloud vectors [B, C, d]), "fusion"/"head": remaining extras.
        A stage with nothing to report contributes an empty dict.
        """
        d = return_details
        x_graph = self.graph_encoder(data.x, data.edge_index, data.edge_attr, data.batch,
                                     num_graphs=data.num_graphs, return_details=d)
        x_global = self.global_encoder(data.global_features, return_details=d)
        if not d:
            return self.classifier(self.fusion(x_graph, x_global))

        (x_graph, graph_det), (x_global, global_det) = x_graph, x_global
        fused, fusion_det = self.fusion(x_graph, x_global, return_details=True)
        output, head_det = self.classifier(fused, return_details=True)
        fusion_det, head_det = dict(fusion_det), dict(head_det)
        return output, {
            "graph": graph_det, "global": global_det,
            "x_graph": x_graph, "x_global": x_global,
            "gate": fusion_det.pop("gate", None), "fused": fused,
            "penultimate": head_det.pop("penultimate", None),
            "fusion": fusion_det, "head": head_det,
        }
