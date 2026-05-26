"""
Adapter: GraphZSLData -> ICIS expected format.

ICIS expects a data object with:
  .train_feature, .train_label
  .test_seen_feature, .test_seen_label
  .test_unseen_feature, .test_unseen_label
  .seenclasses, .unseenclasses
  .attribute  [C, attr_dim]

Our GraphZSLData has:
  .x, .y, .train_mask, .test_seen_mask, .test_unseen_mask
  .class_semantics  [C, 384]
  .seen_classes, .unseen_classes (lists)

This adapter bridges the two.
"""

import torch


class ICISDataAdapter:
    """Wrap a GraphZSLData object to match ICIS's expected interface.

    Args:
        data: GraphZSLData from our unified loader
    """

    def __init__(self, data):
        self._data = data

        # Features split by mask
        self.train_feature = data.x[data.train_mask]
        self.train_label = data.y[data.train_mask]

        self.test_seen_feature = data.x[data.test_seen_mask]
        self.test_seen_label = data.y[data.test_seen_mask]

        self.test_unseen_feature = data.x[data.test_unseen_mask]
        self.test_unseen_label = data.y[data.test_unseen_mask]

        # Class info as tensors (ICIS expects LongTensors)
        self.seenclasses = torch.LongTensor(data.seen_classes)
        self.unseenclasses = torch.LongTensor(data.unseen_classes)

        # Semantic attributes [C, attr_dim] indexed by class ID
        # ICIS indexes as attribute[seenclasses] / attribute[unseenclasses]
        # So we need a [num_total_classes, attr_dim] tensor
        num_classes = data.num_classes
        attr_dim = data.class_semantics.size(1)
        self.attribute = torch.zeros(num_classes, attr_dim)

        # Fill in the semantics for all classes
        all_classes = sorted(data.seen_classes + data.unseen_classes +
                             (data.val_classes if hasattr(data, "val_classes")
                              and data.val_classes else []))
        for i, cls_id in enumerate(sorted(set(range(num_classes)))):
            if i < data.class_semantics.size(0):
                self.attribute[cls_id] = data.class_semantics[i]

    @property
    def feature_dim(self):
        return self.train_feature.size(1)

    @property
    def num_seen(self):
        return len(self.seenclasses)

    @property
    def num_unseen(self):
        return len(self.unseenclasses)