import itertools
import logging
from collections.abc import Iterable

import numpy as np

import gunpowder as gp
import funlib.persistence

logger = logging.getLogger(__name__)


class PresetBalanceLabels(gp.BatchFilter):
    """Creates a scale array to balance the loss between class labels.

    Note that this only balances loss weights per-batch and does not accumulate
    statistics about class balance across batches.

    Args:

        labels (:class:`ArrayKey`):

            An array containing binary or integer labels.

        scales (:class:`ArrayKey`):

            A array with scales to be created. This new array will have the
            same ROI and resolution as ``labels``.
        
        weights (``list`` of ``float``):
            A list of weights, one per class. The length of the list must be
            equal to ``num_classes``. The weight at index ``i`` will be used
            to scale the error for voxels with label ``i``.

        mask (:class:`ArrayKey`, optional):

            An optional mask (or list of masks) to consider for balancing.
            Every voxel marked with a 0 will not contribute to the scaling and
            will have a scale of 0 in ``scales``.

        slab (``tuple`` of ``int``, optional):

            A shape specification to perform the balancing in slabs of this
            size. -1 can be used to refer to the actual size of the label
            array. For example, a slab of::

                (2, -1, -1, -1)

            will perform the balancing for every each slice ``[0:2,:]``,
            ``[2:4,:]``, ... individually.

        num_classes(``int``, optional):

            The number of classes. Labels will be expected to be in the
            interval [0, ``num_classes``). Defaults to 2 for binary
            classification.

    """

    def __init__(
        self,
        labels,
        scales,
        weights,
        mask=None,
        slab=None,
        num_classes=2,
    ):
        self.labels = labels
        self.scales = scales
        if mask is None:
            self.masks = []
        elif not isinstance(mask, Iterable):
            self.masks = [mask]
        else:
            self.masks = mask

        self.slab = slab
        self.num_classes = num_classes
        self.weights = np.array(weights)

    def setup(self):
        assert self.labels in self.spec, (
            "Asked to balance labels %s, which are not provided." % self.labels
        )

        for mask in self.masks:
            assert mask in self.spec, (
                "Asked to apply mask %s to balance labels, but mask is not "
                "provided." % mask
            )

        spec = self.spec[self.labels].copy()
        spec.dtype = np.float32
        self.provides(self.scales, spec)
        self.enable_autoskip()

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.labels] = request[self.scales]
        for mask in self.masks:
            deps[mask] = request[self.scales]
        return deps

    def process(self, batch, request):
        labels = batch.arrays[self.labels]

        assert len(np.unique(labels.data)) <= self.num_classes, (
            "Found more unique labels than classes in %s." % self.labels
        )
        assert 0 <= np.min(labels.data) < self.num_classes, (
            "Labels %s are not in [0, num_classes)." % self.labels
        )
        assert 0 <= np.max(labels.data) < self.num_classes, (
            "Labels %s are not in [0, num_classes)." % self.labels
        )
        
        # initialize error scale with 1s
        error_scale = np.ones(labels.data.shape, dtype=np.float32)

        # set error_scale to 0 in masked-out areas
        for key in self.masks:
            mask = batch.arrays[key]
            assert (
                labels.data.shape == mask.data.shape
            ), "Shape of mask %s %s does not match %s %s" % (
                mask,
                mask.data.shape,
                self.labels,
                labels.data.shape,
            )
            error_scale *= mask.data

        if not self.slab:
            slab = error_scale.shape
        else:
            # slab with -1 replaced by shape
            slab = tuple(
                m if s == -1 else s for m, s in zip(error_scale.shape, self.slab)
            )

        slab_ranges = (range(0, m, s) for m, s in zip(error_scale.shape, slab))

        for start in itertools.product(*slab_ranges):
            slices = tuple(
                slice(start[d], start[d] + slab[d]) for d in range(len(slab))
            )
            self.__balance(labels.data[slices], error_scale[slices])

        spec = self.spec[self.scales].copy()
        spec.roi = labels.spec.roi

        outputs = gp.Batch()
        outputs[self.scales] = gp.Array(error_scale, spec)
        return outputs

    def __balance(self, labels, scale):
        labels = labels.astype(np.int64)
        scale *= np.take(self.weights, labels)
        
if __name__ == "__main__":
    np_arr = np.random.randint(1,5, size=(20,20)).astype(np.uint8)
    np_arr_mask = np.random.randint(0, 2, size=(20,20)).astype(np.uint8)
    np_arr[np_arr==3] = 4
    arr = funlib.persistence.arrays.array.Array(np_arr)
    arr_mask = funlib.persistence.arrays.array.Array(np_arr_mask)
    srcs= (gp.ArraySource(gp.ArrayKey("LABELS"), arr), gp.ArraySource(gp.ArrayKey("MASK"), arr_mask))
    pipeline = srcs + gp.MergeProvider()
    pipeline += PresetBalanceLabels(
        labels=gp.ArrayKey("LABELS"),
        scales=gp.ArrayKey("SCALES"),
        weights=[0, 0.1,0.2,0.3,0.4],
        mask=gp.ArrayKey("MASK"),
        num_classes=5,
        slab=(1,-1,-1,-1)
    )
    request = gp.BatchRequest()
    request.add(gp.ArrayKey("SCALES"), (10,10))
    request.add(gp.ArrayKey("LABELS"), (10,10))
    request.add(gp.ArrayKey("MASK"), (10,10))
    with gp.build(pipeline):
        batch = pipeline.request_batch(request)
        scales = batch.arrays[gp.ArrayKey("SCALES")].data
        labels = batch.arrays[gp.ArrayKey("LABELS")].data
        mask =batch.arrays[gp.ArrayKey("MASK")].data
        print(scales)
        print(labels)
        print(mask)
        print(scales[np.logical_not(mask)])
        print(scales[np.logical_and(labels==1, mask)])
        print(scales[np.logical_and(labels==2, mask)])
        print(scales[np.logical_and(labels==3, mask)])
        print(scales[np.logical_and(labels==4, mask)])
