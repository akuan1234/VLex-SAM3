from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module(force=True)
class OpenEarthMapDataset(BaseSegDataset):
    """OpenEarthMap validation protocol used by VLex-SAM3."""

    METAINFO = dict(
        classes=(
            'background', 'bareland', 'grass', 'pavement', 'road', 'tree',
            'water', 'cropland', 'building'
        ),
        palette=[
            [0, 0, 0], [128, 0, 0], [0, 255, 36], [148, 148, 148],
            [255, 255, 255], [34, 97, 38], [0, 69, 255], [75, 181, 73],
            [222, 31, 7],
        ],
    )

    def __init__(
        self,
        img_suffix='.tif',
        seg_map_suffix='.tif',
        reduce_zero_label=False,
        ignore_index=255,
        **kwargs,
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs,
        )


@DATASETS.register_module(force=True)
class LoveDADataset(BaseSegDataset):
    """LoveDA validation protocol.

    The raw no-data label is 0, while semantic labels are 1..7. The config
    therefore enables ``reduce_zero_label=True``.
    """

    METAINFO = dict(
        classes=(
            'background', 'building', 'road', 'water', 'barren', 'forest',
            'agricultural'
        ),
        palette=[
            [255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
            [159, 129, 183], [0, 255, 0], [255, 195, 128],
        ],
    )

    def __init__(
        self,
        img_suffix='.png',
        seg_map_suffix='.png',
        reduce_zero_label=True,
        ignore_index=255,
        **kwargs,
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs,
        )


@DATASETS.register_module(force=True)
class iSAIDDataset(BaseSegDataset):
    """iSAID semantic segmentation protocol."""

    METAINFO = dict(
        classes=(
            'background', 'ship', 'storage tank', 'baseball diamond',
            'tennis court', 'basketball court', 'running track', 'bridge',
            'large vehicle', 'small vehicle', 'helicopter', 'swimming pool',
            'roundabout', 'soccer field', 'plane', 'harbor'
        ),
        palette=[
            [0, 0, 0], [0, 0, 63], [0, 63, 63], [0, 63, 0],
            [0, 63, 127], [0, 63, 191], [0, 63, 255], [0, 127, 63],
            [0, 127, 127], [0, 0, 127], [0, 0, 191], [0, 0, 255],
            [0, 191, 127], [0, 127, 191], [0, 127, 255], [0, 100, 155],
        ],
    )

    def __init__(
        self,
        img_suffix='.png',
        seg_map_suffix='_instance_color_RGB.png',
        reduce_zero_label=False,
        ignore_index=255,
        **kwargs,
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs,
        )


@DATASETS.register_module(force=True)
class PotsdamDataset(BaseSegDataset):
    """ISPRS Potsdam protocol."""

    METAINFO = dict(
        classes=('road', 'building', 'grass', 'tree', 'car', 'clutter'),
        palette=[
            [255, 255, 255], [0, 0, 255], [0, 255, 255],
            [0, 255, 0], [255, 255, 0], [255, 0, 0],
        ],
    )

    def __init__(
        self,
        img_suffix='.png',
        seg_map_suffix='.png',
        reduce_zero_label=False,
        ignore_index=255,
        **kwargs,
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs,
        )


@DATASETS.register_module(force=True)
class ISPRSDataset(PotsdamDataset):
    """ISPRS Vaihingen protocol with the same six-class order as Potsdam."""


@DATASETS.register_module(force=True)
class UAVidDataset(BaseSegDataset):
    """UAVid protocol with moving/static car merged into ``car``."""

    METAINFO = dict(
        classes=('background', 'building', 'road', 'car', 'tree', 'vegetation', 'human'),
        palette=[
            [0, 0, 0], [128, 0, 0], [128, 64, 128], [192, 0, 192],
            [0, 128, 0], [128, 128, 0], [64, 64, 0],
        ],
    )

    def __init__(
        self,
        img_suffix='.png',
        seg_map_suffix='.png',
        reduce_zero_label=False,
        ignore_index=255,
        **kwargs,
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs,
        )


@DATASETS.register_module(force=True)
class UDD5Dataset(BaseSegDataset):
    """UDD5 protocol."""

    METAINFO = dict(
        classes=('vegetation', 'building', 'road', 'vehicle', 'miscellaneous surface'),
        palette=[
            [107, 142, 35], [102, 102, 156], [128, 64, 128],
            [0, 0, 142], [0, 0, 0],
        ],
    )

    def __init__(
        self,
        img_suffix='.JPG',
        seg_map_suffix='.png',
        reduce_zero_label=False,
        ignore_index=255,
        **kwargs,
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs,
        )


@DATASETS.register_module(force=True)
class VDDDataset(BaseSegDataset):
    """Vaihingen Drone Dataset protocol."""

    METAINFO = dict(
        classes=('background', 'wall', 'road', 'vegetation', 'vehicle', 'roof', 'water'),
        palette=[
            [0, 0, 0], [128, 128, 128], [128, 64, 128], [0, 128, 0],
            [0, 0, 142], [255, 0, 0], [0, 0, 255],
        ],
    )

    def __init__(
        self,
        img_suffix='.JPG',
        seg_map_suffix='.png',
        reduce_zero_label=False,
        ignore_index=255,
        **kwargs,
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs,
        )
