import json
import tempfile
from pathlib import Path

from bse_renderer.config_loader import load_config_bundle
from bse_renderer.domain_randomisation.batch_generator import BatchGenerator, BatchOptions


def test_batch_outputs():
    cfg_dir = Path(__file__).resolve().parents[1] / "config"
    bundle = load_config_bundle(cfg_dir)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "dataset"
        summary = BatchGenerator(
            bundle,
            BatchOptions(num_images=12, output_dir=out, image_size=64, seed=5),
        ).run()
        assert Path(summary["manifest"]).exists()
        assert Path(summary["dataset_yaml"]).exists()
        assert Path(summary["coco"]).exists()
        manifest = json.loads(Path(summary["manifest"]).read_text(encoding="utf-8"))
        assert len(manifest) == 12
        compo_pngs = list((out / "images" / "compo").rglob("*.png"))
        topo_pngs = list((out / "images" / "topo").rglob("*.png"))
        tx_pngs = list((out / "images" / "tx").rglob("*.png"))
        ty_pngs = list((out / "images" / "ty").rglob("*.png"))
        masks = list((out / "masks").rglob("*.png"))
        yolo = list((out / "labels" / "yolo").rglob("*.txt"))
        assert len(compo_pngs) == 12
        assert len(topo_pngs) == 12
        assert len(tx_pngs) == 12
        assert len(ty_pngs) == 12
        assert len(masks) == 12
        assert len(yolo) == 12
        assert "tx_u16" in manifest[0]["artifacts"]
        assert "ty_u16" in manifest[0]["artifacts"]
