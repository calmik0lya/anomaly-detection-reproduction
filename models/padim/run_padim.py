import json

from anomalib.data import MVTecAD
from anomalib.models import Padim
from anomalib.engine import Engine

datamodule = MVTecAD(
    root="/Users/kseniakirillovna/Documents/anomaly-models/data/mvtec",
    category="bottle",
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=0,
)

model = Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"])

engine = Engine(
    accelerator="cpu",
    devices=1,
    max_epochs=1,
    default_root_dir="/Users/kseniakirillovna/Documents/anomaly-models/padim_run/results",
)

engine.fit(model=model, datamodule=datamodule)
test_results = engine.test(model=model, datamodule=datamodule)

print("=== TEST RESULTS ===")
print(json.dumps(test_results, indent=2, default=str))

with open("/Users/kseniakirillovna/Documents/anomaly-models/padim_run/metrics.json", "w") as f:
    json.dump(test_results, f, indent=2, default=str)
