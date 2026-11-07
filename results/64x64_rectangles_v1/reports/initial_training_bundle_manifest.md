# Initial training bundle manifest — 64x64_rectangles_v1

- Path: `C:\Users\ratul\OneDrive\Documents\AI_project_backup\results\64x64_rectangles_v1\64x64_rectangles_v1_kaggle_upload.zip`
- Bytes: 1936226
- SHA-256: `efe7302e786a5e8ec63ac9a309b923c093c89d80dec3834e89a5c7b220efaa07`
- Manifest counts: {'id_pilot': 60, 'training': 900, 'validation_calibration': 180}
- Upload status: not uploaded.
- OOD pilot, ID holdout, and sealed OOD test: excluded.
- NPZ handling: opaque byte copy and hash verification; no arrays loaded.

## Verification

- Packaged notebook matches the current local notebook.
- Every packaged Python source file matches the current local source inventory.
- The only packaged NPZ files are training, validation/calibration, and ID pilot.
- Filtered manifest records and all bundle member hashes were checked.

## Bundle contents

| Member | Uncompressed bytes |
|---|---:|
| `README_upload.md` | 1069 |
| `configs/adaptive_sensitivity_ablations.json` | 1446 |
| `dataset/id_pilot.npz` | 91219 |
| `dataset/manifest.json` | 2296227 |
| `dataset/training.npz` | 1253120 |
| `dataset/validation_calibration.npz` | 259317 |
| `notebooks/kaggle_experiments_64x64_rectangles.ipynb` | 22707 |
| `project_code.zip` | 187967 |
| `upload_manifest.json` | 1554 |

## Packaged Python source inventory

| Source file | Bytes | SHA-256 |
|---|---:|---|
| `adaptive/__init__.py` | 858 | `707526963f5bb1dd7352e25e9cca742b37094a878a8993f11895f913cba32fea` |
| `adaptive/ood_adaptive_seeastar.py` | 13974 | `e7a89fdf8ca850979dc42b5aa4f7e191a515a2f9ce0f6d1a3e9e3ec539b73497` |
| `algorithms/__init__.py` | 62 | `c5e3cb26844e29d6810187f0a5084eb39c730cc73f4df92c158a21acc1b6d7d9` |
| `algorithms/astar.py` | 9790 | `13e11aa3ced42b1a3ac95b95b0c5f101acec2b6ca496565939241978717d2698` |
| `algorithms/dijkstra.py` | 5482 | `ea7edd6575a19795c24daff7b72e682f1e6ee5a5f742f5f30e324ea41138e7f7` |
| `algorithms/movement.py` | 6230 | `e03d41095a2b685929ab890b66b9b1f4669264abb23599b4b394d0e4548fac37` |
| `benchmarks/__init__.py` | 139 | `08d859f61568573b5c7cfa74a874d7dbf93938c3533182288e210292e810bd34` |
| `benchmarks/__main__.py` | 3769 | `2f7df179374b7e620e2e136e5ec5f2d1fae34e6212ca47ffb7676054ca05ea5f` |
| `benchmarks/config.py` | 8431 | `16d2a2c1020d80382a1b0e7c710a8e54025c21cfc5472dd00381a18400b4c895` |
| `benchmarks/data.py` | 23524 | `9e88e1174d3992c722c5b54903d3055e72218937c32170d3683e42aa5bae2c64` |
| `benchmarks/execution.py` | 5083 | `0b94d5dc3ff017a6342a30e4854c9e748b254ea392dab543355475677a4e4aca` |
| `benchmarks/inference.py` | 8745 | `f43505516393d5e4d462b54d8c6bde70b7a42e82479c0c9ca0f1e2b8b280848a` |
| `benchmarks/neural_trace.py` | 4560 | `4f98147847155477aaaee38456d42850b9a69c46663a4d8b96ea794054389afb` |
| `benchmarks/phase_b_report.py` | 16528 | `b90a5f4a04fee2ee9245999b7ba55cf09d9b2ec4c38cc0be28b656398de7d445` |
| `benchmarks/rectangle_pilot.py` | 45017 | `73f6948fbbea050852201ed910ec93a94c85137ba402d0642f4dde5f863bef2d` |
| `benchmarks/reliability.py` | 9298 | `8ee48500004efc893161af7f175559bde2329b5ddb597e7f84f26f5896c16e7e` |
| `benchmarks/runner.py` | 69987 | `e2747340c42cacb808c52aca5a6d5ae5e1e775cccc78d15fd1c1b323551409e3` |
| `benchmarks/search_worker.py` | 11633 | `f8261457fda93e2c8037b8142059ee86e6c1cc77f098d234a258370dc6125bd2` |
| `benchmarks/statistics.py` | 6236 | `ff713fdb16673c7540806bad1512d2c0b596ad29d5fba1912d7213c21c1b7650` |
| `benchmarks/workflow.py` | 19444 | `c03d0593607104689948fbf5560e87080fc61f62837c44613f2d7726dd96262c` |
| `data/__init__.py` | 64 | `b39e631de6454400be0c461fa84f49f7eb42227c994f15306ece2e55ad0320e7` |
| `data/dataset.py` | 10499 | `c5188523bbb0d9ca66c47965ffb29ff3d06ff5c376addc1e203878d2a7c63efb` |
| `data/ground_truth.py` | 3603 | `ccfbb98336d16bd655a9c0a1291a7daf26c4817b388e6989b06b305231350ed9` |
| `data/map_generator.py` | 14362 | `560fc8f1d608183c8e9914458a3aaacaae238d11037924dd39d99e411d0b9de0` |
| `data/rectangular_dataset.py` | 34307 | `9c1a875159f7d53b8962752eced09745ecb243d2a7aa771b8a7df90fe14d0b67` |
| `experiments/__init__.py` | 39 | `ae5db848a98c5068cb0751096830fbf375d19f0f161e5424054f727a6f2af7ee` |
| `experiments/adaptive_seeastar.py` | 36956 | `1c775deeb9d9eee4c6324c0319ebfcbdb1f6e02344a3fe960df5465e48e89e41` |
| `experiments/heuristic_comparison.py` | 5108 | `535cec7bab81c7132606a580fc97144cb2e20696a752c0d41c6481de0fe6acee` |
| `experiments/ood_detection.py` | 5336 | `0d92baa50cfeab5ac06d038d1685703a65e27c606679af104e249a067911128c` |
| `experiments/prepare_kaggle_bundle.py` | 45407 | `bacd44a20ffd73e305bf2cffede96c103713289f5b47a8c078a310086b3bf729` |
| `experiments/rectangle_pipeline.py` | 38567 | `906eff22615f38c29d4bdaf4442f17622d76ac9bde768c67e212061a072210ba` |
| `experiments/run_paths.py` | 2101 | `71a595ea5a88fac620931a3954b3922e529e7a18b44e69db65ca52ef17e268b9` |
| `heuristics/__init__.py` | 374 | `818904a7976cb6ae25933da84067644eef3155b7a4bb31040010d03f924eeceb` |
| `heuristics/cea.py` | 3764 | `44d38c9e0af137a329510b465e79b1d0d1f434162ac4e72dd407a3c678b23666` |
| `heuristics/data.py` | 6058 | `dc60d61ed388e0cee8bb01be25ccfdde561bed2afa2d337c2439b8b2ffdaf605` |
| `heuristics/evaluation.py` | 7913 | `637c0475e5b60df578d984de7222ed06e131bde7ab8aad1c4812355af12cbab3` |
| `heuristics/inference.py` | 4157 | `074c91b94374aaaa7e0ba650fb29d819743859d96f7ba602572f407b2c8c625f` |
| `heuristics/losses.py` | 4631 | `304e84d24b1d93a326b28dad24b83daa73d45165882d9bfaf9d881fe21ca4e0e` |
| `models/__init__.py` | 194 | `c12dd6c64d1898a0b8c2502a1bc8fef8f7a3ec9d17a92c835f9d229147e42eaf` |
| `models/heuristic_cnn.py` | 7271 | `8a086c346ab7d2895f0841d336a13c5e26b914c193334de9d24455fd53a7b81d` |
| `neural_astar/__init__.py` | 438 | `7b19b4be396889ac4dd97bd91c9bdfa288eae2f43086c7d8ff7e507d943b0bad` |
| `neural_astar/differentiable.py` | 8554 | `84e979785543f29646e7a6f6319c7be6f12be521220b68dfbade4042fd06a2f1` |
| `neural_astar/guidance.py` | 3711 | `9bbc4a188aa4029d4a9770b2bfc214ce6498db36b7fbb233c8377ffa142dc824` |
| `neural_astar/inference.py` | 3420 | `7209af9e8f91dbf8e869540a19e3f98b5736b84cc70f72c97628f0fe877da363` |
| `neural_astar/model.py` | 4962 | `94d58544ee5d5dde8c7e72a0341f4300864e6a98df426b311e4b63999e27dd6b` |
| `neural_astar/search.py` | 8957 | `73400f94637eb3984cb3ccb076d10bbb7c0526d8f0725ea0fac1391fdc899be8` |
| `neural_astar/training.py` | 12297 | `a1dc1d2f6a1114bc806393cae3b4bed8522ba780f1e29f4e9f4e2f7caf32a35d` |
| `ood/__init__.py` | 1066 | `0cadad7a5deca9def06cd0d716d458f501b7e8fd709720170857c41c44055be6` |
| `ood/contrastive.py` | 14276 | `eac758d883ffa1c481a7acac5ee9b2f326c540cc28434a43d765ae6afcee0c66` |
| `ood/evaluation.py` | 22033 | `7f3f40854d37062dcd3ab0bb29612333c96af84ba5797fb8c401259afb8ca848` |
| `ood/feature_extractor.py` | 6308 | `d4d35a776e73c238e344c2bb20800875454f5065dd81a661e4f24cd3b686152f` |
| `ood/mahalanobis.py` | 13849 | `ce457ce37b4cbcd9a9291e03fc86972dbf2cd2e0ffe3e97acc47865bd4d3e088` |
| `seeastar/__init__.py` | 1168 | `bc150b0756008b9eebebab00040ae29e66d58b33dc1f78351576f2534407352b` |
| `seeastar/benchmark.py` | 12122 | `2503938fa0297d1da8c0ff68f9b3a5398bc07388b51eeea6b4b27c09394c85ee` |
| `seeastar/neural_guidance.py` | 16147 | `1545dbdfdd2b10f6c66c57d39d643fb2bdfc4afbb97357e05eefbd70b3a50e87` |
| `seeastar/original.py` | 11098 | `e4d66e0cdd23ccf3fc2570ad6b59d7e1b779839428cec312ce02f3f69daf54ae` |
| `seeastar/sampling.py` | 8644 | `d6d31d0c747936a2a7283a6730fe55721d219b01193d5f90c697f56c9b7c12e4` |
| `seeastar/uniform.py` | 2346 | `5da5fc3bd7a8c1696bfd18577fa4377484b035b71ead37ffd5a7c3f34cc9ff23` |
| `training/__init__.py` | 211 | `5a8748da6b406311b154db4fa9c37730d06a47491b31365231e1477b22c0e757` |
| `training/heuristic_training.py` | 14658 | `d258706cb8c5a2fe068125a32fd028774e51b781d86d29a57186e47c83af4b94` |
