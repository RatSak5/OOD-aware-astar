**Current initial bundle:** use `initial_training_bundle_manifest.md` and `.json` for the rebuilt initial bundle, current source inventory, and fresh hashes. The separate OOD-pilot entry below is historical; its archive was not accessed or rebuilt.

# Phase B bundle manifests — 64x64_rectangles_v1

NPZ files were copied and hashed as opaque bytes. No NPZ container was inspected or parsed, and no arrays were read in this correction.

## Initial training bundle

- Path: C:\Users\ratul\OneDrive\Documents\AI_project_backup\results\64x64_rectangles_v1\64x64_rectangles_v1_kaggle_upload.zip
- Bytes: 1923683
- SHA-256: 1bc52b8e783c7d498f6c8c08de30c46bddcbb9eb888342fa843042a273fbfbc5
- Manifest counts: {'id_pilot': 60, 'training': 900, 'validation_calibration': 180}
- OOD pilot and its per-map records: absent.
- ID holdout and sealed OOD test: absent.
- Uploaded: no.

| File | Bytes | SHA-256 |
|---|---:|---|
| README_upload.md | 749 | 61f9da9e763f34ebafd9e966935be09dba42b5df9526e3570034c332c1118567 |
| configs/adaptive_sensitivity_ablations.json | 1446 | ffc29b9daeb662ab37c7796956b7cc26809d603f60b2e619ca4a8c6bc8dc03b6 |
| dataset/id_pilot.npz | 91219 | 471279746c104b36c5d47b3f4619faec38d6c69e4a903c5ef8b21fa12242c1d1 |
| dataset/manifest.json | 2296227 | ceb4e029ee791ec5bc620c1cd48b87f71b5f2f88268bfec4f7de194043ac94c0 |
| dataset/training.npz | 1253120 | fa8eb991ac3ad79111340e4beda8c631fccbd10d89e5f04e94a7450e1a4a5ff3 |
| dataset/validation_calibration.npz | 259317 | 6847105762043792858d74e5ee4e3a175cfce8cb84f13e55262d06a929820221 |
| notebooks/kaggle_experiments_64x64_rectangles.ipynb | 24909 | e26faab4ae4a322dce205f323f2b18ee3acf41b8dd6be65d30a10029716d0e1f |
| project_code.zip | 175167 | 67e44d78226318442bca0d0799072e1c398c127e46a17f287db68dc9dbbb5977 |
| upload_manifest.json | 1553 | 42c74ecd4de464c7d31aff20775068f581b124d376191da3edd1c117dc24aa98 |

### Initial ZIP listing

| Member | Uncompressed bytes |
|---|---:|
| README_upload.md | 749 |
| configs/adaptive_sensitivity_ablations.json | 1446 |
| dataset/id_pilot.npz | 91219 |
| dataset/manifest.json | 2296227 |
| dataset/training.npz | 1253120 |
| dataset/validation_calibration.npz | 259317 |
| notebooks/kaggle_experiments_64x64_rectangles.ipynb | 24909 |
| project_code.zip | 175167 |
| upload_manifest.json | 1553 |

## Separate OOD-pilot bundle

- Path: C:\Users\ratul\OneDrive\Documents\AI_project_backup\results\64x64_rectangles_v1\64x64_rectangles_v1_ood_pilot.zip
- Bytes: 112478
- SHA-256: 4f31c60927a644b1496130b2e59fd0a004e0f687f88516baf8a2a268765bbc1c
- Manifest counts: {'ood_pilot': 60}
- Uploaded or attached: no.
- Premature header inspection: yes; no map evaluation, training, fitting, or calibration used it.

| File | Bytes | SHA-256 |
|---|---:|---|
| README_ood_pilot.md | 651 | d34229220ed2c359551e81089d7a06eeb2711deb4f4a66814c294b6e80c8e9a8 |
| dataset/manifest.json | 156418 | 1fa98c4e670e83dedcb75d2fa35d45e1e83a9210973aec9b5b8d66f976b46fab |
| dataset/ood_pilot.npz | 101256 | ddfd02b5a8af2db0f0f9df59211ae1fc7afcb735a27399767efd1bafcfeee6fa |
| upload_manifest.json | 921 | aa3a128e5bfeee691168bc2cfcb7ced1a4350508c64f89f647f8793baa956af2 |

### Separate OOD ZIP listing

| Member | Uncompressed bytes |
|---|---:|---|
| README_ood_pilot.md | 651 |
| dataset/manifest.json | 156418 |
| dataset/ood_pilot.npz | 101256 |
| upload_manifest.json | 921 |

## Nested project source files

| Source file | Bytes | SHA-256 |
|---|---:|---|
| adaptive/__init__.py | 858 | 707526963f5bb1dd7352e25e9cca742b37094a878a8993f11895f913cba32fea |
| adaptive/ood_adaptive_seeastar.py | 13974 | e7a89fdf8ca850979dc42b5aa4f7e191a515a2f9ce0f6d1a3e9e3ec539b73497 |
| algorithms/__init__.py | 62 | c5e3cb26844e29d6810187f0a5084eb39c730cc73f4df92c158a21acc1b6d7d9 |
| algorithms/astar.py | 9051 | baa4e742fb0d927abe14eb711646b3e8bad0c02c5694edaa037f452aaeddd808 |
| algorithms/dijkstra.py | 2927 | a4372fb0d080c0edafb01390a8f20f97a33f447d2ac2be289e8baeb11634b3cf |
| algorithms/movement.py | 6230 | e03d41095a2b685929ab890b66b9b1f4669264abb23599b4b394d0e4548fac37 |
| benchmarks/__init__.py | 139 | 08d859f61568573b5c7cfa74a874d7dbf93938c3533182288e210292e810bd34 |
| benchmarks/__main__.py | 3769 | 2f7df179374b7e620e2e136e5ec5f2d1fae34e6212ca47ffb7676054ca05ea5f |
| benchmarks/config.py | 8431 | 16d2a2c1020d80382a1b0e7c710a8e54025c21cfc5472dd00381a18400b4c895 |
| benchmarks/data.py | 23524 | 9e88e1174d3992c722c5b54903d3055e72218937c32170d3683e42aa5bae2c64 |
| benchmarks/execution.py | 5083 | 0b94d5dc3ff017a6342a30e4854c9e748b254ea392dab543355475677a4e4aca |
| benchmarks/inference.py | 8745 | f43505516393d5e4d462b54d8c6bde70b7a42e82479c0c9ca0f1e2b8b280848a |
| benchmarks/neural_trace.py | 4560 | 4f98147847155477aaaee38456d42850b9a69c46663a4d8b96ea794054389afb |
| benchmarks/rectangle_pilot.py | 37547 | 650eef97c39cc4e5ffec68a5ef4f2b45a56753cdce21a790b1030089429305d4 |
| benchmarks/reliability.py | 9298 | 8ee48500004efc893161af7f175559bde2329b5ddb597e7f84f26f5896c16e7e |
| benchmarks/runner.py | 69987 | e2747340c42cacb808c52aca5a6d5ae5e1e775cccc78d15fd1c1b323551409e3 |
| benchmarks/search_worker.py | 10061 | 3fb71d24959e8006ae419b45add7a29d7d0f051ec86bd8ac145f312891d59d58 |
| benchmarks/statistics.py | 6236 | ff713fdb16673c7540806bad1512d2c0b596ad29d5fba1912d7213c21c1b7650 |
| benchmarks/workflow.py | 19444 | c03d0593607104689948fbf5560e87080fc61f62837c44613f2d7726dd96262c |
| data/__init__.py | 64 | b39e631de6454400be0c461fa84f49f7eb42227c994f15306ece2e55ad0320e7 |
| data/dataset.py | 10499 | c5188523bbb0d9ca66c47965ffb29ff3d06ff5c376addc1e203878d2a7c63efb |
| data/ground_truth.py | 3603 | ccfbb98336d16bd655a9c0a1291a7daf26c4817b388e6989b06b305231350ed9 |
| data/map_generator.py | 14362 | 560fc8f1d608183c8e9914458a3aaacaae238d11037924dd39d99e411d0b9de0 |
| data/rectangular_dataset.py | 34307 | 9c1a875159f7d53b8962752eced09745ecb243d2a7aa771b8a7df90fe14d0b67 |
| experiments/__init__.py | 39 | ae5db848a98c5068cb0751096830fbf375d19f0f161e5424054f727a6f2af7ee |
| experiments/adaptive_seeastar.py | 36956 | 1c775deeb9d9eee4c6324c0319ebfcbdb1f6e02344a3fe960df5465e48e89e41 |
| experiments/heuristic_comparison.py | 5108 | 535cec7bab81c7132606a580fc97144cb2e20696a752c0d41c6481de0fe6acee |
| experiments/ood_detection.py | 5336 | 0d92baa50cfeab5ac06d038d1685703a65e27c606679af104e249a067911128c |
| experiments/prepare_kaggle_bundle.py | 40590 | 39a896297497df44bde691d30fea9c3a3f411b02d3bb07cc2a9c6fdb64c53c0a |
| experiments/rectangle_pipeline.py | 34160 | cd5d92c32cca79d9930186202071a140455e50fe6ad2e483c51842db7d5a83f6 |
| heuristics/__init__.py | 374 | 818904a7976cb6ae25933da84067644eef3155b7a4bb31040010d03f924eeceb |
| heuristics/cea.py | 3764 | 44d38c9e0af137a329510b465e79b1d0d1f434162ac4e72dd407a3c678b23666 |
| heuristics/data.py | 6058 | dc60d61ed388e0cee8bb01be25ccfdde561bed2afa2d337c2439b8b2ffdaf605 |
| heuristics/evaluation.py | 7913 | 637c0475e5b60df578d984de7222ed06e131bde7ab8aad1c4812355af12cbab3 |
| heuristics/inference.py | 4157 | 074c91b94374aaaa7e0ba650fb29d819743859d96f7ba602572f407b2c8c625f |
| heuristics/losses.py | 4631 | 304e84d24b1d93a326b28dad24b83daa73d45165882d9bfaf9d881fe21ca4e0e |
| models/__init__.py | 194 | c12dd6c64d1898a0b8c2502a1bc8fef8f7a3ec9d17a92c835f9d229147e42eaf |
| models/heuristic_cnn.py | 7271 | 8a086c346ab7d2895f0841d336a13c5e26b914c193334de9d24455fd53a7b81d |
| neural_astar/__init__.py | 438 | 7b19b4be396889ac4dd97bd91c9bdfa288eae2f43086c7d8ff7e507d943b0bad |
| neural_astar/differentiable.py | 8554 | 84e979785543f29646e7a6f6319c7be6f12be521220b68dfbade4042fd06a2f1 |
| neural_astar/guidance.py | 3711 | 9bbc4a188aa4029d4a9770b2bfc214ce6498db36b7fbb233c8377ffa142dc824 |
| neural_astar/inference.py | 3420 | 7209af9e8f91dbf8e869540a19e3f98b5736b84cc70f72c97628f0fe877da363 |
| neural_astar/model.py | 4962 | 94d58544ee5d5dde8c7e72a0341f4300864e6a98df426b311e4b63999e27dd6b |
| neural_astar/search.py | 8342 | 51326464d8335c0d21c9bd6c018720515dc16ab7ef562d96b7402e41eca63af9 |
| neural_astar/training.py | 12297 | a1dc1d2f6a1114bc806393cae3b4bed8522ba780f1e29f4e9f4e2f7caf32a35d |
| ood/__init__.py | 1066 | 0cadad7a5deca9def06cd0d716d458f501b7e8fd709720170857c41c44055be6 |
| ood/contrastive.py | 12514 | 7250aae402665071cbb4bb3cc9a19a234c1793772b3e510fa6a38adec8edb7fa |
| ood/evaluation.py | 22033 | 7f3f40854d37062dcd3ab0bb29612333c96af84ba5797fb8c401259afb8ca848 |
| ood/feature_extractor.py | 6308 | d4d35a776e73c238e344c2bb20800875454f5065dd81a661e4f24cd3b686152f |
| ood/mahalanobis.py | 13849 | ce457ce37b4cbcd9a9291e03fc86972dbf2cd2e0ffe3e97acc47865bd4d3e088 |
| seeastar/__init__.py | 1168 | bc150b0756008b9eebebab00040ae29e66d58b33dc1f78351576f2534407352b |
| seeastar/benchmark.py | 12122 | 2503938fa0297d1da8c0ff68f9b3a5398bc07388b51eeea6b4b27c09394c85ee |
| seeastar/neural_guidance.py | 15839 | 00727d6d91d9d741ab5b4cc1a95351489334825c3e46f8a34d5e840cc96871d3 |
| seeastar/original.py | 11098 | e4d66e0cdd23ccf3fc2570ad6b59d7e1b779839428cec312ce02f3f69daf54ae |
| seeastar/sampling.py | 8644 | d6d31d0c747936a2a7283a6730fe55721d219b01193d5f90c697f56c9b7c12e4 |
| seeastar/uniform.py | 2346 | 5da5fc3bd7a8c1696bfd18577fa4377484b035b71ead37ffd5a7c3f34cc9ff23 |
| training/__init__.py | 211 | 5a8748da6b406311b154db4fa9c37730d06a47491b31365231e1477b22c0e757 |
| training/heuristic_training.py | 14658 | d258706cb8c5a2fe068125a32fd028774e51b781d86d29a57186e47c83af4b94 |

## Status

- Sealed OOD-test archive not opened during this correction.
- No browser action, upload, Kaggle job, training, calibration, integration test, or pilot occurred.
