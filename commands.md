command to predict :  bazel run //astronet:predict -- --model=AstroCNNModel --config_json="C:/Users/bibin.a.thomas/bazel_projects/exoplanet-ml/model/config.json" --model_dir="C:/Users/bibin.a.thomas/bazel_projects/exoplanet-ml/model" --kepler_data_dir="C:/Users/bibin.a.thomas/bazel_projects/kepler" --kepler_id=11442793 --period=14.44912 --t0=2.2 --duration=0.11267 --output_image_file="C:/Users/bibin.a.thomas/bazel_projects/kepler-90i.png" --output_prediction_file="C:/Users/bibin.a.thomas/bazel_projects/prediction.txt"

command to evaluate: bazel run //astronet:evalute --model=AstroCNNModel --config_json=local_global --eval_files = "C:/Users/bibin.a.thomas/bazel_projects/exoplanet-ml/tfrecord/test-*"

command to clea : bazel clean

command to build: bazel build astronet/...

command to train: bazel run //astronet:train -- `
  --model=AstroCNNModel `
  --config_name=local_global `
  --train_files="C:/Users/bibin.a.thomas/bazel_projects/exoplanet-ml/tfrecord/train-*" `
  --model_dir="C:/Users/bibin.a.thomas/bazel_projects/exoplanet-ml/model" `
  --train_steps=10000


  
  9595827

  (confirmed planet) Planet candidate score (0-1) [max of output array]: 0.9991153478622437
Score stats:
  Max: 0.99911535
  Mean: 0.41251066
  Median: 0.3540951

  (false positive)Planet candidate score (0-1) [max of output array]: 0.9989097118377686
Score stats:
  Max: 0.9989097
  Mean: 0.7456665
  Median: 0.8665699