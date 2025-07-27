command to predict :  bazel run //astronet:predict -- --model=AstroCNNModel --config_json="C:/Users/bibin.a.thomas/bazel_projects/exoplanet-ml/model/config.json" --model_dir="C:/Users/bibin.a.thomas/bazel_projects/exoplanet-ml/model" --kepler_data_dir="C:/Users/bibin.a.thomas/bazel_projects/kepler" --kepler_id=11442793 --period=14.44912 --t0=2.2 --duration=0.11267 --output_image_file="C:/Users/bibin.a.thomas/bazel_projects/kepler-90i.png" --output_prediction_file="C:/Users/bibin.a.thomas/bazel_projects/prediction.txt"

command to clea : bazel clean

command to build: bazel build astronet/...

command to train: bazel run //astronet:train -- `
  --model=AstroCNNModel `
  --config_name=local_global `
  --train_files="C:/Users/bibin.a.thomas/bazel_projects/exoplanet-ml/tfrecord/train-*" `
  --model_dir="C:/Users/bibin.a.thomas/bazel_projects/exoplanet-ml/model" `
  --max_steps=10000


  