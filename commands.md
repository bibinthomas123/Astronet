command to predict :  bazel run //astronet:predict -- --model=AstroCNNModel --config_json="C:/Users/bibin.a.thomas/bazel_projects/exoplanet-ml/model/config.json" --model_dir="C:/Users/bibin.a.thomas/bazel_projects/exoplanet-ml/model" --kepler_data_dir="C:/Users/bibin.a.thomas/bazel_projects/kepler" --kepler_id=11442793 --period=14.44912 --t0=2.2 --duration=0.11267 --output_image_file="C:/Users/bibin.a.thomas/bazel_projects/kepler-90i.png" --output_prediction_file="C:/Users/bibin.a.thomas/bazel_projects/prediction.txt"

command to evaluate: bazel run //astronet:evalute --model=AstroCNNModel --config_json=local_global --eval_files = "C:/Users/bibin.a.thomas/bazel_projects/exoplanet-ml/tfrecord/test-*" --model_dir="C:/Users/bibin.a.thomas/bazel_projects/exoplanet-ml/model"

command to clean : bazel clean

command to build: bazel build astronet/...

command to train: bazel run //astronet:train -- `
  --model=AstroCNNModel `
  --config_name=local_global `
  --train_files="C:/Users/bibin.a.thomas/bazel_projects/exoplanet-ml/tfrecord/train-*" `
  --model_dir="C:/Users/bibin.a.thomas/bazel_projects/exoplanet-ml/model" `
  --train_steps=10000



 accuracy/accuracy = 0.95679796, accuracy/num_correct = 1506.0, auc = 0.9843217, confusion_matrix/label_0_pred_0 = 1167.0, confusion_matrix/label_0_pred_1 = 47.0, confusion_matrix/label_1_pred_0 = 21.0, confusion_matrix/label_1_pred_1 = 339.0, global_step = 20000, loss = 0.12627172, losses/weighted_cross_entropy = 0.12721325, num_examples = 1574.0


 Saving dict for global step 20000: accuracy/accuracy = 0.95064604, accuracy/num_correct = 13464.0, auc = 0.9839036, confusion_matrix/label_0_pred_0 = 10458.0, confusion_matrix/label_0_pred_1 = 460.0, confusion_matrix/label_1_pred_0 = 239.0, confusion_matrix/label_1_pred_1 = 3006.0, global_step = 20000, loss = 0.13387372, losses/weighted_cross_entropy = 0.1328648, num_examples = 14163.0

 