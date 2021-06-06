git config --global user.email "snehal.v.uphale@gmail.com"
git config --global user.name "Snehal Uphale"
sudo apt-get install -y python3-tk
pip3 install --user Cython matplotlib opencv-python-headless pyyaml Pillow
pip3 install --user 'git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI'
pip3 install --user -U gast
sudo pip3 install --user -r /usr/share/models/official/requirements.txt
export STORAGE_BUCKET=gs://snehal_bucket
export PYTHONPATH="${PYTHONPATH}:/home/suphale/retinanet/tpu/models"
export DATA_DIR=${STORAGE_BUCKET}/coco
export MODEL_DIR=${STORAGE_BUCKET}/retinanet-train
export RESNET_CHECKPOINT=${STORAGE_BUCKET}/resnet/resnet50-checkpoint-2018-02-07
export TRAIN_FILE_PATTERN=${DATA_DIR}/train-*
export EVAL_FILE_PATTERN=${DATA_DIR}/val-*
export VAL_JSON_FILE=${DATA_DIR}/instances_val2017.json
export PYTHONPATH=/usr/share/models
python3 ~/retinanet/models/official/vision/detection/main.py --strategy_type=tpu --tpu=${TPU_NAME} --model_dir=${MODEL_DIR} --mode="train" --params_override="{ 
type: retinanet, train: { total_steps: 10, checkpoint: { path: ${RESNET_CHECKPOINT}, prefix: resnet50/ }, train_file_pattern: ${TRAIN_FILE_PATTERN} }, eval: { val_json_file: ${VAL_JSON_FILE}, eval_file_pattern: ${EVAL_FILE_PATTERN}, eval_samples: 5000 } }"
export EVAL_SAMPLES=5000
python3 ~/retinanet/models/official/vision/detection/main.py --strategy_type=tpu --tpu=${TPU_NAME} --model_dir=${MODEL_DIR} --checkpoint_path=${MODEL_DIR} --mode=eval_once --params_override="{ type: retinanet, eval: { val_json_file: ${VAL_JSON_FILE}, eval_file_pattern: ${EVAL_FILE_PATTERN}, eval_samples: ${EVAL_SAMPLES} } }"
