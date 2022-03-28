#! /bin/bash

DATASET_ROOT="/path/to/mvtec"

DATASET=bottle
WAY=4
echo "Start ${DATASET}"
python -u run_simple_cnaps_mvtec.py --dataset-dir "${DATASET_ROOT}/${DATASET}" --feature_adaptation film --checkpoint_dir /home/sanno/work/simple-cnaps-checkpoints --pretrained_resnet_path /home/sanno/work/simple-cnaps/model-checkpoints/pretrained_resnets/pretrained_resnet_mini_imagenet.pt.tar --test_model_path /home/sanno/work/simple-cnaps/model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt --shot 5 --way ${WAY} --testnum 1

DATASET=cable
WAY=9
echo "Start ${DATASET}"
python -u run_simple_cnaps_mvtec.py --dataset-dir "${DATASET_ROOT}/${DATASET}" --feature_adaptation film --checkpoint_dir /home/sanno/work/simple-cnaps-checkpoints --pretrained_resnet_path /home/sanno/work/simple-cnaps/model-checkpoints/pretrained_resnets/pretrained_resnet_mini_imagenet.pt.tar --test_model_path /home/sanno/work/simple-cnaps/model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt --shot 5 --way ${WAY} --testnum 1

DATASET=capsule
WAY=6
echo "Start ${DATASET}"
python -u run_simple_cnaps_mvtec.py --dataset-dir "${DATASET_ROOT}/${DATASET}" --feature_adaptation film --checkpoint_dir /home/sanno/work/simple-cnaps-checkpoints --pretrained_resnet_path /home/sanno/work/simple-cnaps/model-checkpoints/pretrained_resnets/pretrained_resnet_mini_imagenet.pt.tar --test_model_path /home/sanno/work/simple-cnaps/model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt --shot 5 --way ${WAY} --testnum 1

DATASET=carpet
WAY=6
echo "Start ${DATASET}"
python -u run_simple_cnaps_mvtec.py --dataset-dir "${DATASET_ROOT}/${DATASET}" --feature_adaptation film --checkpoint_dir /home/sanno/work/simple-cnaps-checkpoints --pretrained_resnet_path /home/sanno/work/simple-cnaps/model-checkpoints/pretrained_resnets/pretrained_resnet_mini_imagenet.pt.tar --test_model_path /home/sanno/work/simple-cnaps/model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt --shot 5 --way ${WAY} --testnum 1

DATASET=grid
WAY=6
echo "Start ${DATASET}"
python -u run_simple_cnaps_mvtec.py --dataset-dir "${DATASET_ROOT}/${DATASET}" --feature_adaptation film --checkpoint_dir /home/sanno/work/simple-cnaps-checkpoints --pretrained_resnet_path /home/sanno/work/simple-cnaps/model-checkpoints/pretrained_resnets/pretrained_resnet_mini_imagenet.pt.tar --test_model_path /home/sanno/work/simple-cnaps/model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt --shot 5 --way ${WAY} --testnum 1

DATASET=hazelnut
WAY=5
echo "Start ${DATASET}"
python -u run_simple_cnaps_mvtec.py --dataset-dir "${DATASET_ROOT}/${DATASET}" --feature_adaptation film --checkpoint_dir /home/sanno/work/simple-cnaps-checkpoints --pretrained_resnet_path /home/sanno/work/simple-cnaps/model-checkpoints/pretrained_resnets/pretrained_resnet_mini_imagenet.pt.tar --test_model_path /home/sanno/work/simple-cnaps/model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt --shot 5 --way ${WAY} --testnum 1

DATASET=leather
WAY=6
echo "Start ${DATASET}"
python -u run_simple_cnaps_mvtec.py --dataset-dir "${DATASET_ROOT}/${DATASET}" --feature_adaptation film --checkpoint_dir /home/sanno/work/simple-cnaps-checkpoints --pretrained_resnet_path /home/sanno/work/simple-cnaps/model-checkpoints/pretrained_resnets/pretrained_resnet_mini_imagenet.pt.tar --test_model_path /home/sanno/work/simple-cnaps/model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt --shot 5 --way ${WAY} --testnum 1

DATASET=metal_nut
WAY=5
echo "Start ${DATASET}"
python -u run_simple_cnaps_mvtec.py --dataset-dir "${DATASET_ROOT}/${DATASET}" --feature_adaptation film --checkpoint_dir /home/sanno/work/simple-cnaps-checkpoints --pretrained_resnet_path /home/sanno/work/simple-cnaps/model-checkpoints/pretrained_resnets/pretrained_resnet_mini_imagenet.pt.tar --test_model_path /home/sanno/work/simple-cnaps/model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt --shot 5 --way ${WAY} --testnum 1

DATASET=pill
WAY=8
echo "Start ${DATASET}"
python -u run_simple_cnaps_mvtec.py --dataset-dir "${DATASET_ROOT}/${DATASET}" --feature_adaptation film --checkpoint_dir /home/sanno/work/simple-cnaps-checkpoints --pretrained_resnet_path /home/sanno/work/simple-cnaps/model-checkpoints/pretrained_resnets/pretrained_resnet_mini_imagenet.pt.tar --test_model_path /home/sanno/work/simple-cnaps/model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt --shot 5 --way ${WAY} --testnum 1

DATASET=screw
WAY=6
echo "Start ${DATASET}"
python -u run_simple_cnaps_mvtec.py --dataset-dir "${DATASET_ROOT}/${DATASET}" --feature_adaptation film --checkpoint_dir /home/sanno/work/simple-cnaps-checkpoints --pretrained_resnet_path /home/sanno/work/simple-cnaps/model-checkpoints/pretrained_resnets/pretrained_resnet_mini_imagenet.pt.tar --test_model_path /home/sanno/work/simple-cnaps/model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt --shot 5 --way ${WAY} --testnum 1

DATASET=tile
WAY=6
echo "Start ${DATASET}"
python -u run_simple_cnaps_mvtec.py --dataset-dir "${DATASET_ROOT}/${DATASET}" --feature_adaptation film --checkpoint_dir /home/sanno/work/simple-cnaps-checkpoints --pretrained_resnet_path /home/sanno/work/simple-cnaps/model-checkpoints/pretrained_resnets/pretrained_resnet_mini_imagenet.pt.tar --test_model_path /home/sanno/work/simple-cnaps/model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt --shot 5 --way ${WAY} --testnum 1

DATASET=toothbrush
WAY=2
echo "Start ${DATASET}"
python -u run_simple_cnaps_mvtec.py --dataset-dir "${DATASET_ROOT}/${DATASET}" --feature_adaptation film --checkpoint_dir /home/sanno/work/simple-cnaps-checkpoints --pretrained_resnet_path /home/sanno/work/simple-cnaps/model-checkpoints/pretrained_resnets/pretrained_resnet_mini_imagenet.pt.tar --test_model_path /home/sanno/work/simple-cnaps/model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt --shot 5 --way ${WAY} --testnum 1

DATASET=transistor
WAY=5
echo "Start ${DATASET}"
python -u run_simple_cnaps_mvtec.py --dataset-dir "${DATASET_ROOT}/${DATASET}" --feature_adaptation film --checkpoint_dir /home/sanno/work/simple-cnaps-checkpoints --pretrained_resnet_path /home/sanno/work/simple-cnaps/model-checkpoints/pretrained_resnets/pretrained_resnet_mini_imagenet.pt.tar --test_model_path /home/sanno/work/simple-cnaps/model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt --shot 5 --way ${WAY} --testnum 1

DATASET=wood
WAY=6
echo "Start ${DATASET}"
python -u run_simple_cnaps_mvtec.py --dataset-dir "${DATASET_ROOT}/${DATASET}" --feature_adaptation film --checkpoint_dir /home/sanno/work/simple-cnaps-checkpoints --pretrained_resnet_path /home/sanno/work/simple-cnaps/model-checkpoints/pretrained_resnets/pretrained_resnet_mini_imagenet.pt.tar --test_model_path /home/sanno/work/simple-cnaps/model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt --shot 5 --way ${WAY} --testnum 1

DATASET=zipper
WAY=8
echo "Start ${DATASET}"
python -u run_simple_cnaps_mvtec.py --dataset-dir "${DATASET_ROOT}/${DATASET}" --feature_adaptation film --checkpoint_dir /home/sanno/work/simple-cnaps-checkpoints --pretrained_resnet_path /home/sanno/work/simple-cnaps/model-checkpoints/pretrained_resnets/pretrained_resnet_mini_imagenet.pt.tar --test_model_path /home/sanno/work/simple-cnaps/model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt --shot 5 --way ${WAY} --testnum 1
