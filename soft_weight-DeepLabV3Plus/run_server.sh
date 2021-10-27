ROOT=../dataset
MODEL=deeplabv3plus_resnet101 # deeplabv3plus_resnet101, deeplabv3_resnet101
ITER=4000
BATCH=20
LR=0.06

mkdir -p logs

identifier='28_10_2021_IRNet_full_2_DLV3+_sw'

output_dir=results/seg_labels
soft_weight_root=/home/xzhan/soft_weight_6_10_2021
seg_labels_root=~/data/irn_full_result/sem_seg
train_file_path=./datasets/data/infer.txt
val_file_path=./datasets/data/infer.txt
output_imgs_list_path=datasets/data/infer.txt

num_workers=8

#output_dir=results/seg_labels
#output_imgs_list_path=datasets/data/infer.txt
#irn_mask_root=/home/xuzhan/Documents/irn/data/cross_epoch_exp_5/sem_seg
#irn_imgs_name_path=datasets/data/infer.txt

# training with 2 GPUs
CUDA_VISLBLE_DEVICES=0,1 python main.py --data_root ${ROOT} \
                                        --model ${MODEL} \
                                        --gpu_id 0,1 \
                                        --output_dir ${output_dir} \
                                        --output_imgs_list_path ${output_imgs_list_path} \
                                        --seg_labels_root ${seg_labels_root} \
                                        --train_file_path ${train_file_path} \
                                        --val_file_path ${val_file_path} \
                                        --amp \
                                        --total_itrs ${ITER} \
                                        --batch_size ${BATCH} \
                                        --lr ${LR}  \
                                        --num_workers ${num_workers} \
                                        --soft_weight_root ${soft_weight_root} \
                                        --crop_val 2>&1 | tee logs/${identifier}.txt


## evalutation with crf
#CUDA_VISIBLE_DEVICES=0,1 python eval.py --gpu_id 0,1 --data_root ${ROOT} --model ${MODEL}  --val_batch_size 16  --ckpt checkpoints/best_${MODEL}_voc_os16.pth  --crop_val | tee logs/'eval'${identifier}.txt