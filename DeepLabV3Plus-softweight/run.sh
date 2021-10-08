ROOT=../dataset
MODEL=deeplabv3plus_resnet101 # deeplabv3plus_resnet101, deeplabv3_resnet101
ITER=8000
BATCH=32
LR=0.04

identifier='08_10_2021_IRNet'

# training with 2 GPUs
CUDA_VISLBLE_DEVICES=0,1 python main.py --data_root ${ROOT} --model ${MODEL} --gpu_id 0,1 --amp --total_itrs ${ITER} --batch_size ${BATCH} --lr ${LR}  --crop_val | tee logs/${identifier}.txt


# evalutation with crf
CUDA_VISIBLE_DEVICES=0,1 python eval.py --gpu_id 0,1 --data_root ${ROOT} --model ${MODEL}  --val_batch_size 16  --ckpt checkpoints/best_${MODEL}_voc_os16.pth  --crop_val | tee logs/'eval'${identifier}.txt