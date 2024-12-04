python finetune_scaffaug.py --dataset AID1798 --split random_cv1 --ratio 0.1 & \
python finetune_scaffaug.py --dataset AID1798 --split random_cv2 --ratio 0.1 & \
python finetune_scaffaug.py --dataset AID1798 --split random_cv3 --ratio 0.1 & \
python finetune_scaffaug.py --dataset AID1798 --split random_cv4 --ratio 0.1 & \
python finetune_scaffaug.py --dataset AID1798 --split random_cv5 --ratio 0.1 & \
wait

python finetune_scaffaug.py --dataset AID1798 --split scaffold_seed1 --ratio 0.1 & \
python finetune_scaffaug.py --dataset AID1798 --split scaffold_seed2 --ratio 0.1 & \
python finetune_scaffaug.py --dataset AID1798 --split scaffold_seed3 --ratio 0.1 & \
python finetune_scaffaug.py --dataset AID1798 --split scaffold_seed4 --ratio 0.1 & \
python finetune_scaffaug.py --dataset AID1798 --split scaffold_seed5 --ratio 0.1 & \
wait