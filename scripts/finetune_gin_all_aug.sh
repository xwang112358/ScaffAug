echo "Running gin baseline experiments..."
# AID1798
python finetune_gin_baseline.py --dataset AID1798 --split random_cv1 --aug & \
python finetune_gin_baseline.py --dataset AID1798 --split random_cv2 --aug & \
python finetune_gin_baseline.py --dataset AID1798 --split random_cv3 --aug & \
python finetune_gin_baseline.py --dataset AID1798 --split random_cv4 --aug & \
python finetune_gin_baseline.py --dataset AID1798 --split random_cv5 --aug
wait

python finetune_gin_baseline.py --dataset AID1798 --split scaffold_seed1 --aug & \
python finetune_gin_baseline.py --dataset AID1798 --split scaffold_seed2 --aug & \
python finetune_gin_baseline.py --dataset AID1798 --split scaffold_seed3 --aug & \
python finetune_gin_baseline.py --dataset AID1798 --split scaffold_seed4 --aug & \
python finetune_gin_baseline.py --dataset AID1798 --split scaffold_seed5 --aug
wait

# AID463087
python finetune_gin_baseline.py --dataset AID463087 --split random_cv1 --aug & \
python finetune_gin_baseline.py --dataset AID463087 --split random_cv2 --aug & \
python finetune_gin_baseline.py --dataset AID463087 --split random_cv3 --aug & \
python finetune_gin_baseline.py --dataset AID463087 --split random_cv4 --aug & \
python finetune_gin_baseline.py --dataset AID463087 --split random_cv5 --aug
wait

python finetune_gin_baseline.py --dataset AID463087 --split scaffold_seed1 --aug & \
python finetune_gin_baseline.py --dataset AID463087 --split scaffold_seed2 --aug & \
python finetune_gin_baseline.py --dataset AID463087 --split scaffold_seed3 --aug & \
python finetune_gin_baseline.py --dataset AID463087 --split scaffold_seed4 --aug & \
python finetune_gin_baseline.py --dataset AID463087 --split scaffold_seed5 --aug
wait

# AID488997
python finetune_gin_baseline.py --dataset AID488997 --split random_cv1 --aug & \
python finetune_gin_baseline.py --dataset AID488997 --split random_cv2 --aug & \
python finetune_gin_baseline.py --dataset AID488997 --split random_cv3 --aug & \
python finetune_gin_baseline.py --dataset AID488997 --split random_cv4 --aug & \
python finetune_gin_baseline.py --dataset AID488997 --split random_cv5 --aug
wait

python finetune_gin_baseline.py --dataset AID488997 --split scaffold_seed1 --aug & \
python finetune_gin_baseline.py --dataset AID488997 --split scaffold_seed2 --aug & \
python finetune_gin_baseline.py --dataset AID488997 --split scaffold_seed3 --aug & \
python finetune_gin_baseline.py --dataset AID488997 --split scaffold_seed4 --aug & \
python finetune_gin_baseline.py --dataset AID488997 --split scaffold_seed5 --aug
wait

# AID2689
python finetune_gin_baseline.py --dataset AID2689 --split random_cv1 --aug & \
python finetune_gin_baseline.py --dataset AID2689 --split random_cv2 --aug & \
python finetune_gin_baseline.py --dataset AID2689 --split random_cv3 --aug & \
python finetune_gin_baseline.py --dataset AID2689 --split random_cv4 --aug & \
python finetune_gin_baseline.py --dataset AID2689 --split random_cv5 --aug
wait

python finetune_gin_baseline.py --dataset AID2689 --split scaffold_seed1 --aug & \
python finetune_gin_baseline.py --dataset AID2689 --split scaffold_seed2 --aug & \
python finetune_gin_baseline.py --dataset AID2689 --split scaffold_seed3 --aug & \
python finetune_gin_baseline.py --dataset AID2689 --split scaffold_seed4 --aug & \
python finetune_gin_baseline.py --dataset AID2689 --split scaffold_seed5 --aug
wait

# AID485290
python finetune_gin_baseline.py --dataset AID485290 --split random_cv1 --aug & \
python finetune_gin_baseline.py --dataset AID485290 --split random_cv2 --aug & \
python finetune_gin_baseline.py --dataset AID485290 --split random_cv3 --aug & \
python finetune_gin_baseline.py --dataset AID485290 --split random_cv4 --aug & \
python finetune_gin_baseline.py --dataset AID485290 --split random_cv5 --aug
wait

python finetune_gin_baseline.py --dataset AID485290 --split scaffold_seed1 --aug & \
python finetune_gin_baseline.py --dataset AID485290 --split scaffold_seed2 --aug & \
python finetune_gin_baseline.py --dataset AID485290 --split scaffold_seed3 --aug & \
python finetune_gin_baseline.py --dataset AID485290 --split scaffold_seed4 --aug & \
python finetune_gin_baseline.py --dataset AID485290 --split scaffold_seed5 --aug
wait







