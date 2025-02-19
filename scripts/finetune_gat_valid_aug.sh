echo "Running gat baseline experiments..."
# AID1798
python finetune_gat_baseline.py --dataset AID1798 --split random_cv1 --valid & \
python finetune_gat_baseline.py --dataset AID1798 --split random_cv2 --valid & \
python finetune_gat_baseline.py --dataset AID1798 --split random_cv3 --valid & \
python finetune_gat_baseline.py --dataset AID1798 --split random_cv4 --valid & \
python finetune_gat_baseline.py --dataset AID1798 --split random_cv5 --valid
wait

python finetune_gat_baseline.py --dataset AID1798 --split scaffold_seed1 --valid & \
python finetune_gat_baseline.py --dataset AID1798 --split scaffold_seed2 --valid & \
python finetune_gat_baseline.py --dataset AID1798 --split scaffold_seed3 --valid & \
python finetune_gat_baseline.py --dataset AID1798 --split scaffold_seed4 --valid & \
python finetune_gat_baseline.py --dataset AID1798 --split scaffold_seed5 --valid
wait

# AID463087
python finetune_gat_baseline.py --dataset AID463087 --split random_cv1 --valid & \
python finetune_gat_baseline.py --dataset AID463087 --split random_cv2 --valid & \
python finetune_gat_baseline.py --dataset AID463087 --split random_cv3 --valid & \
python finetune_gat_baseline.py --dataset AID463087 --split random_cv4 --valid & \
python finetune_gat_baseline.py --dataset AID463087 --split random_cv5 --valid
wait

python finetune_gat_baseline.py --dataset AID463087 --split scaffold_seed1 --valid & \
python finetune_gat_baseline.py --dataset AID463087 --split scaffold_seed2 --valid & \
python finetune_gat_baseline.py --dataset AID463087 --split scaffold_seed3 --valid & \
python finetune_gat_baseline.py --dataset AID463087 --split scaffold_seed4 --valid & \
python finetune_gat_baseline.py --dataset AID463087 --split scaffold_seed5 --valid
wait

# AID488997
python finetune_gat_baseline.py --dataset AID488997 --split random_cv1 --valid & \
python finetune_gat_baseline.py --dataset AID488997 --split random_cv2 --valid & \
python finetune_gat_baseline.py --dataset AID488997 --split random_cv3 --valid & \
python finetune_gat_baseline.py --dataset AID488997 --split random_cv4 --valid & \
python finetune_gat_baseline.py --dataset AID488997 --split random_cv5 --valid
wait

python finetune_gat_baseline.py --dataset AID488997 --split scaffold_seed1 --valid & \
python finetune_gat_baseline.py --dataset AID488997 --split scaffold_seed2 --valid & \
python finetune_gat_baseline.py --dataset AID488997 --split scaffold_seed3 --valid & \
python finetune_gat_baseline.py --dataset AID488997 --split scaffold_seed4 --valid & \
python finetune_gat_baseline.py --dataset AID488997 --split scaffold_seed5 --valid
wait

# AID2689
python finetune_gat_baseline.py --dataset AID2689 --split random_cv1 --valid & \
python finetune_gat_baseline.py --dataset AID2689 --split random_cv2 --valid & \
python finetune_gat_baseline.py --dataset AID2689 --split random_cv3 --valid & \
python finetune_gat_baseline.py --dataset AID2689 --split random_cv4 --valid & \
python finetune_gat_baseline.py --dataset AID2689 --split random_cv5 --valid
wait

python finetune_gat_baseline.py --dataset AID2689 --split scaffold_seed1 --valid & \
python finetune_gat_baseline.py --dataset AID2689 --split scaffold_seed2 --valid & \
python finetune_gat_baseline.py --dataset AID2689 --split scaffold_seed3 --valid & \
python finetune_gat_baseline.py --dataset AID2689 --split scaffold_seed4 --valid & \
python finetune_gat_baseline.py --dataset AID2689 --split scaffold_seed5 --valid
wait

# AID485290
python finetune_gat_baseline.py --dataset AID485290 --split random_cv1 --valid & \
python finetune_gat_baseline.py --dataset AID485290 --split random_cv2 --valid & \
python finetune_gat_baseline.py --dataset AID485290 --split random_cv3 --valid & \
python finetune_gat_baseline.py --dataset AID485290 --split random_cv4 --valid & \
python finetune_gat_baseline.py --dataset AID485290 --split random_cv5 --valid
wait

python finetune_gat_baseline.py --dataset AID485290 --split scaffold_seed1 --valid & \
python finetune_gat_baseline.py --dataset AID485290 --split scaffold_seed2 --valid & \
python finetune_gat_baseline.py --dataset AID485290 --split scaffold_seed3 --valid & \
python finetune_gat_baseline.py --dataset AID485290 --split scaffold_seed4 --valid & \
python finetune_gat_baseline.py --dataset AID485290 --split scaffold_seed5 --valid
wait







