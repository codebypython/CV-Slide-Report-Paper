import json
import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("=== SMOKE TESTING NOTEBOOK PRE-TRAINING RUNTIME EXECUTION ===")


notebooks_to_test = [
    '02d_champion_dual_branch_5folds_full_session.ipynb',
    '02e_champion_scale_and_resolution.ipynb',
    '02f_anatomical_roi_multicrop.ipynb'
]

for nb_name in notebooks_to_test:
    nb_path = os.path.join('kaggle_notebooks', nb_name)
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Isolated execution environment
    ns = {}
    for idx, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            raw_lines = c['source']
            clean_lines = []
            for l in raw_lines:
                if l.strip().startswith(('!', '%')):
                    clean_lines.append('# ' + l)
                else:
                    clean_lines.append(l)
            code = ''.join(clean_lines)

            # Skip Step 2 data discovery if not on Kaggle
            if 'DATASET_DIR = None' in code:
                continue
            # Stop before training loop (Step 5)
            if 'session_start_time = time.time()' in code or 'for fold_idx in' in code:
                break

            try:
                exec(code, ns)
                print(f"  [OK] Cell #{idx} in {nb_name}")
            except Exception as e:
                print(f"  [FAILED] Cell #{idx} in {nb_name}: {type(e).__name__} - {e}")
                sys.exit(1)

    print(f"✅ PASSED: All pre-training cells in {nb_name} executed without error!\n")

print("ALL NOTEBOOKS RUNTIME VALIDATION COMPLETED SUCCESSFULLY!")
