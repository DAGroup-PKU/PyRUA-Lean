# PyRUA-Lean on RoboDojo: RGB-D observation, three eval seeds

RoboDojo's own summary (`bash scripts/robodojo.sh summarize`) of the full protocol: 54 task configurations at
their native episode counts, eval seeds 0, 1 and 2 (6,300 episodes), GPT-6 Astra, RoboDojo `main` `cb2a37d`.
In these runs the evaluation client's depth image and camera matrices were enabled. The XPolicyLab policy in
[`../PyRUA_Lean`](../PyRUA_Lean) reads the default RGB-only observation instead; its summary (eval seed 0) is
[`../PyRUA_Lean/eval/robodojo_summary.md`](../PyRUA_Lean/eval/robodojo_summary.md).

Everything below this line is the script's output, unedited.

---

# RoboDojo Evaluation Summary

## Overview (by capability dimension)

Progress † marks incomplete evaluation (126 = 42 tasks × 3 seeds). Cells show `mean ±std` across seeds; scores reflect completed cells only. Policies are sorted by overall Avg SR.

| Policy | Progress | Generalization SR (%) | Generalization Score | Precision SR (%) | Precision Score | Long-Horizon SR (%) | Long-Horizon Score | Memory SR (%) | Memory Score | Open SR (%) | Open Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyRUA_Lean | 100% | 45.44 ±1.23 | 51.11 ±1.10 | 41.08 ±0.51 | 49.78 ±0.57 | 51.75 ±2.15 | 64.73 ±1.39 | 64.00 ±0.00 | 64.02 ±0.02 | 62.25 ±2.16 | 64.39 ±2.16 | 52.91 ±0.35 | 58.81 ±0.15 |

## Generalization: Standard vs Random (25 episodes each)

Each cell shows `mean ±std` across seeds, averaged over the 12 Generalization tasks with completed data for that half. Δ = Random − Standard (mean SR). Sorted by Standard SR.

| Policy | Standard SR (%) | Standard Score | Random SR (%) | Random Score | Δ SR | Δ Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyRUA_Lean | 46.78 ±2.08 | 52.28 ±2.02 | 44.11 ±1.66 | 49.93 ±1.41 | -2.67 | -2.35 |

### Per-task Standard vs Random

Additional view only: each of the 12 Generalization tasks is listed as the 25-episode standard split and its `_random` sibling. Columns match the per-policy all-task tables. Existing merged 50-episode scores are unchanged.

#### PyRUA_Lean

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>stack_bowls</nobr> | 84 | 85.80 | 64 | 68.80 | 76 | 78.40 | 75 ±10 | 77.67 ±8.50 |
| <nobr>stack_bowls_random</nobr> | 64 | 67.60 | 84 | 86.40 | 80 | 81.80 | 76 ±10 | 78.60 ±9.40 |
| <nobr>push_T</nobr> | 72 | 72.00 | 80 | 80.00 | 76 | 76.00 | 76 ±4 | 76.00 ±4.00 |
| <nobr>push_T_random</nobr> | 88 | 88.00 | 80 | 80.00 | 72 | 72.00 | 80 ±8 | 80.00 ±8.00 |
| <nobr>pack_objects_into_box</nobr> | 4 | 25.20 | 20 | 31.80 | 24 | 38.20 | 16 ±10 | 31.73 ±6.50 |
| <nobr>pack_objects_into_box_random</nobr> | 12 | 29.00 | 16 | 32.00 | 28 | 43.60 | 19 ±8 | 34.87 ±7.30 |
| <nobr>fold_clothes</nobr> | 52 | 60.00 | 48 | 55.20 | 64 | 67.20 | 55 ±8 | 60.80 ±6.00 |
| <nobr>fold_clothes_random</nobr> | 36 | 44.80 | 52 | 60.80 | 40 | 49.60 | 43 ±8 | 51.73 ±8.00 |
| <nobr>hang_mugs</nobr> | 12 | 32.00 | 12 | 35.00 | 12 | 39.60 | 12 ±0 | 35.53 ±3.80 |
| <nobr>hang_mugs_random</nobr> | 12 | 34.00 | 12 | 33.20 | 24 | 42.00 | 16 ±6 | 36.40 ±4.40 |
| <nobr>sweep_blocks</nobr> | 28 | 28.00 | 32 | 32.00 | 44 | 44.00 | 35 ±8 | 34.67 ±8.00 |
| <nobr>sweep_blocks_random</nobr> | 16 | 16.00 | 16 | 16.00 | 12 | 12.00 | 15 ±2 | 14.67 ±2.00 |
| <nobr>pour_liquid_into_cup</nobr> | 12 | 12.00 | 4 | 4.00 | 8 | 8.00 | 8 ±4 | 8.00 ±4.00 |
| <nobr>pour_liquid_into_cup_random</nobr> | 20 | 20.00 | 24 | 24.00 | 28 | 28.00 | 24 ±4 | 24.00 ±4.00 |
| <nobr>make_toast</nobr> | 12 | 18.00 | 36 | 42.00 | 24 | 27.00 | 24 ±12 | 29.00 ±12.00 |
| <nobr>make_toast_random</nobr> | 20 | 26.00 | 16 | 17.00 | 4 | 10.00 | 13 ±8 | 17.67 ±8.00 |
| <nobr>arrange_largest_number</nobr> | 80 | 85.00 | 84 | 88.40 | 80 | 86.20 | 81 ±2 | 86.53 ±1.70 |
| <nobr>arrange_largest_number_random</nobr> | 80 | 84.00 | 72 | 79.40 | 64 | 73.20 | 72 ±8 | 78.87 ±5.40 |
| <nobr>sort_nesting_dolls_by_size</nobr> | 72 | 72.00 | 88 | 88.00 | 80 | 80.00 | 80 ±8 | 80.00 ±8.00 |
| <nobr>sort_nesting_dolls_by_size_random</nobr> | 76 | 76.00 | 84 | 84.00 | 56 | 56.00 | 72 ±14 | 72.00 ±14.00 |
| <nobr>store_laptop_and_headphones</nobr> | 0 | 4.00 | 0 | 12.00 | 0 | 6.40 | 0 ±0 | 7.47 ±4.00 |
| <nobr>store_laptop_and_headphones_random</nobr> | 0 | 13.60 | 0 | 7.20 | 0 | 10.40 | 0 ±0 | 10.40 ±3.20 |
| <nobr>stack_blocks</nobr> | 100 | 100.00 | 100 | 100.00 | 100 | 100.00 | 100 ±0 | 100.00 ±0.00 |
| <nobr>stack_blocks_random</nobr> | 100 | 100.00 | 100 | 100.00 | 100 | 100.00 | 100 ±0 | 100.00 ±0.00 |
| **Avg Standard** | 44.0 | 49.50 | 47.3 | 53.10 | 49.0 | 54.25 | 47 ±2 | 52.28 ±2.38 |
| **Avg Random** | 43.7 | 49.92 | 46.3 | 51.67 | 42.3 | 48.22 | 44 ±2 | 49.93 ±1.72 |

## PyRUA_Lean

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>align_blocks</nobr> | 82 | 82.00 | 64 | 64.00 | 68 | 68.00 | 71 ±9 | 71.33 ±9.00 |
| <nobr>arrange_largest_number</nobr> | 80 | 84.50 | 78 | 83.90 | 72 | 79.70 | 77 ±4 | 82.70 ±2.40 |
| <nobr>build_tower</nobr> | 6 | 30.40 | 6 | 29.00 | 0 | 24.40 | 4 ±3 | 27.93 ±3.00 |
| <nobr>classify_objects</nobr> | 100 | 100.00 | 98 | 98.80 | 100 | 100.00 | 99 ±1 | 99.60 ±0.60 |
| <nobr>classify_objects_by_language</nobr> | 88 | 92.80 | 90 | 94.00 | 94 | 96.40 | 91 ±3 | 94.40 ±1.80 |
| <nobr>cover_blocks</nobr> | 100 | 100.00 | 100 | 100.00 | 100 | 100.00 | 100 ±0 | 100.00 ±0.00 |
| <nobr>deposit_coin</nobr> | 36 | 42.40 | 38 | 44.80 | 28 | 36.40 | 34 ±5 | 41.20 ±4.20 |
| <nobr>fasten_screws</nobr> | 66 | 82.40 | 70 | 82.80 | 72 | 83.60 | 69 ±3 | 82.93 ±0.60 |
| <nobr>fill_egg_holder</nobr> | 4 | 15.90 | 2 | 21.30 | 0 | 15.50 | 2 ±2 | 17.57 ±2.90 |
| <nobr>fill_pen_holder</nobr> | 4 | 36.80 | 20 | 47.10 | 26 | 48.50 | 17 ±11 | 44.13 ±5.85 |
| <nobr>fold_clothes</nobr> | 44 | 52.40 | 50 | 58.00 | 52 | 58.40 | 49 ±4 | 56.27 ±3.00 |
| <nobr>general_pickup</nobr> | 96 | 96.00 | 92 | 92.00 | 92 | 92.00 | 93 ±2 | 93.33 ±2.00 |
| <nobr>hang_mugs</nobr> | 12 | 33.00 | 12 | 34.10 | 18 | 40.80 | 14 ±3 | 35.97 ±3.90 |
| <nobr>imitate_sorting_sequence</nobr> | 100 | 100.00 | 96 | 96.10 | 96 | 96.30 | 97 ±2 | 97.47 ±1.95 |
| <nobr>insert_key</nobr> | 0 | 15.00 | 0 | 15.00 | 0 | 15.00 | 0 ±0 | 15.00 ±0.00 |
| <nobr>insert_tubes</nobr> | 70 | 79.20 | 68 | 79.60 | 74 | 82.80 | 71 ±3 | 80.53 ±1.80 |
| <nobr>make_kong</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>make_toast</nobr> | 16 | 22.00 | 26 | 29.50 | 14 | 18.50 | 19 ±6 | 23.33 ±5.50 |
| <nobr>match_and_pick_from_conveyor</nobr> | 84 | 84.00 | 88 | 88.00 | 88 | 88.00 | 87 ±2 | 86.67 ±2.00 |
| <nobr>organize_table</nobr> | 42 | 79.00 | 44 | 78.50 | 32 | 75.00 | 39 ±6 | 77.50 ±2.00 |
| <nobr>pack_objects_into_box</nobr> | 8 | 27.10 | 18 | 31.90 | 26 | 40.90 | 17 ±9 | 33.30 ±6.90 |
| <nobr>pick_from_conveyor_by_image</nobr> | 28 | 28.00 | 24 | 24.00 | 34 | 34.00 | 29 ±5 | 28.67 ±5.00 |
| <nobr>play_Xylophone</nobr> | 70 | 70.00 | 68 | 68.00 | 64 | 64.00 | 67 ±3 | 67.33 ±3.00 |
| <nobr>play_stacking_toy</nobr> | 56 | 79.60 | 72 | 87.00 | 74 | 86.80 | 67 ±9 | 84.47 ±3.70 |
| <nobr>play_tic_tac_toe</nobr> | 94 | 98.00 | 100 | 100.00 | 90 | 95.70 | 95 ±5 | 97.90 ±2.15 |
| <nobr>plug_in_charger</nobr> | 12 | 12.00 | 16 | 16.00 | 14 | 14.00 | 14 ±2 | 14.00 ±2.00 |
| <nobr>pour_balls_into_vase</nobr> | 68 | 68.00 | 68 | 68.00 | 72 | 72.00 | 69 ±2 | 69.33 ±2.00 |
| <nobr>pour_by_language</nobr> | 28 | 28.00 | 24 | 24.40 | 30 | 30.80 | 27 ±3 | 27.73 ±3.20 |
| <nobr>pour_liquid_into_cup</nobr> | 16 | 16.00 | 14 | 14.00 | 18 | 18.00 | 16 ±2 | 16.00 ±2.00 |
| <nobr>press_by_number</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>push_T</nobr> | 80 | 80.00 | 80 | 80.00 | 74 | 74.00 | 78 ±3 | 78.00 ±3.00 |
| <nobr>put_bottles_into_dustbin</nobr> | 92 | 94.90 | 98 | 98.80 | 94 | 96.40 | 95 ±3 | 96.70 ±1.95 |
| <nobr>solve_equation</nobr> | 92 | 92.00 | 80 | 80.00 | 88 | 88.00 | 87 ±6 | 86.67 ±6.00 |
| <nobr>sort_nesting_dolls_by_size</nobr> | 74 | 74.00 | 86 | 86.00 | 68 | 68.00 | 76 ±9 | 76.00 ±9.00 |
| <nobr>stack_blocks</nobr> | 100 | 100.00 | 100 | 100.00 | 100 | 100.00 | 100 ±0 | 100.00 ±0.00 |
| <nobr>stack_blocks_by_language</nobr> | 100 | 100.00 | 100 | 100.00 | 100 | 100.00 | 100 ±0 | 100.00 ±0.00 |
| <nobr>stack_bowls</nobr> | 74 | 76.70 | 74 | 77.60 | 78 | 80.10 | 75 ±2 | 78.13 ±1.70 |
| <nobr>store_laptop_and_headphones</nobr> | 0 | 8.80 | 0 | 9.60 | 0 | 8.40 | 0 ±0 | 8.93 ±0.60 |
| <nobr>store_tools_in_toolbox</nobr> | 0 | 11.50 | 0 | 12.50 | 0 | 15.00 | 0 ±0 | 13.00 ±1.75 |
| <nobr>swap_T</nobr> | 100 | 100.00 | 100 | 100.00 | 100 | 100.00 | 100 ±0 | 100.00 ±0.00 |
| <nobr>swap_blocks</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>sweep_blocks</nobr> | 22 | 22.00 | 24 | 24.00 | 28 | 28.00 | 25 ±3 | 24.67 ±3.00 |
| **Avg (per seed)** | 51.0 | 57.49 | 52.1 | 58.05 | 51.9 | 57.94 | 52 ±1 | 57.83 ±0.28 |
