# RoboDojo Evaluation Summary (official)

Run by the RoboDojo team with the `PyRUA_Lean` adapter in this directory, on all 42 tasks with seed 3; the file is
reproduced as they sent it. Our own run of the same adapter is in [robodojo_summary_rgb.md](robodojo_summary_rgb.md).

## Overview (by capability dimension)

Progress † marks incomplete evaluation (42 = 42 tasks × seed 3). Cells show `mean ±std` across seeds; scores reflect completed cells only. Avg is the mean of all five dimension means. Policies are sorted by overall Avg SR.

| Policy | Progress | Generalization SR (%) | Generalization Score | Precision SR (%) | Precision Score | Long-Horizon SR (%) | Long-Horizon Score | Memory SR (%) | Memory Score | Open SR (%) | Open Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyRUA_Lean | 100% | 43.17 ±0.00 | 48.27 ±0.00 | 38.25 ±0.00 | 44.77 ±0.00 | 43.75 ±0.00 | 54.30 ±0.00 | 68.33 ±0.00 | 68.40 ±0.00 | 58.25 ±0.00 | 60.56 ±0.00 | 50.35 ±0.00 | 55.26 ±0.00 |

## Generalization: Standard vs Random (25 episodes each)

Each cell shows `mean ±std` across seeds, averaged over the 12 Generalization tasks with completed data for that half. Δ = Random − Standard (mean SR). Sorted by Standard SR.

| Policy | Standard SR (%) | Standard Score | Random SR (%) | Random Score | Δ SR | Δ Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyRUA_Lean | 45.33 ±0.00 | 50.47 ±0.00 | 41.00 ±0.00 | 46.07 ±0.00 | -4.33 | -4.40 |

### Per-task Standard vs Random

Additional view only: each of the 12 Generalization tasks is listed as the 25-episode standard split and its `_random` sibling. Columns match the per-policy all-task tables. Existing merged 50-episode scores are unchanged.

#### PyRUA_Lean

| Task | Seed 3 SR (%) | Seed 3 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: |
| <nobr>stack_bowls</nobr> | 80.00 | 82.40 | 80.00 ±0.00 | 82.40 ±0.00 |
| <nobr>stack_bowls_random</nobr> | 60.00 | 65.40 | 60.00 ±0.00 | 65.40 ±0.00 |
| <nobr>push_T</nobr> | 80.00 | 80.00 | 80.00 ±0.00 | 80.00 ±0.00 |
| <nobr>push_T_random</nobr> | 92.00 | 92.00 | 92.00 ±0.00 | 92.00 ±0.00 |
| <nobr>pack_objects_into_box</nobr> | 40.00 | 52.80 | 40.00 ±0.00 | 52.80 ±0.00 |
| <nobr>pack_objects_into_box_random</nobr> | 20.00 | 39.20 | 20.00 ±0.00 | 39.20 ±0.00 |
| <nobr>fold_clothes</nobr> | 52.00 | 58.40 | 52.00 ±0.00 | 58.40 ±0.00 |
| <nobr>fold_clothes_random</nobr> | 52.00 | 60.00 | 52.00 ±0.00 | 60.00 ±0.00 |
| <nobr>hang_mugs</nobr> | 4.00 | 27.00 | 4.00 ±0.00 | 27.00 ±0.00 |
| <nobr>hang_mugs_random</nobr> | 20.00 | 28.40 | 20.00 ±0.00 | 28.40 ±0.00 |
| <nobr>sweep_blocks</nobr> | 4.00 | 4.00 | 4.00 ±0.00 | 4.00 ±0.00 |
| <nobr>sweep_blocks_random</nobr> | 8.00 | 8.00 | 8.00 ±0.00 | 8.00 ±0.00 |
| <nobr>pour_liquid_into_cup</nobr> | 4.00 | 4.00 | 4.00 ±0.00 | 4.00 ±0.00 |
| <nobr>pour_liquid_into_cup_random</nobr> | 4.00 | 4.00 | 4.00 ±0.00 | 4.00 ±0.00 |
| <nobr>make_toast</nobr> | 16.00 | 26.00 | 16.00 ±0.00 | 26.00 ±0.00 |
| <nobr>make_toast_random</nobr> | 12.00 | 17.00 | 12.00 ±0.00 | 17.00 ±0.00 |
| <nobr>arrange_largest_number</nobr> | 80.00 | 83.80 | 80.00 ±0.00 | 83.80 ±0.00 |
| <nobr>arrange_largest_number_random</nobr> | 68.00 | 76.40 | 68.00 ±0.00 | 76.40 ±0.00 |
| <nobr>sort_nesting_dolls_by_size</nobr> | 84.00 | 84.00 | 84.00 ±0.00 | 84.00 ±0.00 |
| <nobr>sort_nesting_dolls_by_size_random</nobr> | 56.00 | 56.00 | 56.00 ±0.00 | 56.00 ±0.00 |
| <nobr>store_laptop_and_headphones</nobr> | 0.00 | 3.20 | 0.00 ±0.00 | 3.20 ±0.00 |
| <nobr>store_laptop_and_headphones_random</nobr> | 0.00 | 6.40 | 0.00 ±0.00 | 6.40 ±0.00 |
| <nobr>stack_blocks</nobr> | 100.00 | 100.00 | 100.00 ±0.00 | 100.00 ±0.00 |
| <nobr>stack_blocks_random</nobr> | 100.00 | 100.00 | 100.00 ±0.00 | 100.00 ±0.00 |
| **Avg Standard** | 45.33 | 50.47 | 45.33 ±0.00 | 50.47 ±0.00 |
| **Avg Random** | 41.00 | 46.07 | 41.00 ±0.00 | 46.07 ±0.00 |

## PyRUA_Lean

| Task | Seed 3 SR (%) | Seed 3 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: |
| <nobr>align_blocks</nobr> | 76.00 | 76.00 | 76.00 ±0.00 | 76.00 ±0.00 |
| <nobr>arrange_largest_number</nobr> | 74.00 | 80.10 | 74.00 ±0.00 | 80.10 ±0.00 |
| <nobr>build_tower</nobr> | 4.00 | 23.20 | 4.00 ±0.00 | 23.20 ±0.00 |
| <nobr>classify_objects</nobr> | 96.00 | 97.10 | 96.00 ±0.00 | 97.10 ±0.00 |
| <nobr>classify_objects_by_language</nobr> | 92.00 | 94.60 | 92.00 ±0.00 | 94.60 ±0.00 |
| <nobr>cover_blocks</nobr> | 100.00 | 100.00 | 100.00 ±0.00 | 100.00 ±0.00 |
| <nobr>deposit_coin</nobr> | 40.00 | 44.00 | 40.00 ±0.00 | 44.00 ±0.00 |
| <nobr>fasten_screws</nobr> | 68.00 | 80.60 | 68.00 ±0.00 | 80.60 ±0.00 |
| <nobr>fill_egg_holder</nobr> | 6.00 | 21.20 | 6.00 ±0.00 | 21.20 ±0.00 |
| <nobr>fill_pen_holder</nobr> | 8.00 | 23.50 | 8.00 ±0.00 | 23.50 ±0.00 |
| <nobr>fold_clothes</nobr> | 52.00 | 59.20 | 52.00 ±0.00 | 59.20 ±0.00 |
| <nobr>general_pickup</nobr> | 86.00 | 86.00 | 86.00 ±0.00 | 86.00 ±0.00 |
| <nobr>hang_mugs</nobr> | 12.00 | 27.70 | 12.00 ±0.00 | 27.70 ±0.00 |
| <nobr>imitate_sorting_sequence</nobr> | 92.00 | 92.40 | 92.00 ±0.00 | 92.40 ±0.00 |
| <nobr>insert_key</nobr> | 0.00 | 0.00 | 0.00 ±0.00 | 0.00 ±0.00 |
| <nobr>insert_tubes</nobr> | 50.00 | 66.40 | 50.00 ±0.00 | 66.40 ±0.00 |
| <nobr>make_kong</nobr> | 0.00 | 0.00 | 0.00 ±0.00 | 0.00 ±0.00 |
| <nobr>make_toast</nobr> | 14.00 | 21.50 | 14.00 ±0.00 | 21.50 ±0.00 |
| <nobr>match_and_pick_from_conveyor</nobr> | 86.00 | 86.00 | 86.00 ±0.00 | 86.00 ±0.00 |
| <nobr>organize_table</nobr> | 4.00 | 25.00 | 4.00 ±0.00 | 25.00 ±0.00 |
| <nobr>pack_objects_into_box</nobr> | 30.00 | 46.00 | 30.00 ±0.00 | 46.00 ±0.00 |
| <nobr>pick_from_conveyor_by_image</nobr> | 22.00 | 22.00 | 22.00 ±0.00 | 22.00 ±0.00 |
| <nobr>play_Xylophone</nobr> | 62.00 | 62.00 | 62.00 ±0.00 | 62.00 ±0.00 |
| <nobr>play_stacking_toy</nobr> | 54.00 | 78.20 | 54.00 ±0.00 | 78.20 ±0.00 |
| <nobr>play_tic_tac_toe</nobr> | 94.00 | 97.20 | 94.00 ±0.00 | 97.20 ±0.00 |
| <nobr>plug_in_charger</nobr> | 16.00 | 16.00 | 16.00 ±0.00 | 16.00 ±0.00 |
| <nobr>pour_balls_into_vase</nobr> | 66.00 | 66.00 | 66.00 ±0.00 | 66.00 ±0.00 |
| <nobr>pour_by_language</nobr> | 18.00 | 18.40 | 18.00 ±0.00 | 18.40 ±0.00 |
| <nobr>pour_liquid_into_cup</nobr> | 4.00 | 4.00 | 4.00 ±0.00 | 4.00 ±0.00 |
| <nobr>press_by_number</nobr> | 0.00 | 0.00 | 0.00 ±0.00 | 0.00 ±0.00 |
| <nobr>push_T</nobr> | 86.00 | 86.00 | 86.00 ±0.00 | 86.00 ±0.00 |
| <nobr>put_bottles_into_dustbin</nobr> | 88.00 | 92.20 | 88.00 ±0.00 | 92.20 ±0.00 |
| <nobr>solve_equation</nobr> | 72.00 | 72.00 | 72.00 ±0.00 | 72.00 ±0.00 |
| <nobr>sort_nesting_dolls_by_size</nobr> | 70.00 | 70.00 | 70.00 ±0.00 | 70.00 ±0.00 |
| <nobr>stack_blocks</nobr> | 100.00 | 100.00 | 100.00 ±0.00 | 100.00 ±0.00 |
| <nobr>stack_blocks_by_language</nobr> | 100.00 | 100.00 | 100.00 ±0.00 | 100.00 ±0.00 |
| <nobr>stack_bowls</nobr> | 70.00 | 73.90 | 70.00 ±0.00 | 73.90 ±0.00 |
| <nobr>store_laptop_and_headphones</nobr> | 0.00 | 4.80 | 0.00 ±0.00 | 4.80 ±0.00 |
| <nobr>store_tools_in_toolbox</nobr> | 0.00 | 15.50 | 0.00 ±0.00 | 15.50 ±0.00 |
| <nobr>swap_T</nobr> | 92.00 | 92.00 | 92.00 ±0.00 | 92.00 ±0.00 |
| <nobr>swap_blocks</nobr> | 40.00 | 40.00 | 40.00 ±0.00 | 40.00 ±0.00 |
| <nobr>sweep_blocks</nobr> | 6.00 | 6.00 | 6.00 ±0.00 | 6.00 ±0.00 |
| **Avg (per seed)** | 48.81 | 53.97 | 48.81 ±0.00 | 53.97 ±0.00 |

