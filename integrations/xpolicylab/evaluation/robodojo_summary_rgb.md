# RoboDojo Evaluation Summary

## Overview (by capability dimension)

Progress † marks incomplete evaluation (126 = 42 tasks × 3 seeds). Cells show `mean ±std` across seeds; scores reflect completed cells only. Policies are sorted by overall Avg SR.

| Policy | Progress | Generalization SR (%) | Generalization Score | Precision SR (%) | Precision Score | Long-Horizon SR (%) | Long-Horizon Score | Memory SR (%) | Memory Score | Open SR (%) | Open Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyRUA_Lean | 33.3% † (42/126) | 45.83 ±0.00 | 50.65 ±0.00 | 37.25 ±0.00 | 47.02 ±0.00 | 48.00 ±0.00 | 63.26 ±0.00 | 63.00 ±0.00 | 63.00 ±0.00 | 64.00 ±0.00 | 67.09 ±0.00 | 51.62 ±0.00 | 58.20 ±0.00 |

## Generalization: Standard vs Random (25 episodes each)

Each cell shows `mean ±std` across seeds, averaged over the 12 Generalization tasks with completed data for that half. Δ = Random − Standard (mean SR). Sorted by Standard SR.

| Policy | Standard SR (%) | Standard Score | Random SR (%) | Random Score | Δ SR | Δ Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyRUA_Lean | 49.33 ±0.00 | 53.60 ±0.00 | 42.33 ±0.00 | 47.70 ±0.00 | -7.00 | -5.90 |

### Per-task Standard vs Random

Additional view only: each of the 12 Generalization tasks is listed as the 25-episode standard split and its `_random` sibling. Columns match the per-policy all-task tables. Existing merged 50-episode scores are unchanged.

#### PyRUA_Lean

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>stack_bowls</nobr> | 84 | 85.80 |  |  |  |  |  |  |
| <nobr>stack_bowls_random</nobr> | 72 | 75.00 |  |  |  |  |  |  |
| <nobr>push_T</nobr> | 76 | 76.00 |  |  |  |  |  |  |
| <nobr>push_T_random</nobr> | 76 | 76.00 |  |  |  |  |  |  |
| <nobr>pack_objects_into_box</nobr> | 44 | 58.40 |  |  |  |  |  |  |
| <nobr>pack_objects_into_box_random</nobr> | 20 | 37.40 |  |  |  |  |  |  |
| <nobr>fold_clothes</nobr> | 88 | 90.40 |  |  |  |  |  |  |
| <nobr>fold_clothes_random</nobr> | 52 | 59.20 |  |  |  |  |  |  |
| <nobr>hang_mugs</nobr> | 0 | 20.20 |  |  |  |  |  |  |
| <nobr>hang_mugs_random</nobr> | 4 | 20.60 |  |  |  |  |  |  |
| <nobr>sweep_blocks</nobr> | 16 | 16.00 |  |  |  |  |  |  |
| <nobr>sweep_blocks_random</nobr> | 20 | 20.00 |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup</nobr> | 4 | 4.00 |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup_random</nobr> | 12 | 12.00 |  |  |  |  |  |  |
| <nobr>make_toast</nobr> | 12 | 18.00 |  |  |  |  |  |  |
| <nobr>make_toast_random</nobr> | 0 | 10.00 |  |  |  |  |  |  |
| <nobr>arrange_largest_number</nobr> | 92 | 94.40 |  |  |  |  |  |  |
| <nobr>arrange_largest_number_random</nobr> | 80 | 83.80 |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size</nobr> | 76 | 76.00 |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size_random</nobr> | 72 | 72.00 |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones</nobr> | 0 | 4.00 |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones_random</nobr> | 0 | 6.40 |  |  |  |  |  |  |
| <nobr>stack_blocks</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>stack_blocks_random</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| **Avg Standard** | 49.3 | 53.60 |  |  |  |  | 49 ±0 | 53.60 ±0.00 |
| **Avg Random** | 42.3 | 47.70 |  |  |  |  | 42 ±0 | 47.70 ±0.00 |

## PyRUA_Lean

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>align_blocks</nobr> | 78 | 78.00 |  |  |  |  |  |  |
| <nobr>arrange_largest_number</nobr> | 86 | 89.10 |  |  |  |  |  |  |
| <nobr>build_tower</nobr> | 2 | 24.20 |  |  |  |  |  |  |
| <nobr>classify_objects</nobr> | 98 | 98.80 |  |  |  |  |  |  |
| <nobr>classify_objects_by_language</nobr> | 92 | 95.20 |  |  |  |  |  |  |
| <nobr>cover_blocks</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>deposit_coin</nobr> | 34 | 38.80 |  |  |  |  |  |  |
| <nobr>fasten_screws</nobr> | 66 | 83.00 |  |  |  |  |  |  |
| <nobr>fill_egg_holder</nobr> | 2 | 19.80 |  |  |  |  |  |  |
| <nobr>fill_pen_holder</nobr> | 14 | 46.10 |  |  |  |  |  |  |
| <nobr>fold_clothes</nobr> | 70 | 74.80 |  |  |  |  |  |  |
| <nobr>general_pickup</nobr> | 98 | 98.00 |  |  |  |  |  |  |
| <nobr>hang_mugs</nobr> | 2 | 20.40 |  |  |  |  |  |  |
| <nobr>imitate_sorting_sequence</nobr> | 98 | 98.00 |  |  |  |  |  |  |
| <nobr>insert_key</nobr> | 0 | 15.00 |  |  |  |  |  |  |
| <nobr>insert_tubes</nobr> | 44 | 63.20 |  |  |  |  |  |  |
| <nobr>make_kong</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>make_toast</nobr> | 6 | 14.00 |  |  |  |  |  |  |
| <nobr>match_and_pick_from_conveyor</nobr> | 82 | 82.00 |  |  |  |  |  |  |
| <nobr>organize_table</nobr> | 34 | 76.00 |  |  |  |  |  |  |
| <nobr>pack_objects_into_box</nobr> | 32 | 47.90 |  |  |  |  |  |  |
| <nobr>pick_from_conveyor_by_image</nobr> | 34 | 34.00 |  |  |  |  |  |  |
| <nobr>play_Xylophone</nobr> | 64 | 64.00 |  |  |  |  |  |  |
| <nobr>play_stacking_toy</nobr> | 66 | 82.20 |  |  |  |  |  |  |
| <nobr>play_tic_tac_toe</nobr> | 94 | 98.50 |  |  |  |  |  |  |
| <nobr>plug_in_charger</nobr> | 12 | 12.00 |  |  |  |  |  |  |
| <nobr>pour_balls_into_vase</nobr> | 76 | 76.00 |  |  |  |  |  |  |
| <nobr>pour_by_language</nobr> | 20 | 21.00 |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup</nobr> | 8 | 8.00 |  |  |  |  |  |  |
| <nobr>press_by_number</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>push_T</nobr> | 76 | 76.00 |  |  |  |  |  |  |
| <nobr>put_bottles_into_dustbin</nobr> | 76 | 84.70 |  |  |  |  |  |  |
| <nobr>solve_equation</nobr> | 90 | 90.00 |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size</nobr> | 74 | 74.00 |  |  |  |  |  |  |
| <nobr>stack_blocks</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>stack_blocks_by_language</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>stack_bowls</nobr> | 78 | 80.40 |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones</nobr> | 0 | 5.20 |  |  |  |  |  |  |
| <nobr>store_tools_in_toolbox</nobr> | 0 | 20.50 |  |  |  |  |  |  |
| <nobr>swap_T</nobr> | 98 | 98.00 |  |  |  |  |  |  |
| <nobr>swap_blocks</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>sweep_blocks</nobr> | 18 | 18.00 |  |  |  |  |  |  |
| **Avg (per seed)** | 50.5 | 57.26 |  |  |  |  | 51 ±0 | 57.26 ±0.00 |
