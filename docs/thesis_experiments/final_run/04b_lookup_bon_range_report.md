# Test 04b Final Report: Lookup Best-of-N Range Check

## Data Source

```text
/home/oss/Downloads/thesis_tests_final_test4b
```

The folder `/home/oss/Downloads/thesis_tests_final_test34/04b_lookup_bon_ranges` exists but has no model result folders, so this report uses `thesis_tests_final_test4b`.

Completeness check:

| Model | Repeat | Expected configs | Completed configs | Rows | Status |
| --- | --- | --- | --- | --- | --- |
| `gemma4:e4b` | rep01 | 8 | 8 | 80 | included |
| `gemma4:e4b` | rep02 | 8 | 8 | 80 | included |
| `gemma4:e4b` | rep03 | 8 | 8 | 80 | included |

## Method Notes

This follow-up compares static lookup temperatures with Best-of-2 lookup ranges and `lookup_cot_n2` for `gemma4:e4b`. Expected missing GT score slots are counted as `0`.

## Executive Findings

- Static lookup temperatures are competitive with or better than Best-of-2 ranges in this run.
- Best-of-N should not be recommended unless the selector actually uses later candidates and the quality gain exceeds the extra energy.
- This test is mainly a methodological guardrail against over-crediting Best-of-N when the first candidate dominates.

## Overall Resource Use

| Model | Repeats | Rows | Mean quality | Mean sec / prompt | Mean kWh / prompt | Mean completion |
| --- | --- | --- | --- | --- | --- | --- |
| `gemma4:e4b` | 3 | 240 | 0.7494 | 144.1 | 0.00728 | 89.7% |

## Aggregate Results

| Config | Type | Quality | Delta quality | csv_iou | csv delta | Completion | Sec / prompt | kWh / prompt | Delta kWh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lookup_baseline` | baseline | 0.7070 | 0.0000 | 0.7919 | 0.0000 | 86.4% | 126.5 | 0.00639 | 0.00000 |
| `lookup_temp_0_3` | static_temperature | 0.7491 | +0.0420 | 0.8070 | +0.0151 | 87.7% | 128.4 | 0.00647 | 0.00008 |
| `lookup_temp_0_6` | static_temperature | 0.6732 | -0.0338 | 0.7305 | -0.0614 | 82.7% | 115.9 | 0.00580 | -0.00060 |
| `lookup_temp_0_8` | static_temperature | 0.7778 | +0.0708 | 0.8484 | +0.0565 | 95.1% | 143.3 | 0.00727 | 0.00088 |
| `lookup_bon_temp_n2_narrow` | best_of_n_range | 0.7372 | +0.0301 | 0.7851 | -0.0069 | 90.1% | 154.9 | 0.00774 | 0.00135 |
| `lookup_bon_temp_n2_medium` | best_of_n_range | 0.7584 | +0.0514 | 0.8243 | +0.0324 | 90.1% | 161.3 | 0.00817 | 0.00178 |
| `lookup_bon_temp_n2_wide` | best_of_n_range | 0.7384 | +0.0314 | 0.8175 | +0.0256 | 87.7% | 154.3 | 0.00778 | 0.00139 |
| `lookup_cot_n2` | cot_n2 | 0.8542 | +0.1472 | 0.9304 | +0.1385 | 97.5% | 168.0 | 0.00858 | 0.00218 |

## Selection Behavior

| Config | Non-first selections | Mean margin |
| --- | --- | --- |
| `lookup_bon_temp_n2_narrow` | 0 / 30 | 0.0000 |
| `lookup_bon_temp_n2_medium` | 0 / 30 | 0.0000 |
| `lookup_bon_temp_n2_wide` | 0 / 30 | 0.0000 |

## Figures

![quality](plots/test04b_v2_lookup_bon_range_quality.png)

![pareto](plots/test04b_v2_lookup_bon_range_pareto.png)

![selection](plots/test04b_v2_lookup_bon_selection.png)

## Thesis Interpretation

The report supports keeping lookup Best-of-N out of final recommendations unless it beats simpler static temperature settings and shows real non-first candidate usage.
