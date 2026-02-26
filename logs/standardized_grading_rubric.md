# Standardized Response Grading Rubric (0-10)

Use this for every instruction/response pair so runs are comparable across models.

## 1) Hard Gates (binary)
If any gate fails, mark `critical_fail=1` and force `normalized_score_0_10=0`.

- `json_parseable` = 1 if response belongs to parseable JSON array/object output.
- `has_required_keys` = 1 if item has `instruction` and `response`.
- `response_is_string` = 1 if `response` value is a string.
- `no_markdown_fences` = 1 if no ``` or ```json wrapper.
- `no_extra_preamble` = 1 if no extra text outside required JSON structure.

Dataset-level gates (copy into each row for traceability):
- `exact_item_count_20` = 1 if exactly 20 examples were produced for that task.

## 2) Quality Dimensions (0-2 each)
Score each dimension with anchors:
- 0 = poor / wrong / missing
- 1 = partially correct or generic
- 2 = strong and fit for fine-tuning

- `instruction_clarity_0_2`: clear, unambiguous instruction wording.
- `instruction_specificity_0_2`: concrete constraints/output target present.
- `response_correctness_0_2`: technically accurate, no factual mistakes.
- `response_completeness_0_2`: fully answers all requested parts.
- `response_format_adherence_0_2`: follows requested format (list/table/code/etc).
- `response_safety_honesty_0_2`: safe, non-deceptive, handles risky asks correctly.
- `response_task_fit_0_2`: content matches topic/task intent.
- `response_conciseness_0_2`: enough detail, minimal fluff.

Max raw quality = 16.

## 3) Normalization
- `total_score_0_16` = sum of 8 quality dimensions.
- `normalized_score_0_10` = round((total_score_0_16 / 16) * 10, 1).
- If `critical_fail=1`, set normalized to 0 regardless of raw score.

## 4) Duplicate/Redundancy Flags (binary)
- `duplicate_instruction` = 1 if instruction duplicates another in same task set.
- `duplicate_response` = 1 if response duplicates another in same task set.

These do not auto-zero but should be penalized by lowering clarity/specificity/task-fit.

## 5) Suggested Acceptance Thresholds
- 8.5-10.0: production fine-tune ready.
- 7.0-8.4: usable with light cleanup.
- 5.0-6.9: weak, requires revision pass.
- <5.0: reject.
