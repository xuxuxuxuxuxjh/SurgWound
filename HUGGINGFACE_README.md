---
license: cc-by-sa-4.0
task_categories:
- visual-question-answering
- question-answering
- text-generation
language:
- en
tags:
- medical
- surgical
- wound
- multimodal
pretty_name: SurgWound
size_categories:
- 1K<n<10K
arxiv: "2508.15189"
---

# SurgWound

SurgWound is an open dataset and benchmark for multimodal surgical-wound analysis. It contains **686 de-identified surgical-wound images** with eight image-level clinical attributes annotated under a difficulty-aware review protocol involving a team of three professional surgeons. On top of the image annotations, SurgWound-Bench defines two tasks:

- **SurgWound-VQA:** multiple-choice visual question answering for wound characteristics and diagnostic outcomes.
- **SurgWound-Report:** generation of a structured medical report from a wound image.

The accompanying three-stage WoundQwen framework is described in the published article and its [GitHub repository](https://github.com/xuxuxuxuxuxjh/SurgWound).

## Paper and links

- **Published article:** [SurgWound-Bench: a benchmark for surgical wound diagnosis](https://doi.org/10.1038/s41746-026-02791-3), *npj Digital Medicine*, published online 23 May 2026.
- **Code and documentation:** [github.com/xuxuxuxuxuxjh/SurgWound](https://github.com/xuxuxuxuxuxjh/SurgWound)
- **Historical preprint:** [arXiv:2508.15189](https://arxiv.org/abs/2508.15189)

## Dataset composition

The image split is fixed at 480/69/137 (approximately 7:1:2), with image-level disjoint train, validation, and test sets.

| Split | Unique images | VQA records | Report records | Files |
| --- | ---: | ---: | ---: | --- |
| Train | 480 | 3,435 | 480 | train_question.json, train_report.json |
| Validation | 69 | 500 | 69 | val_question.json, val_report.json |
| Test | 137 | 979 | 137 | test_question.json, test_report.json |
| **Total** | **686** | **4,914** | **686** | 6 JSON files |

VQA records whose image-level annotation is `Uncertain` are omitted for that attribute. Therefore, the VQA counts are lower than eight records per image for some fields. `Uncertain` may still appear in a VQA record's `options` list; it is simply not emitted as the answer for an omitted annotation. The VQA record totals by attribute over all splits are:

| Attribute | Records |
| --- | ---: |
| Wound Location | 524 |
| Healing Status | 686 |
| Closure Method | 492 |
| Exudate Type | 626 |
| Erythema | 660 |
| Edema | 554 |
| Urgency Level | 686 |
| Infection Risk Assessment | 686 |

`train_question.json`, `val_question.json`, `test_question.json`, and `train_report.json` use Git LFS; `val_report.json` and `test_report.json` are regular Git blobs. All six are downloadable through the Hub APIs. The embedded base64 images make the complete download approximately 452 MB; loading a whole file into memory may require considerably more RAM. The record counts above describe the release documented by this card; check the dataset revision when building a reproducible experiment.

## Label schema

The canonical field values and answer vocabularies are:

| Field | Values |
| --- | --- |
| Wound Location | Abdomen, Patella, Ankle, Facial region, Manus, Cervical region, Other, Uncertain |
| Healing Status | Healed, Not Healed |
| Closure Method | Invisible, Sutures, Staples, Adhesives, Uncertain |
| Exudate Type | Non-existent, Serous, Sanguineous, Purulent, Seropurulent, Uncertain |
| Erythema | Non-existent, Existent, Uncertain |
| Edema | Non-existent, Existent, Uncertain |
| Infection Risk Assessment | Low, Medium, High |
| Urgency Level | Home Care (Green): Manage with routine care; Clinic Visit (Yellow): Requires professional evaluation within 48 hours; Emergency Care (Red): Seek immediate medical attention |

Wound Location, Healing Status, Closure Method, Exudate Type, Erythema, and Edema describe observable wound characteristics. Infection Risk Assessment and Urgency Level are clinical assessment labels. The raw VQA files include location records, but the paper's main WoundQwen analysis focuses on seven non-location prediction sub-tasks because location is supplied as known clinical context.

## File format and schema

Each JSON file is a JSON **array**, not JSONL. Records share the following schema; `options` is present for VQA records and omitted from report records:

| Key | Description |
| --- | --- |
| id | Unique record identifier, for example 76.jpg_closure_method or 606.jpg_report. |
| image_name | Randomized JPEG name shared by records for the same image. |
| field | One of the fields listed above, or Medical Report for report records. |
| task_type | multi_choice for VQA; report_generation for reports. |
| image | Base64-encoded JPEG bytes, without a data-URI prefix. The payload is repeated across records for the same image. |
| question | The VQA question or report-generation prompt. |
| options | List of answer options for VQA records; absent from report records. |
| answer | Ground-truth label for VQA or the reviewed clinical report for report generation. |

The answer text in report files is generated from the image and structured annotations and then reviewed/refined by surgeons. These reports are research references, not patient-specific medical advice.

### Example VQA record

~~~json
{
  "id": "76.jpg_closure_method",
  "image_name": "76.jpg",
  "field": "Closure Method",
  "task_type": "multi_choice",
  "image": "<base64-encoded JPEG>",
  "question": "What is the closure method of this surgical wound?",
  "options": ["Invisible", "Sutures", "Staples", "Adhesives", "Uncertain"],
  "answer": "Sutures"
}
~~~

### Example report record

~~~json
{
  "id": "606.jpg_report",
  "image_name": "606.jpg",
  "field": "Medical Report",
  "task_type": "report_generation",
  "image": "<base64-encoded JPEG>",
  "question": "Given a surgical wound image, generate a detailed medical report ...",
  "answer": "The wound assessment ..."
}
~~~

## Loading the data

Use json.load because the raw files are arrays. The following example downloads only one split and decodes its first image:

```bash
python -m pip install -U huggingface_hub pillow
hf download xuxuxuxuxu/SurgWound --repo-type dataset --local-dir ./SurgWound
```

~~~python
import base64
import json
from io import BytesIO
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image

path = hf_hub_download(
    repo_id="xuxuxuxuxu/SurgWound",
    repo_type="dataset",
    filename="test_question.json",
)
rows = json.loads(Path(path).read_text(encoding="utf-8"))
image = Image.open(BytesIO(base64.b64decode(rows[0]["image"]))).convert("RGB")
print(rows[0]["field"], rows[0]["answer"], image.size)
~~~

To create a datasets.Dataset after loading a manageable split:

~~~python
from datasets import Dataset

dataset = Dataset.from_list(rows)
~~~

The GitHub processing scripts are a legacy local workflow and expect JSONL records plus decoded JPEG files. They also contain blank or machine-specific paths and older spreadsheet field names (Wound Status and Exudate Characteristics). Adapt those scripts or normalize the fields before using them with this release; the canonical HF names are Healing Status and Exudate Type. Report-generation `question` strings retain the same legacy prose, but this does not change the canonical attributes. When evaluating at the image level, group records by `image_name`; repeated VQA records for one image are not independent images.

## Dataset creation

### Source data

Images were collected from publicly available content on RedNote, Twitter, Facebook, Instagram, and Reddit using domain-specific hashtags/keywords (for example, #surgicalwoundinfection and postoperative wound), as well as relevant accounts belonging to surgeons and other medical professionals. No restricted clinical records were used.

### Filtering and annotation

1. **AI-assisted filtering:** GPT-4o was used to screen for visible surgical wounds and image quality.
2. **Redundancy and expert review:** perceptual hashing was used to flag near duplicates; three surgeons reviewed the remaining images and removed low-quality, redundant, or non-surgical cases.
3. **Difficulty-aware annotation:** cases judged low difficulty by unanimous low-risk predictions from three MLLMs were assigned to one randomly selected surgeon. Higher-difficulty cases were independently annotated by two surgeons; disagreements were adjudicated by a third surgeon.
4. **Report construction:** GPT-4o generated an initial report from the image and structured labels, after which surgeons reviewed and refined the report.

The paper reports mean inter-annotator agreement of 0.931 and Cohen's kappa of 0.866 across the eight attributes. See the article and supplementary information for per-label agreement and the complete curation protocol.

## Privacy, ethics, and sensitive content

The images were publicly accessible but may depict human bodies and postoperative wounds. Before release, the curation process removed or obscured facial regions, tattoos, usernames, watermarks, and social-media identifiers; images were randomly renamed and associated text was screened for identifiable information. The study followed the Declaration of Helsinki and was reviewed by The Ohio State University Institutional Review Board as **Exempt Category 4**. Informed consent was not required because the research used publicly available, irreversibly de-identified data.

De-identification does not guarantee zero re-identification risk. Do not attempt to identify, contact, or re-identify individuals, and do not combine the data with external information for that purpose. Users remain responsible for complying with applicable privacy, copyright, platform, and data-protection requirements. Source-platform terms and third-party rights may impose additional restrictions beyond the dataset license.

## Intended use, out-of-scope use, and limitations

**Intended use:** research and education on multimodal surgical-wound VQA, report generation, representation learning, and benchmark evaluation.

**Out of scope:** autonomous diagnosis, triage, treatment selection, patient communication, or deployment in clinical care without appropriate clinical validation, governance, and qualified human oversight. The labels are research annotations and are not medical advice.

Known limitations include:

- strong class imbalance (18 Emergency Care images and 23 high-infection-risk images overall; only four emergency cases are in the test split);
- predominance of low-risk/home-care cases (573/686 low infection risk; 600/686 home care);
- possible selection, source-platform, copyright, camera, skin-tone, and demographic biases in publicly shared images;
- no longitudinal outcomes or complete patient-level clinical metadata; and
- omission of Uncertain labels from the corresponding VQA questions.

External and prospective clinical validation is required before any clinical use.

## License

This dataset is released under the **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** license: [creativecommons.org/licenses/by-sa/4.0](https://creativecommons.org/licenses/by-sa/4.0/). Please provide attribution, preserve the license notice, and indicate changes. The published article is separately licensed under **CC BY-NC-ND 4.0**; that article license does not replace the dataset license. Public-source images may also be subject to the rights of the original uploaders and platforms.

## Citation

Please cite the published article when using this dataset or benchmark:

~~~bibtex
@article{xu2026surgwoundbench,
  author  = {Xu, Jiahao and Yin, Changchang and Chatzipanagiotou, Odysseas P. and
             Tsilimigras, Diamantis I. and Clear, Kevin and Yao, Bingsheng and
             Cao, Weidan and Wang, Dakuo and Pawlik, Timothy M. and Zhang, Ping},
  title   = {SurgWound-Bench: a benchmark for surgical wound diagnosis},
  journal = {npj Digital Medicine},
  year    = {2026},
  month   = may,
  day     = {23},
  doi     = {10.1038/s41746-026-02791-3},
  url     = {https://doi.org/10.1038/s41746-026-02791-3}
}
~~~

## More information

- [GitHub implementation and documentation](https://github.com/xuxuxuxuxuxjh/SurgWound)
- [Published article](https://www.nature.com/articles/s41746-026-02791-3)
- [Supplementary information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41746-026-02791-3/MediaObjects/41746_2026_2791_MOESM1_ESM.pdf)
