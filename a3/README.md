# AIST4010 Assignment 3

`A3_report.pdf` answers Questions 1–5 and includes the plots and prediction times for Question 4. Blackboard asks for the PDF.

## Reproduce Question 4

Run the following from this folder with CUDA-enabled PyTorch, NumPy and Matplotlib installed. The recorded run used PyTorch 2.9.0 on an NVIDIA GeForce RTX 5090.

```sh
python main_q4.py
```

`utils_q4.py` defines the dataset and four models. The script uses seed 42, tunes the RNN on an 800/200 chronological split and refits on all 990 windows. It writes the two figures, all 200 predictions (`forecasts.csv`) and the settings, errors and measured times (`metrics.json`) to `results/`. Timings vary between runs.

The report source is `A3_report.tex`. To rebuild it with the included figures, run `pdflatex A3_report.tex` twice.
