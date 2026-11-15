# Report (Deliverable 10)

LaTeX source of the course-project report, compiled on **Overleaf** (pdfLaTeX, no shell-escape).
**No number in the report is typed by hand**: every value, table and figure comes from the
pipeline outputs through `make_numbers.py`.

## Updating the report

```bash
make all                                  # 1. from the repo root: full pipeline, steps 00 -> 12
uv run python report/make_numbers.py      # 2. analysis, numbers, tables, figures and the zip
```

3. On Overleaf: **New Project → Upload Project** and choose `report/overleaf.zip`
   (or, in an existing project, upload the files from the zip to overwrite `generated/` and
   `figures/`). Compiler: pdfLaTeX. Main document: `HotHotChips_2024112008_2024102068_2023112006_2024102055.tex`.

`make_numbers.py` prints a `MISSING` list if a source file or column is absent (otherwise
"nothing missing"). The affected macro or table then shows a red `[TODO: missing: ...]` in the PDF
instead of a wrong number, so search the PDF for "TODO" before submitting.

Prose is written in `sections/*.tex`. If you edit text on Overleaf, copy the changed section files
back into the repo before the next upload, because the zip is always built from the repo.

## Layout

| Path | Content |
|---|---|
| `HotHotChips_2024112008_2024102068_2023112006_2024102055.tex` | Main document: preamble, title page, abstract, `\input` of the sections in the brief's order |
| `sections/01_setup.tex` ... `08_appendix.tex` | One file per report section (brief §6, item 10) |
| `sections/09_contributions.tex` | Individual contributions (last section) |
| `generated/numbers.tex` | One `\newcommand` per quoted number (generated, committed) |
| `generated/tab_*.tex` | Booktabs `tabular`s only (no caption/label): `kirchhoff` (D3), `models` (D4, per case), `models_full` (D4, per current), `chosen` (D5), `learning` (learning curve), `gates` (D6), `gate_worst`, `circuit` (D8) |
| `figures/*.pdf` | Copies of the `results/plots/*.pdf` the report uses (generated, committed) |
| `overleaf.zip` | Main `.tex` + `sections/` + `generated/` + `figures/` (generated, not committed) |
| `make_numbers.py` | The generator |
| `make_analysis.py` | Section 6 error analysis (called by `make_numbers.py`); writes `generated/analysis/` (CSVs + `error_vs_corner.pdf`, not in the zip) |
| `analysis/` | One-off validation studies (SPICE oracle, fresh corners) with their scripts and summaries; see `analysis/README.md` |
| `HotHotChips_2024112008_2024102068_2023112006_2024102055.pdf` | The final compiled report (D10), built from the main `.tex` of the same name |

## Using the numbers

```latex
The gate-level mean accuracy is \GateMeanAcc{} (median error \GateMedianErr{}).
\begin{table}[H]\centering\small
  \input{generated/tab_gates}
  \caption{Gate results (D6).}\label{tab:gates}
\end{table}
\begin{figure}[H]\centering
  \includegraphics[width=0.6\linewidth]{gate_parity}
  \caption{Gate parity plot (D7).}\label{fig:gateparity}
\end{figure}
```

Macro names use letters only (`p95` -> `Pninetyfive`, `R^2` -> `Rtwo`, `nmos_on` -> `NmosOn`,
`id` -> `Id`), for example `\TestMedianErrNmosOffId` or `\GatePninetyfiveErr`. Learning-curve
values take arguments: `\LCMedian{nmos_off}{id}{7000}`, `\LCPninetyfive{pmos_on}{ig}{21000}`.
See `generated/numbers.tex` for the full list.
