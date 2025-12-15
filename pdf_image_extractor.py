import json
import shutil
import subprocess
import tempfile
import re
from pathlib import Path


class PdfImageExtractor:
    def __init__(self, tool_root):
        self.tool_root = Path(tool_root).resolve()

    def run_extractor(self, pdf, tmp):
        cmd = (
            f'sbt "runMain org.allenai.pdffigures2.FigureExtractorBatchCli '
            f'{pdf} -s {tmp}/stats.json -m {tmp}/img/ -d {tmp}/json/"'
        )
        subprocess.run(cmd, cwd=self.tool_root, shell=True, capture_output=False)

    def _extract_figure_number(self, caption):
        """
        Extracts figure number like:
        Figure 3
        Fig. 2a
        fig 10B
        """
        match = re.search(
            r'\b(fig|figure)\.?\s*(\d+[a-zA-Z]?)',
            caption,
            re.IGNORECASE
        )
        if match:
            return match.group(2).lower()
        return None

    def extract_all_figures(self, pdf_path, out_dir):
        pdf_path = Path(pdf_path).resolve()
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(exist_ok=True, parents=True)

        fig_word_pattern = re.compile(r"\b(fig|fig\.|figure|figure\.)\b", re.IGNORECASE)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shutil.copy2(pdf_path, tmp / "input.pdf")
            (tmp / "img").mkdir()
            (tmp / "json").mkdir()

            self.run_extractor(tmp / "input.pdf", tmp)

            metadata_file = tmp / "json" / "input.json"
            if not metadata_file.exists():
                return

            metadata = json.load(open(metadata_file))

            all_fig_meta = []

            for fig in metadata:

                # ---- exclude tables ----
                if fig.get("figureType", "").lower() == "table":
                    continue

                caption = fig.get("caption", "")
                if not fig_word_pattern.search(caption):
                    continue

                fig_no = self._extract_figure_number(caption)
                if not fig_no:
                    continue  # skip if no explicit figure number

                render_url = fig.get("renderURL")
                if not render_url:
                    continue

                img_src = tmp / "img" / Path(render_url).name
                if not img_src.exists():
                    continue

                final_img = out_dir / f"figure_{fig_no}{img_src.suffix}"

                # avoid overwrite (rare but safe)
                counter = 1
                while final_img.exists():
                    final_img = out_dir / f"figure_{fig_no}_{counter}{img_src.suffix}"
                    counter += 1

                shutil.copy2(img_src, final_img)

                fig = fig.copy()
                fig.pop("renderURL", None)
                fig["filename"] = final_img.name

                all_fig_meta.append(fig)

            json.dump(
                all_fig_meta,
                open(out_dir / "figures_all.json", "w"),
                indent=4
            )
