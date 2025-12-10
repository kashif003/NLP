import json
import shutil
import subprocess
import tempfile
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

    def extract(self, pdf_path, page, out_dir, name):
        pdf_path = Path(pdf_path).resolve()
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(exist_ok=True, parents=True)

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
            fig = next((m for m in metadata if m["page"] == page - 1), None)
            if not fig:
                return

            img_src = tmp / "img" / Path(fig["renderURL"]).name
            final_img = out_dir / f"{name}{img_src.suffix}"
            shutil.copy2(img_src, final_img)

            fig.pop("renderURL", None)
            fig["filename"] = final_img.name
            json.dump([fig], open(out_dir / f"{name}.json", "w"), indent=4)
