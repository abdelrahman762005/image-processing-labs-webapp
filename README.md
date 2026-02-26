# Image Processing Labs Website

Web app that applies multiple image-processing lab operations from a single UI.

## Run

1. Create/activate virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start server:
   ```bash
   python app.py
   ```
4. Open:
   - `http://127.0.0.1:5000`

## Notes

- Upload `Image A` for all operations.
- Upload `Image B` for two-image operations (blend/subtract/compare).
- Processing algorithms are implemented manually in `image_ops.py` (from scratch style), not via high-level image-processing library filters.
- Large images are resized using manual nearest-neighbor resize for faster processing.

