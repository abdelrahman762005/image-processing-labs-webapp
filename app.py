from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from image_ops import array_to_base64_png, load_image_from_upload, process_operation


app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/process")
def process_image():
    operation = request.form.get("operation", "").strip()
    if not operation:
        return jsonify({"ok": False, "error": "Missing operation name."}), 400

    image1_file = request.files.get("image1")
    if image1_file is None or image1_file.filename == "":
        return jsonify({"ok": False, "error": "Image A is required."}), 400

    try:
        image1 = load_image_from_upload(image1_file)

        image2_file = request.files.get("image2")
        image2 = None
        if image2_file is not None and image2_file.filename != "":
            image2 = load_image_from_upload(image2_file)

        result = process_operation(operation, image1, image2, request.form)
        payload = {
            "ok": True,
            "operation": operation,
            "result_image": array_to_base64_png(result["result"]),
            "metrics": result.get("metrics", {}),
        }

        if "histogram_image" in result:
            payload["histogram_image"] = array_to_base64_png(result["histogram_image"])
        if "threshold" in result:
            payload["threshold"] = result["threshold"]
        if "message" in result:
            payload["message"] = result["message"]

        return jsonify(payload)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception:
        return jsonify({"ok": False, "error": "Unexpected server error."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

