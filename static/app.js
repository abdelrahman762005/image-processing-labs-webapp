const image1Input = document.getElementById("image1");
const image2Input = document.getElementById("image2");
const resultImg = document.getElementById("preview-result");
const originalImg = document.getElementById("preview-original");
const secondImg = document.getElementById("preview-second");
const secondWrap = document.getElementById("preview-second-wrap");
const histImg = document.getElementById("preview-hist");
const histContainer = document.getElementById("hist-container");
const statusEl = document.getElementById("status");
const metaEl = document.getElementById("meta");
const buttons = Array.from(document.querySelectorAll("button[data-op]"));
const NEEDS_SECOND_IMAGE = new Set([
  "blend_add",
  "blend_weighted",
  "subtract_a_b",
  "subtract_b_a",
  "difference_abs",
  "metrics_compare",
]);

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function formatMetricValue(value) {
  if (value === undefined || value === null) {
    return "-";
  }
  if (!Number.isFinite(value)) {
    return value > 0 ? "Infinity" : "-Infinity";
  }
  return value.toFixed(4);
}

function renderMeta(data) {
  const lines = [];
  if (data.metrics) {
    lines.push(`MSE: ${formatMetricValue(data.metrics.mse)}`);
    lines.push(`PSNR: ${formatMetricValue(data.metrics.psnr)} dB`);
    lines.push(`SNR: ${formatMetricValue(data.metrics.snr)} dB`);
  }
  if (typeof data.threshold === "number") {
    lines.push(`Chosen threshold: ${data.threshold}`);
  }
  if (data.message) {
    lines.push(data.message);
  }

  metaEl.innerHTML = lines.length ? lines.join("<br>") : "No extra metadata.";
}

function previewFile(file, targetImg) {
  const reader = new FileReader();
  reader.onload = (event) => {
    targetImg.src = event.target.result;
  };
  reader.readAsDataURL(file);
}

function disableAllButtons(disabled) {
  for (const button of buttons) {
    button.disabled = disabled;
  }
}

function setBusyButton(activeOp) {
  for (const button of buttons) {
    button.classList.toggle("busy", button.dataset.op === activeOp);
  }
}

function flashImage(imgElement) {
  imgElement.classList.remove("flash");
  void imgElement.offsetWidth;
  imgElement.classList.add("flash");
}

function buildFormData(operation) {
  const formData = new FormData();
  formData.append("image1", image1Input.files[0]);
  if (image2Input.files.length) {
    formData.append("image2", image2Input.files[0]);
  }

  const fields = [
    "constant",
    "threshold",
    "kernel_size",
    "sigma",
    "alpha",
    "noise_amount",
    "binary_threshold",
  ];

  for (const field of fields) {
    const input = document.getElementById(field);
    formData.append(field, input.value);
  }

  formData.append("operation", operation);
  return formData;
}

async function runOperation(operation) {
  if (!image1Input.files.length) {
    setStatus("Upload Image A first.", true);
    return;
  }
  if (NEEDS_SECOND_IMAGE.has(operation) && !image2Input.files.length) {
    setStatus("This operation needs Image B.", true);
    return;
  }

  setStatus(`Processing: ${operation}`);
  disableAllButtons(true);
  setBusyButton(operation);

  try {
    const response = await fetch("/process", {
      method: "POST",
      body: buildFormData(operation),
    });

    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "Failed to process image.");
    }

    resultImg.src = `data:image/png;base64,${data.result_image}`;
    flashImage(resultImg);
    renderMeta(data);

    if (data.histogram_image) {
      histImg.src = `data:image/png;base64,${data.histogram_image}`;
      flashImage(histImg);
      histContainer.classList.remove("hidden");
    } else {
      histContainer.classList.add("hidden");
      histImg.src = "";
    }

    setStatus(`Done: ${operation}`);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    disableAllButtons(false);
    setBusyButton(null);
  }
}

image1Input.addEventListener("change", () => {
  if (image1Input.files.length) {
    previewFile(image1Input.files[0], originalImg);
  }
});

image2Input.addEventListener("change", () => {
  if (image2Input.files.length) {
    previewFile(image2Input.files[0], secondImg);
    secondWrap.classList.remove("hidden");
    return;
  }
  secondImg.src = "";
  secondWrap.classList.add("hidden");
});

for (const button of buttons) {
  button.addEventListener("click", () => runOperation(button.dataset.op));
}
