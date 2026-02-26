from __future__ import annotations

import base64
import io
import math
from typing import Any, Dict, Mapping, Optional

import numpy as np
from PIL import Image


MAX_DIMENSION = 900


def _clip_byte(value: float) -> int:
    if value < 0:
        return 0
    if value > 255:
        return 255
    return int(value)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sanitize_odd_size(value: Any, default: int = 3, minimum: int = 3, maximum: int = 21) -> int:
    size = _to_int(value, default)
    if size < minimum:
        size = minimum
    if size > maximum:
        size = maximum
    if size % 2 == 0:
        size += 1
    return size


def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return image.astype(np.uint8)
    h, w = image.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            pixel = int(image[i, j])
            rgb[i, j, 0] = pixel
            rgb[i, j, 1] = pixel
            rgb[i, j, 2] = pixel
    return rgb


def to_gray_average(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb.astype(np.uint8)
    h, w, _ = rgb.shape
    gray = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            r = int(rgb[i, j, 0])
            g = int(rgb[i, j, 1])
            b = int(rgb[i, j, 2])
            gray[i, j] = _clip_byte((r + g + b) / 3.0)
    return gray


def to_gray_weighted(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb.astype(np.uint8)
    h, w, _ = rgb.shape
    gray = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            r = int(rgb[i, j, 0])
            g = int(rgb[i, j, 1])
            b = int(rgb[i, j, 2])
            gray[i, j] = _clip_byte(0.3 * r + 0.59 * g + 0.11 * b)
    return gray


def extract_channel_tinted(rgb: np.ndarray, channel_idx: int) -> np.ndarray:
    h, w, _ = rgb.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            out[i, j, channel_idx] = int(rgb[i, j, channel_idx])
    return out


def extract_channel_gray(rgb: np.ndarray, channel_idx: int) -> np.ndarray:
    h, w, _ = rgb.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            out[i, j] = int(rgb[i, j, channel_idx])
    return out


def resize_nearest(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    src_h = image.shape[0]
    src_w = image.shape[1]
    if src_h == target_h and src_w == target_w:
        return image.copy()

    if image.ndim == 2:
        out = np.zeros((target_h, target_w), dtype=np.uint8)
        for i in range(target_h):
            src_i = int(i * src_h / target_h)
            if src_i >= src_h:
                src_i = src_h - 1
            for j in range(target_w):
                src_j = int(j * src_w / target_w)
                if src_j >= src_w:
                    src_j = src_w - 1
                out[i, j] = int(image[src_i, src_j])
        return out

    channels = image.shape[2]
    out = np.zeros((target_h, target_w, channels), dtype=np.uint8)
    for i in range(target_h):
        src_i = int(i * src_h / target_h)
        if src_i >= src_h:
            src_i = src_h - 1
        for j in range(target_w):
            src_j = int(j * src_w / target_w)
            if src_j >= src_w:
                src_j = src_w - 1
            for c in range(channels):
                out[i, j, c] = int(image[src_i, src_j, c])
    return out


def limit_image_size(image: np.ndarray, max_dim: int = MAX_DIMENSION) -> np.ndarray:
    h = image.shape[0]
    w = image.shape[1]
    if h <= max_dim and w <= max_dim:
        return image

    if h >= w:
        target_h = max_dim
        target_w = int((w * max_dim) / h)
    else:
        target_w = max_dim
        target_h = int((h * max_dim) / w)
    target_h = max(target_h, 1)
    target_w = max(target_w, 1)
    return resize_nearest(image, target_h, target_w)


def load_image_from_upload(upload, max_dim: int = MAX_DIMENSION) -> np.ndarray:
    pil_image = Image.open(upload.stream).convert("RGB")
    image = np.array(pil_image, dtype=np.uint8)
    return limit_image_size(image, max_dim=max_dim)


def array_to_base64_png(image: np.ndarray) -> str:
    rgb = to_rgb(image)
    pil_image = Image.fromarray(rgb)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def point_add(gray: np.ndarray, constant: int) -> np.ndarray:
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            out[i, j] = _clip_byte(int(gray[i, j]) + constant)
    return out


def point_subtract(gray: np.ndarray, constant: int) -> np.ndarray:
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            out[i, j] = _clip_byte(int(gray[i, j]) - constant)
    return out


def point_multiply(gray: np.ndarray, constant: float) -> np.ndarray:
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            out[i, j] = _clip_byte(int(gray[i, j]) * constant)
    return out


def point_divide(gray: np.ndarray, constant: float) -> np.ndarray:
    if constant == 0:
        constant = 1.0
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            out[i, j] = _clip_byte(int(gray[i, j]) / constant)
    return out


def complement(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            out[i, j] = 255 - int(gray[i, j])
    return out


def solarize(gray: np.ndarray, threshold: int, invert_below: bool) -> np.ndarray:
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            pixel = int(gray[i, j])
            if invert_below:
                out[i, j] = 255 - pixel if pixel < threshold else pixel
            else:
                out[i, j] = 255 - pixel if pixel > threshold else pixel
    return out


def compute_histogram(gray: np.ndarray) -> np.ndarray:
    histogram = np.zeros(256, dtype=np.int64)
    h, w = gray.shape
    for i in range(h):
        for j in range(w):
            histogram[int(gray[i, j])] += 1
    return histogram


def histogram_image(histogram: np.ndarray, width: int = 512, height: int = 280) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            canvas[y, x, 0] = 248
            canvas[y, x, 1] = 250
            canvas[y, x, 2] = 252

    max_count = int(histogram.max())
    if max_count == 0:
        return canvas

    for x in range(width):
        if width > 256:
            level = int((x * 256) / width)
        else:
            level = x
        if level > 255:
            level = 255
        bar_height = int((int(histogram[level]) * (height - 20)) / max_count)
        y_start = height - 1
        y_end = max(height - bar_height - 1, 0)
        for y in range(y_start, y_end, -1):
            canvas[y, x, 0] = 26
            canvas[y, x, 1] = 38
            canvas[y, x, 2] = 61
    return canvas


def histogram_stretch(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    min_val = 255
    max_val = 0
    for i in range(h):
        for j in range(w):
            px = int(gray[i, j])
            if px < min_val:
                min_val = px
            if px > max_val:
                max_val = px

    if max_val == min_val:
        return gray.copy()

    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            px = int(gray[i, j])
            stretched = ((px - min_val) * 255.0) / (max_val - min_val)
            out[i, j] = _clip_byte(stretched)
    return out


def blend_add(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    h, w, c = img1.shape
    out = np.zeros((h, w, c), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            for k in range(c):
                out[i, j, k] = _clip_byte(int(img1[i, j, k]) + int(img2[i, j, k]))
    return out


def blend_weighted(img1: np.ndarray, img2: np.ndarray, alpha: float) -> np.ndarray:
    if alpha < 0:
        alpha = 0
    if alpha > 1:
        alpha = 1
    beta = 1.0 - alpha

    h, w, c = img1.shape
    out = np.zeros((h, w, c), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            for k in range(c):
                value = alpha * int(img1[i, j, k]) + beta * int(img2[i, j, k])
                out[i, j, k] = _clip_byte(value)
    return out


def subtract_images(img1: np.ndarray, img2: np.ndarray, reverse: bool = False) -> np.ndarray:
    h, w, c = img1.shape
    out = np.zeros((h, w, c), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            for k in range(c):
                a = int(img1[i, j, k])
                b = int(img2[i, j, k])
                value = b - a if reverse else a - b
                out[i, j, k] = _clip_byte(value)
    return out


def absolute_difference(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    h, w, c = img1.shape
    out = np.zeros((h, w, c), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            for k in range(c):
                diff = abs(int(img1[i, j, k]) - int(img2[i, j, k]))
                out[i, j, k] = _clip_byte(diff)
    return out


def zero_pad(gray: np.ndarray, pad: int) -> np.ndarray:
    h, w = gray.shape
    padded = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            padded[i + pad, j + pad] = int(gray[i, j])
    return padded


def convolve(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    size = kernel.shape[0]
    pad = size // 2
    padded = zero_pad(gray, pad)
    out = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            total = 0.0
            for u in range(size):
                for v in range(size):
                    total += float(padded[i + u, j + v]) * float(kernel[u, v])
            out[i, j] = _clip_byte(total)
    return out


def average_kernel(size: int) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.float64)
    value = 1.0 / float(size * size)
    for i in range(size):
        for j in range(size):
            kernel[i, j] = value
    return kernel


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if sigma <= 0:
        sigma = 1.0
    kernel = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    total = 0.0

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            exponent = -((x * x + y * y) / (2.0 * sigma * sigma))
            value = math.exp(exponent) / (2.0 * math.pi * sigma * sigma)
            kernel[i, j] = value
            total += value

    if total == 0:
        return kernel

    for i in range(size):
        for j in range(size):
            kernel[i, j] /= total
    return kernel


def mean_filter(gray: np.ndarray, size: int) -> np.ndarray:
    return convolve(gray, average_kernel(size))


def _collect_neighbors(padded: np.ndarray, i: int, j: int, size: int) -> list[int]:
    values: list[int] = []
    for u in range(size):
        for v in range(size):
            values.append(int(padded[i + u, j + v]))
    return values


def median_filter(gray: np.ndarray, size: int) -> np.ndarray:
    h, w = gray.shape
    pad = size // 2
    padded = zero_pad(gray, pad)
    out = np.zeros((h, w), dtype=np.uint8)
    middle = (size * size) // 2

    for i in range(h):
        for j in range(w):
            values = _collect_neighbors(padded, i, j, size)
            values.sort()
            out[i, j] = values[middle]
    return out


def min_filter(gray: np.ndarray, size: int) -> np.ndarray:
    h, w = gray.shape
    pad = size // 2
    padded = zero_pad(gray, pad)
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            values = _collect_neighbors(padded, i, j, size)
            out[i, j] = min(values)
    return out


def max_filter(gray: np.ndarray, size: int) -> np.ndarray:
    h, w = gray.shape
    pad = size // 2
    padded = zero_pad(gray, pad)
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            values = _collect_neighbors(padded, i, j, size)
            out[i, j] = max(values)
    return out


def add_salt_pepper_noise(gray: np.ndarray, amount: float, salt_ratio: float = 0.5) -> np.ndarray:
    if amount < 0:
        amount = 0.0
    if amount > 1:
        amount = 1.0
    if salt_ratio < 0:
        salt_ratio = 0.0
    if salt_ratio > 1:
        salt_ratio = 1.0

    h, w = gray.shape
    out = gray.copy()
    rng = np.random.default_rng()
    noisy_count = int(amount * h * w)

    for _ in range(noisy_count):
        i = int(rng.integers(0, h))
        j = int(rng.integers(0, w))
        choose_salt = rng.random() < salt_ratio
        out[i, j] = 255 if choose_salt else 0
    return out


def binary_threshold(gray: np.ndarray, threshold: int) -> np.ndarray:
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            out[i, j] = 255 if int(gray[i, j]) >= threshold else 0
    return out


def otsu_threshold(gray: np.ndarray) -> int:
    hist = compute_histogram(gray)
    total = gray.shape[0] * gray.shape[1]
    sum_total = 0.0
    for i in range(256):
        sum_total += i * int(hist[i])

    sum_background = 0.0
    weight_background = 0
    max_variance = -1.0
    threshold = 0

    for t in range(256):
        weight_background += int(hist[t])
        if weight_background == 0:
            continue

        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * int(hist[t])
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        diff = mean_background - mean_foreground
        variance = weight_background * weight_foreground * diff * diff

        if variance > max_variance:
            max_variance = variance
            threshold = t
    return threshold


def dilation(binary_img: np.ndarray, size: int) -> np.ndarray:
    h, w = binary_img.shape
    pad = size // 2
    padded = zero_pad(binary_img, pad)
    out = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            found = False
            for u in range(size):
                if found:
                    break
                for v in range(size):
                    if padded[i + u, j + v] > 0:
                        found = True
                        break
            out[i, j] = 255 if found else 0
    return out


def erosion(binary_img: np.ndarray, size: int) -> np.ndarray:
    h, w = binary_img.shape
    pad = size // 2
    padded = zero_pad(binary_img, pad)
    out = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            all_white = True
            for u in range(size):
                if not all_white:
                    break
                for v in range(size):
                    if padded[i + u, j + v] == 0:
                        all_white = False
                        break
            out[i, j] = 255 if all_white else 0
    return out


def opening(binary_img: np.ndarray, size: int) -> np.ndarray:
    return dilation(erosion(binary_img, size), size)


def closing(binary_img: np.ndarray, size: int) -> np.ndarray:
    return erosion(dilation(binary_img, size), size)


def floyd_steinberg_dither(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    work = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            work[i, j] = float(gray[i, j])

    for i in range(h):
        for j in range(w):
            old_val = work[i, j]
            new_val = 255.0 if old_val >= 128.0 else 0.0
            error = old_val - new_val
            work[i, j] = new_val

            if j + 1 < w:
                work[i, j + 1] += error * (7.0 / 16.0)
            if i + 1 < h and j > 0:
                work[i + 1, j - 1] += error * (3.0 / 16.0)
            if i + 1 < h:
                work[i + 1, j] += error * (5.0 / 16.0)
            if i + 1 < h and j + 1 < w:
                work[i + 1, j + 1] += error * (1.0 / 16.0)

    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            out[i, j] = _clip_byte(work[i, j])
    return out


def compute_metrics(reference: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    ref_gray = to_gray_weighted(reference) if reference.ndim == 3 else reference.astype(np.uint8)
    tar_gray = to_gray_weighted(target) if target.ndim == 3 else target.astype(np.uint8)

    if ref_gray.shape != tar_gray.shape:
        tar_gray = resize_nearest(tar_gray, ref_gray.shape[0], ref_gray.shape[1])

    h, w = ref_gray.shape
    total = h * w
    sum_sq_noise = 0.0
    sum_sq_signal = 0.0

    for i in range(h):
        for j in range(w):
            a = float(ref_gray[i, j])
            b = float(tar_gray[i, j])
            diff = a - b
            sum_sq_noise += diff * diff
            sum_sq_signal += a * a

    mse = sum_sq_noise / total if total else 0.0
    if mse == 0:
        psnr = float("inf")
        snr = float("inf")
    else:
        psnr = 10.0 * math.log10((255.0 * 255.0) / mse)
        signal_power = sum_sq_signal / total if total else 0.0
        snr = 10.0 * math.log10(signal_power / mse) if signal_power > 0 else float("-inf")

    return {"mse": mse, "psnr": psnr, "snr": snr}


def _prepare_second(image2: Optional[np.ndarray], h: int, w: int) -> np.ndarray:
    if image2 is None:
        raise ValueError("This operation needs a second image.")
    return resize_nearest(image2, h, w)


def process_operation(
    operation: str,
    image1: np.ndarray,
    image2: Optional[np.ndarray],
    params: Mapping[str, Any],
) -> Dict[str, Any]:
    op = operation.strip().lower()
    if not op:
        raise ValueError("Missing operation.")

    constant = _to_int(params.get("constant"), 40)
    threshold = _to_int(params.get("threshold"), 128)
    kernel_size = _sanitize_odd_size(params.get("kernel_size"), default=3, minimum=3, maximum=21)
    sigma = _to_float(params.get("sigma"), 1.3)
    alpha = _to_float(params.get("alpha"), 0.6)
    noise_amount = _to_float(params.get("noise_amount"), 0.08)
    binary_t = _to_int(params.get("binary_threshold"), 127)

    h, w = image1.shape[0], image1.shape[1]
    gray = to_gray_weighted(image1)
    response: Dict[str, Any] = {}

    if op == "gray_avg":
        response["result"] = to_gray_average(image1)

    elif op == "gray_weighted":
        response["result"] = gray

    elif op == "channel_r":
        response["result"] = extract_channel_tinted(image1, 0)

    elif op == "channel_g":
        response["result"] = extract_channel_tinted(image1, 1)

    elif op == "channel_b":
        response["result"] = extract_channel_tinted(image1, 2)

    elif op == "add_constant":
        response["result"] = point_add(gray, constant)

    elif op == "subtract_constant":
        response["result"] = point_subtract(gray, constant)

    elif op == "multiply_constant":
        response["result"] = point_multiply(gray, max(_to_float(params.get("constant"), 1.2), 0.0))

    elif op == "divide_constant":
        divisor = _to_float(params.get("constant"), 2.0)
        response["result"] = point_divide(gray, divisor)

    elif op == "complement":
        response["result"] = complement(gray)

    elif op == "solarize_below":
        response["result"] = solarize(gray, threshold=threshold, invert_below=True)

    elif op == "solarize_above":
        response["result"] = solarize(gray, threshold=threshold, invert_below=False)

    elif op == "histogram":
        hist = compute_histogram(gray)
        response["result"] = gray
        response["histogram_image"] = histogram_image(hist)

    elif op == "hist_stretch":
        stretched = histogram_stretch(gray)
        response["result"] = stretched
        response["histogram_image"] = histogram_image(compute_histogram(stretched))

    elif op == "blend_add":
        second = _prepare_second(image2, h, w)
        response["result"] = blend_add(image1, second)

    elif op == "blend_weighted":
        second = _prepare_second(image2, h, w)
        response["result"] = blend_weighted(image1, second, alpha)

    elif op == "subtract_a_b":
        second = _prepare_second(image2, h, w)
        response["result"] = subtract_images(image1, second, reverse=False)

    elif op == "subtract_b_a":
        second = _prepare_second(image2, h, w)
        response["result"] = subtract_images(image1, second, reverse=True)

    elif op == "difference_abs":
        second = _prepare_second(image2, h, w)
        response["result"] = absolute_difference(image1, second)

    elif op == "mean_filter":
        response["result"] = mean_filter(gray, kernel_size)

    elif op == "median_filter":
        response["result"] = median_filter(gray, kernel_size)

    elif op == "min_filter":
        response["result"] = min_filter(gray, kernel_size)

    elif op == "max_filter":
        response["result"] = max_filter(gray, kernel_size)

    elif op == "average_convolution":
        response["result"] = convolve(gray, average_kernel(kernel_size))

    elif op == "gaussian_convolution":
        response["result"] = convolve(gray, gaussian_kernel(kernel_size, sigma))

    elif op == "salt_pepper_noise":
        noisy = add_salt_pepper_noise(gray, amount=noise_amount)
        response["result"] = noisy

    elif op == "restore_median":
        response["result"] = median_filter(gray, kernel_size)

    elif op == "dilation":
        binary = binary_threshold(gray, binary_t)
        response["result"] = dilation(binary, kernel_size)

    elif op == "erosion":
        binary = binary_threshold(gray, binary_t)
        response["result"] = erosion(binary, kernel_size)

    elif op == "opening":
        binary = binary_threshold(gray, binary_t)
        response["result"] = opening(binary, kernel_size)

    elif op == "closing":
        binary = binary_threshold(gray, binary_t)
        response["result"] = closing(binary, kernel_size)

    elif op == "threshold":
        response["result"] = binary_threshold(gray, threshold)

    elif op == "otsu":
        best_t = otsu_threshold(gray)
        binary = binary_threshold(gray, best_t)
        response["result"] = binary
        response["threshold"] = best_t
        hist = compute_histogram(gray)
        response["histogram_image"] = histogram_image(hist)

    elif op == "floyd_steinberg":
        response["result"] = floyd_steinberg_dither(gray)

    elif op == "metrics_compare":
        second = _prepare_second(image2, h, w)
        response["result"] = absolute_difference(image1, second)
        response["metrics"] = compute_metrics(image1, second)
        response["message"] = "Metrics were computed between Image A and Image B."

    else:
        raise ValueError(f"Unsupported operation: {operation}")

    if "result" not in response:
        raise ValueError("Processing produced no output image.")

    if "metrics" not in response:
        response["metrics"] = compute_metrics(image1, to_rgb(response["result"]))

    return response

