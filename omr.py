"""
Lõi nhận dạng & chấm phiếu trả lời trắc nghiệm mẫu THPT
(Số báo danh 6 số, Mã đề 3 số, Phần I 40 câu A-D, Phần II 8 câu Đúng/Sai, Phần III 6 câu điền số).

Dùng chung cho giao diện (app.py) và dòng lệnh (cham_phieu.py).
"""
import re
import unicodedata

import cv2
import numpy as np

STD_HEIGHT = 2048          # chuẩn hoá ảnh về chiều cao này trước khi nhận dạng
FILL_DELTA = 30            # độ tối vượt nền (kênh đỏ, 0-255) tối thiểu để coi một ô là đã tô
FAINT_RATIO = 0.75         # ô tô nhạt hơn 75% ô đậm nhất cùng câu -> coi là vết tẩy / chấm nháp
LETTERS = "ABCD"
TF = "ĐS"                  # Đúng / Sai
SUB_ITEMS = "abcd"         # các ý của câu Phần II
P2_DEFAULT = [0.0, 0.1, 0.25, 0.5, 1.0]   # điểm Phần II theo số ý đúng (0..4) - quy chế THPT


class SheetError(Exception):
    """Không nhận dạng được phiếu."""


class KeyFormatError(ValueError):
    """Đáp án viết sai định dạng."""


# ----------------------------------------------------------------------------
# Khuôn mẫu phiếu: vị trí tâm ô tròn tính theo tỉ lệ (u, v) trong khung bao của từng khung.
# Đo trên phiếu mẫu; khi chấm sẽ được tinh chỉnh (affine) theo các ô phát hiện thực tế.
# ----------------------------------------------------------------------------
def _uv(us, vs):
    return np.array([(u, v) for v in vs for u in us])


P1_UV = _uv([0.261, 0.470, 0.677, 0.883],                       # A B C D
            [0.139, 0.228, 0.317, 0.405, 0.494, 0.582, 0.670, 0.760, 0.848, 0.936])
P2_UV = _uv([0.260, 0.465, 0.669, 0.873],                       # Đ S | Đ S
            [0.430, 0.578, 0.725, 0.872])                       # a b c d
_P3_U = [0.304, 0.468, 0.632, 0.797]
_P3_V = [0.3357 + 0.0668 * k for k in range(10)]                # hàng chữ số 0..9
P3_CELLS = [("-", 0), (",", 1), (",", 2)] + [(str(d), c) for d in range(10) for c in range(4)]
P3_UV = np.array([(_P3_U[0], 0.2023), (_P3_U[1], 0.2689), (_P3_U[2], 0.2689)] +
                 [(_P3_U[c], _P3_V[d]) for d in range(10) for c in range(4)])


def _place(box, uv, circles, r):
    """Đặt khuôn mẫu vào khung rồi khớp affine với các ô tròn phát hiện được,
    nên vẫn đúng vị trí kể cả khi một số ô không phát hiện được (ảnh tối, mờ)."""
    x, y, w, h = box
    T = np.c_[x + uv[:, 0] * w, y + uv[:, 1] * h]
    A = np.c_[T, np.ones(len(T))]
    pts = T.copy()
    for tol in (r * 1.6, r * 1.1, r * 0.8):
        d = np.hypot(pts[:, None, 0] - circles[None, :, 0], pts[:, None, 1] - circles[None, :, 1])
        idx, dist = d.argmin(1), d.min(1)
        m = dist < tol
        if m.sum() < 6:
            break
        M = np.linalg.lstsq(A[m], circles[idx[m], :2], rcond=None)[0]
        pts = A @ M
    return _snap(pts, circles, r * 0.6)


# ----------------------------------------------------------------------------
# Tiện ích hình học
# ----------------------------------------------------------------------------
def _normalize(image):
    s = STD_HEIGHT / image.shape[0]
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, None, fx=s, fy=s, interpolation=interp)


def _detect_bubbles(gray):
    g = cv2.medianBlur(gray, 5)
    cs = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=18,
                          param1=80, param2=18, minRadius=7, maxRadius=14)
    return np.zeros((0, 3)) if cs is None else cs[0]


def _find_boxes(gray):
    th = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5, 5), 0), 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
    cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 150 or h < 150:
            continue
        if len(cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)) != 4:
            continue
        if any(abs(x - bx) < 30 and abs(y - by) < 30 and abs(w - bw) < 30 and abs(h - bh) < 30
               for bx, by, bw, bh in boxes):
            continue  # viền trong / viền ngoài của cùng một khung
        boxes.append((x, y, w, h))
    return boxes


def _add_aligned(strong, weak):
    """Nhận thêm các khung "yếu" nếu cùng hàng và cùng kích thước với các khung chắc chắn."""
    if not strong:
        return list(strong)
    ys = np.median([b[1] for b in strong])
    ws = np.median([b[2] for b in strong])
    hs = np.median([b[3] for b in strong])
    out = list(strong)
    for b in weak:
        if (abs(b[1] - ys) < 0.15 * hs and abs(b[2] - ws) < 0.12 * ws and abs(b[3] - hs) < 0.12 * hs
                and all(abs(b[0] - o[0]) > ws / 2 for o in out)):
            out.append(b)
    return out


def _main_row(boxes):
    """Giữ các khung nằm cùng hàng với nhiều khung nhất (loại khung lạ như vùng SBD có tỉ lệ giống)."""
    if len(boxes) < 2:
        return list(boxes)

    def row(b):
        return [o for o in boxes if abs(o[1] - b[1]) < 0.15 * b[3] and abs(o[3] - b[3]) < 0.15 * b[3]]
    return row(max(boxes, key=lambda b: (len(row(b)), -b[1])))


def _complete_row(boxes, n, inside, min_circles):
    """Các khung cùng một phần nằm thẳng hàng, cách đều nhau. Nếu có khung không phát hiện
    được (do mờ / tối) thì suy ra vị trí từ các khung còn lại."""
    boxes = sorted(boxes, key=lambda b: b[0])
    if len(boxes) >= n or len(boxes) < 2:
        return boxes[:n]
    xs = np.array([b[0] for b in boxes], float)
    ys = np.array([b[1] for b in boxes], float)
    step = float(np.min(np.diff(xs)))
    if step < 50:
        return boxes
    w = int(np.median([b[2] for b in boxes]))
    h = int(np.median([b[3] for b in boxes]))
    slope, icpt = np.polyfit(xs, ys, 1)
    cands = []
    for k in range(-n, n + 1):
        x = xs[0] + k * step
        if np.min(np.abs(xs - x)) < step / 2:
            continue
        b = (int(round(x)), int(round(icpt + slope * x)), w, h)
        cnt = len(inside(b))
        if cnt >= min_circles:
            cands.append((cnt, b))
    for _, b in sorted(cands, reverse=True)[:n - len(boxes)]:
        boxes.append(b)
    return sorted(boxes, key=lambda b: b[0])


def _clusters(values, gap):
    """Gom chỉ số theo giá trị 1 chiều; tách cụm khi khoảng cách > gap."""
    order = np.argsort(values)
    groups = [[order[0]]]
    for a, b in zip(order[:-1], order[1:]):
        if values[b] - values[a] > gap:
            groups.append([b])
        else:
            groups[-1].append(b)
    return groups


def _fit_grid(circles, ncols, nrows, gap):
    """Dựng lưới nrows x ncols từ các ô tròn phát hiện được.
    Mỗi cột/hàng được khớp bằng một đường thẳng nên chịu được ảnh hơi nghiêng.
    Trả về mảng (nrows, ncols, 2) hoặc None."""
    if len(circles) < max(ncols, nrows):
        return None
    cg = sorted(_clusters(circles[:, 0], gap), key=len, reverse=True)[:ncols]
    rg = sorted(_clusters(circles[:, 1], gap), key=len, reverse=True)[:nrows]
    if len(cg) < ncols or len(rg) < nrows:
        return None
    cg.sort(key=lambda g: np.median(circles[g, 0]))
    rg.sort(key=lambda g: np.median(circles[g, 1]))
    in_rows = set(i for g in rg for i in g)
    in_cols = set(i for g in cg for i in g)

    def line(groups, members, a, b):  # khớp a = p + q*b
        res = []
        for g in groups:
            g = [i for i in g if i in members] or g
            pa, pb = circles[g, a], circles[g, b]
            if len(g) >= 3 and np.ptp(pb) > gap:
                q, p = np.polyfit(pb, pa, 1)
            else:
                q, p = 0.0, float(np.median(pa))
            res.append((p, q))
        return res

    cols = line(cg, in_rows, 0, 1)   # x = p + q*y
    rows = line(rg, in_cols, 1, 0)   # y = p + q*x
    grid = np.zeros((nrows, ncols, 2))
    for i, (ry, rq) in enumerate(rows):
        for j, (cx, cq) in enumerate(cols):
            x = cx
            for _ in range(3):
                y = ry + rq * x
                x = cx + cq * y
            grid[i, j] = (x, y)
    return grid


def _snap(pts, circles, tol):
    """Dời mỗi điểm về tâm ô tròn phát hiện được gần nhất (nếu đủ gần)."""
    out = np.array(pts, dtype=float).copy()
    flat = out.reshape(-1, 2)
    for k, (x, y) in enumerate(flat):
        d = np.hypot(circles[:, 0] - x, circles[:, 1] - y)
        i = int(np.argmin(d))
        if d[i] < tol:
            flat[k] = circles[i, :2]
    return out


def _darkness(channel, pts, r):
    """Độ tối trung bình bên trong từng ô tròn (0 = trắng, 255 = đen)."""
    rr = max(int(r * 0.65), 3)
    yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    disk = (xx ** 2 + yy ** 2) <= rr ** 2
    h, w = channel.shape
    vals = []
    for x, y in np.asarray(pts).reshape(-1, 2):
        x, y = int(round(x)), int(round(y))
        if x - rr < 0 or y - rr < 0 or x + rr >= w or y + rr >= h:
            vals.append(0.0)
            continue
        patch = channel[y - rr:y + rr + 1, x - rr:x + rr + 1]
        vals.append(255.0 - float(patch[disk].mean()))
    return np.array(vals)


def _fill_scores(vals, pts, delta):
    """Trừ đi "nền" (mặt cong bậc 2 khớp theo các ô trống) để khử ánh sáng không đều.
    Trả về độ tối vượt nền của từng ô."""
    vals = np.asarray(vals, float)
    p = np.asarray(pts, float).reshape(-1, 2)
    x = (p[:, 0] - p[:, 0].mean()) / 100.0
    y = (p[:, 1] - p[:, 1].mean()) / 100.0
    A = np.c_[np.ones_like(x), x, y, x * y, x * x, y * y]
    if len(vals) < 12:
        A = A[:, :3]
    mask = vals <= np.median(vals)          # nửa sáng hơn chắc chắn là ô trống
    coef = None
    for _ in range(6):
        if mask.sum() < A.shape[1] + 1:
            break
        coef = np.linalg.lstsq(A[mask], vals[mask], rcond=None)[0]
        new = (vals - A @ coef) < delta / 2
        if (new == mask).all():
            break
        mask = new
    if coef is None:
        return vals - np.median(vals)
    return vals - A @ coef


def _decide(scores, idx, delta):
    """Chọn ô được tô trong nhóm `idx`. Trả về (ô_tô, ô_mờ)."""
    filled = [i for i in idx if scores[i] > delta]
    faint = [i for i in idx if delta * 0.6 < scores[i] <= delta]
    if len(filled) > 1:
        top = max(scores[i] for i in filled)
        faint += [i for i in filled if scores[i] < FAINT_RATIO * top]
        filled = [i for i in filled if scores[i] >= FAINT_RATIO * top]
    return filled, sorted(faint)


# ----------------------------------------------------------------------------
# Nhận dạng phiếu
# ----------------------------------------------------------------------------
def read_sheet(image, delta=FILL_DELTA):
    """Đọc bài làm từ ảnh BGR. Tự thử xoay ảnh nếu phiếu bị chụp ngang/ngược."""
    err = None
    for boost in (False, True):   # lần 2: tăng tương phản (ảnh chụp dư sáng làm ô in hồng bị nhạt)
        for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
            img = image if rot is None else cv2.rotate(image, rot)
            try:
                res = _read(_normalize(img), delta, boost)
            except SheetError as e:
                err = err or e
                continue
            if res["upright"]:
                return res
            err = err or SheetError("Phiếu có vẻ bị ngược chiều.")
    raise err or SheetError("Không nhận dạng được phiếu.")


def _read(img, delta, boost=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if boost:  # gamma 2: kéo vùng sáng (nét in hồng nhạt) tối xuống để dò ô tròn / khung
        gray = (255.0 * (gray / 255.0) ** 2).astype(np.uint8)
    # Kênh đỏ: màu in hồng/đỏ của phiếu gần như biến mất, vết bút chì vẫn tối
    red = img[:, :, 2]
    circles = _detect_bubbles(gray)
    if len(circles) < 100:
        raise SheetError("Không tìm thấy các ô tròn trên phiếu. Hãy chụp rõ, đủ sáng và lấy trọn tờ phiếu.")
    r = float(np.median(circles[:, 2]))
    gap = r * 1.5
    tol = r * 0.9

    def inside(b):
        x, y, w, h = b
        m = ((circles[:, 0] > x) & (circles[:, 0] < x + w) &
             (circles[:, 1] > y) & (circles[:, 1] < y + h))
        return circles[m]

    # Phân loại khung theo tỉ lệ cao/rộng. "Chắc chắn" = có nhiều ô tròn phát hiện được;
    # "yếu" = đúng hình dạng nhưng ô in nhạt / ảnh tối nên Hough bắt được ít ô.
    strong = {1: [], 2: [], 3: []}
    weak = {1: [], 2: [], 3: []}
    for b in _find_boxes(gray):
        n, ratio = len(inside(b)), b[3] / b[2]
        if 1.2 < ratio < 1.8:
            part, need = 1, 30
        elif 0.7 < ratio < 1.1 and b[2] > 200:
            part, need = 2, 12
        elif 2.0 < ratio < 3.2:
            part, need = 3, 30
        else:
            continue
        (strong if n >= need else weak)[part].append(b)

    p1b = _complete_row(_add_aligned(_main_row(strong[1]), weak[1]), 4, inside, 12)
    if len(p1b) != 4:
        raise SheetError(f"Cần tìm 4 khung Phần I nhưng chỉ thấy {len(p1b)}. "
                         "Hãy chụp thẳng, rõ nét, không che khuất phiếu.")
    top1 = min(b[1] for b in p1b)
    # Phiếu đúng chiều: các khung Phần II/III nằm bên dưới Phần I
    lower = strong[2] + strong[3]
    above = sum(b[1] < top1 for b in lower)
    upright = above <= len(lower) - above
    below = {k: [b for b in strong[k] if b[1] > top1] for k in (2, 3)}
    p2b = _complete_row(_add_aligned(below[2], [b for b in weak[2] if b[1] > top1]), 4, inside, 5)
    p3b = _complete_row(_add_aligned(below[3], [b for b in weak[3] if b[1] > top1]), 6, inside, 5)
    warnings = []

    # ---- Phần I ----
    part1 = []
    for bi, b in enumerate(p1b):
        grid = _place(b, P1_UV, circles, r).reshape(10, 4, 2)
        sc = _fill_scores(_darkness(red, grid, r), grid, delta).reshape(10, 4)
        for ri in range(10):
            filled, faint = _decide(sc[ri], range(4), delta)
            part1.append({"q": bi * 10 + ri + 1, "pts": grid[ri], "scores": sc[ri],
                          "filled": filled, "faint": faint,
                          "answer": LETTERS[filled[0]] if len(filled) == 1 else None,
                          "multi": len(filled) > 1})

    # ---- Số báo danh (6 cột) + Mã đề (3 cột) ----
    sbd = made = None
    id_cols = []
    upper = circles[(circles[:, 1] < top1 - 20) & (circles[:, 0] > p1b[-1][0] - 100)]
    grid = _fit_grid(upper, 9, 10, gap)
    if grid is None:
        warnings.append("Không đọc được vùng Số báo danh / Mã đề.")
    else:
        grid = _snap(grid, circles, tol)
        sc = _fill_scores(_darkness(red, grid, r), grid, delta).reshape(10, 9)
        digits = []
        for ci in range(9):
            filled, _ = _decide(sc[:, ci], range(10), delta)
            id_cols.append({"pts": grid[:, ci], "filled": filled})
            digits.append(str(filled[0]) if len(filled) == 1 else ("*" if filled else "?"))
        sbd, made = "".join(digits[:6]), "".join(digits[6:])
        if "?" in sbd or "*" in sbd:
            warnings.append(f"Số báo danh chưa tô đủ/hợp lệ: {sbd} (? = cột bỏ trống, * = tô nhiều ô).")
        if "?" in made or "*" in made:
            warnings.append(f"Mã đề chưa tô đủ/hợp lệ: {made}.")

    # ---- Phần II: 4 khung, mỗi khung 2 câu x 4 ý x (Đúng, Sai) ----
    part2 = []
    if len(p2b) != 4:
        if p2b or p3b:
            warnings.append(f"Chỉ tìm thấy {len(p2b)}/4 khung Phần II — bỏ qua Phần II.")
    else:
        for bi, b in enumerate(p2b):
            grid = _place(b, P2_UV, circles, r).reshape(4, 4, 2)
            sc = _fill_scores(_darkness(red, grid, r), grid, delta).reshape(4, 4)
            for k in range(2):
                items = []
                for ri in range(4):
                    cols = (2 * k, 2 * k + 1)
                    filled, faint = _decide(sc[ri, list(cols)], range(2), delta)
                    items.append({"pts": grid[ri, list(cols)], "filled": filled, "faint": faint,
                                  "answer": TF[filled[0]] if len(filled) == 1 else None,
                                  "multi": len(filled) > 1})
                part2.append({"q": bi * 2 + k + 1, "items": items})

    # ---- Phần III: 6 khung, đọc các ô tròn "-", "," và chữ số 0..9 của 4 cột ----
    part3 = []
    if len(p3b) != 6:
        if p2b or p3b:
            warnings.append(f"Chỉ tìm thấy {len(p3b)}/6 khung Phần III — bỏ qua Phần III.")
    else:
        for bi, b in enumerate(p3b):
            it = read_p3(img, red, b, circles, r, delta)
            it["q"] = q = bi + 1
            for col, f in it["multi"]:
                warnings.append(f"Phần III câu {q}: cột {col + 1} tô nhiều ô "
                                f"({', '.join(_p3_cell_name(i) for i in f)}).")
            if it["value"] and not it["valid"] and not it["multi"]:
                why = " (có cột trống xen giữa)" if it["gap"] else ""
                warnings.append(f"Phần III câu {q}: bài tô '{it['value']}' không phải số hợp lệ{why}.")
            if it["faint"]:
                warnings.append(f"Phần III câu {q}: có vết tô mờ ở ô "
                                f"{', '.join(_p3_cell_name(i) for i in it['faint'])} — nên kiểm tra lại.")
            part3.append(it)

    for it in part1:
        if it["multi"]:
            warnings.append(f"Phần I câu {it['q']}: tô nhiều ô ({', '.join(LETTERS[j] for j in it['filled'])}).")
        if it["faint"]:
            chosen = f", máy chọn {it['answer']}" if it["answer"] else ""
            warnings.append(f"Phần I câu {it['q']}: có vết tô mờ ở ô {', '.join(LETTERS[j] for j in it['faint'])}"
                            f"{chosen} — nên kiểm tra lại.")
    for it in part2:
        for s, sub in zip(SUB_ITEMS, it["items"]):
            if sub["multi"]:
                warnings.append(f"Phần II câu {it['q']}{s}: tô cả Đúng và Sai.")
            if sub["faint"]:
                warnings.append(f"Phần II câu {it['q']}{s}: có vết tô mờ ở ô "
                                f"{', '.join(TF[t] for t in sub['faint'])} — nên kiểm tra lại.")

    return {"image": img, "r": r, "sbd": sbd, "made": made, "id_cols": id_cols,
            "part1": part1, "part2": part2, "part3": part3,
            "warnings": warnings, "upright": upright}


# ----------------------------------------------------------------------------
# Phần III – đọc các Ô TRÒN đã tô (giống quy tắc tô của phiếu THPT)
#   hàng 1: 1 ô "-"  -> số âm
#   hàng 2: ô ","    -> vị trí dấu phẩy (ở cột 2 hoặc cột 3)
#   hàng 3..12: chữ số 0..9 của từng cột (4 cột)
# ----------------------------------------------------------------------------
# Vị trí dải 4 ô viết tay trong khung (tỉ lệ theo khung) - chỉ để cắt ảnh đối chiếu / ghi nhãn
HW_STRIP_UV = (0.252, 0.086, 0.610, 0.080)   # u0, v0, rộng, cao
P3_MINUS = 0                                  # chỉ số ô "-" trong P3_CELLS
P3_COLS = [[i for i, (s, c) in enumerate(P3_CELLS) if c == col and s != "-"] for col in range(4)]


def _p3_cell_name(i):
    s, c = P3_CELLS[i]
    return "dấu trừ" if s == "-" else (f"dấu phẩy cột {c + 1}" if s == "," else f"số {s} cột {c + 1}")


def _locate_strip(img, box):
    """Tìm hình chữ nhật bao 4 ô vuông viết đáp án ở đầu khung Phần III."""
    x, y, w, h = box
    y0, y1 = y + int(0.02 * h), y + int(0.22 * h)
    reg = img[y0:y1, x:x + w]
    th = cv2.adaptiveThreshold(reg[:, :, 1], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 8)
    cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        if 0.45 * w < bw < 0.8 * w and 2.8 < bw / max(bh, 1) < 5.5:
            if best is None or bw * bh > best[2] * best[3]:
                best = (bx, by, bw, bh)
    if best is None:
        u0, v0, uw, vh = HW_STRIP_UV
        return (int(x + u0 * w), int(y + v0 * h), int(uw * w), int(vh * h))
    return (x + best[0], y0 + best[1], best[2], best[3])


def read_p3(img, red, box, circles, r, delta):
    """Đọc bài làm 1 câu Phần III từ các ô tròn đã tô.
    Mỗi cột (4 cột) là 1 ký tự: chữ số 0..9, hoặc dấu phẩy (cột 2, 3). Ô "-" đánh dấu số âm."""
    pts = _place(box, P3_UV, circles, r)
    sc = _fill_scores(_darkness(red, pts, r), pts, delta)
    minus, faint = _decide(sc, [P3_MINUS], delta)
    filled = list(minus)
    chars, multi = [], []
    for col, idx in enumerate(P3_COLS):
        f, fa = _decide(sc, idx, delta)
        faint += fa
        filled += f
        if len(f) == 1:
            chars.append(P3_CELLS[f[0]][0])
        elif f:
            chars.append("*")
            multi.append((col, f))
        else:
            chars.append("")
    used = [i for i, ch in enumerate(chars) if ch]
    gap = bool(used) and (used[-1] - used[0] + 1) != len(used)     # cột trống xen giữa
    value = ("-" if minus else "") + "".join(chars)
    valid = bool(re.fullmatch(r"-?\d*,?\d+", value)) and not gap and not multi
    sx, sy, sw, sh = _locate_strip(img, box)
    H, W = img.shape[:2]
    mx = int(0.15 * sh)
    crop = img[max(sy - int(0.25 * sh), 0):min(sy + sh + int(0.2 * sh), H),
               max(sx - mx, 0):min(sx + sw + mx, W)].copy()
    return {"value": value, "valid": valid, "conf": 0.5 if (faint or multi or (value and not valid)) else 1.0,
            "pts": pts, "scores": sc, "filled": sorted(filled), "faint": sorted(set(faint)),
            "multi": multi, "gap": gap, "box": box, "strip": (sx, sy, sw, sh), "crop": crop}


# ----------------------------------------------------------------------------
# Đáp án
# ----------------------------------------------------------------------------
KEY_TEMPLATE = """MÃ ĐỀ: 104

PHẦN I
1D 2B 3A 4C 5B 6A 7C 8B 9D 10B
11C 12B 13D 14C 15A 16A 17B 18D 19D 20A

PHẦN II
1: ĐSĐS
2: SSĐĐ

PHẦN III
1: 12,5
2: -3
"""

_HEADER = re.compile(r"^\s*PH[ẦÂA]N\s*(III|II|I|3|2|1)\b\s*[:.\-]?", re.M)


def _ascii_upper(text):
    """Viết hoa và chuẩn hoá Unicode (giữ nguyên chữ Đ)."""
    return unicodedata.normalize("NFC", text).upper()


def parse_key(text):
    """Đọc bộ đáp án dạng văn bản. Trả về dict {made, p1, p2, p3}."""
    text = _ascii_upper(text or "")
    key = {"made": None, "p1": {}, "p2": {}, "p3": {}}
    m = re.search(r"M[ÃA]\s*Đ[ỀE]\s*(?:THI)?\s*[:=]?\s*(\d{1,4})", text)
    if m:
        key["made"] = m.group(1)
        text = text[:m.start()] + text[m.end():]

    heads = list(_HEADER.finditer(text))
    if not heads:
        sections = {"1": text}            # không ghi tên phần -> coi là Phần I
    else:
        sections = {}
        for i, h in enumerate(heads):
            part = {"I": "1", "II": "2", "III": "3"}.get(h.group(1), h.group(1))
            end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
            sections[part] = sections.get(part, "") + "\n" + text[h.end():end]

    # Phần I
    s = sections.get("1", "")
    if s.strip():
        num = re.findall(r"(\d+)\s*[.:)\-=]?\s*([ABCD])(?![A-ZĐ])", s)
        if num:
            for q, a in num:
                key["p1"][int(q)] = a
        else:
            for i, a in enumerate(re.findall(r"[ABCD]", re.sub(r"CÂU|CAU", "", s))):
                key["p1"][i + 1] = a
    # Phần II
    s = sections.get("2", "").replace("T", "Đ").replace("F", "S")
    if s.strip():
        tf = r"([ĐDS])\s*[,;]?\s*([ĐDS])\s*[,;]?\s*([ĐDS])\s*[,;]?\s*([ĐDS])"
        num = re.findall(r"(\d+)\s*[.:)\-=]?\s*" + tf, s)
        groups = [(int(g[0]), g[1:]) for g in num] if num else \
            [(i + 1, g) for i, g in enumerate(re.findall(tf, re.sub(r"CÂU|CAU", "", s)))]
        for q, g in groups:
            key["p2"][q] = "".join("Đ" if c in "ĐD" else "S" for c in g)
    # Phần III
    s = sections.get("3", "")
    if s.strip():
        val = r"(-?\d*[.,]?\d+)"
        num = re.findall(r"(\d+)\s*(?::|\)|=|\.(?=\s))\s*" + val, s)
        if num:
            for q, v in num:
                key["p3"][int(q)] = v.replace(".", ",")
        else:
            for i, v in enumerate(re.findall(val, re.sub(r"CÂU|CAU", "", s))):
                key["p3"][i + 1] = v.replace(".", ",")

    for part, n in (("p1", 40), ("p2", 8), ("p3", 6)):
        bad = [q for q in key[part] if not 1 <= q <= n]
        if bad:
            name = {"p1": "I", "p2": "II", "p3": "III"}[part]
            raise KeyFormatError(f"Phần {name} chỉ có câu 1–{n}, nhưng đáp án có câu {bad[0]}.")
    if not (key["p1"] or key["p2"] or key["p3"]):
        raise KeyFormatError("Chưa đọc được câu nào từ đáp án. Xem phần hướng dẫn định dạng.")
    for part in ("p1", "p2", "p3"):
        key[part] = dict(sorted(key[part].items()))
    return key


# ----------------------------------------------------------------------------
# Chấm điểm
# ----------------------------------------------------------------------------
def _num(s):
    try:
        return float(s.replace(",", "."))
    except (ValueError, AttributeError):
        return None


def is_valid_number(v):
    return bool(re.fullmatch(r"-?\d*,?\d+", v or ""))


def p3_answers(res, override=None):
    """Bài làm Phần III cuối cùng: giá trị máy đọc, hoặc giá trị giáo viên đã sửa (override)."""
    override = override or {}
    out = {}
    for it in res["part3"]:
        if it["q"] in override:
            v = (override[it["q"]] or "").strip().replace(".", ",").replace(" ", "")
            out[it["q"]] = {"value": v, "valid": is_valid_number(v), "edited": True}
        else:
            out[it["q"]] = {"value": it["value"], "valid": it["valid"], "edited": False}
    return out


def grade(res, key, p1=0.25, p2=P2_DEFAULT, p3=0.25, p3_override=None):
    """Chấm bài làm `res` (từ read_sheet) theo `key` (từ parse_key).
    p3_override: {câu: giá trị} - bài làm Phần III đã được giáo viên sửa lại."""
    out = {"p1": [], "p2": [], "p3": [], "warnings": []}
    km, rm = key.get("made"), res.get("made")
    if km and rm and (km.lstrip("0") or "0") != (rm.lstrip("0") or "0"):
        out["warnings"].append(f"Mã đề trên phiếu ({res['made']}) khác mã đề của đáp án ({key['made']}).")

    by_q = {it["q"]: it for it in res["part1"]}
    for q, a in key["p1"].items():
        it = by_q[q]
        ok = it["answer"] == a
        shown = it["answer"] or ("/".join(LETTERS[j] for j in it["filled"]) + " (nhiều)" if it["multi"] else "")
        out["p1"].append({"q": q, "student": shown, "key": a, "ok": ok, "points": p1 if ok else 0.0})

    by_q = {it["q"]: it for it in res["part2"]}
    for q, k in key["p2"].items():
        it = by_q.get(q)
        if it is None:
            out["p2"].append({"q": q, "student": "", "key": k, "n_ok": 0, "ok_items": [False] * 4,
                              "points": 0.0})
            continue
        stu = [sub["answer"] for sub in it["items"]]
        ok_items = [s == c for s, c in zip(stu, k)]
        n = sum(ok_items)
        out["p2"].append({"q": q, "student": "".join(s or ("*" if sub["multi"] else "_")
                                                   for s, sub in zip(stu, it["items"])),
                          "key": k, "n_ok": n, "ok_items": ok_items, "points": float(p2[n])})

    final = p3_answers(res, p3_override)
    for q, k in key["p3"].items():
        a = final.get(q, {"value": "", "valid": False, "edited": False})
        v = a["value"]
        ok = bool(a["valid"] and _num(v) is not None and _num(k) is not None
                  and abs(_num(v) - _num(k)) < 1e-9)
        out["p3"].append({"q": q, "student": v, "key": k, "ok": ok, "edited": a["edited"],
                          "points": p3 if ok else 0.0})

    out["score"] = sum(x["points"] for part in ("p1", "p2", "p3") for x in out[part])
    out["max"] = len(key["p1"]) * p1 + len(key["p2"]) * float(p2[4]) + len(key["p3"]) * p3
    out["summary"] = {
        "p1": (sum(x["ok"] for x in out["p1"]), len(out["p1"]), sum(x["points"] for x in out["p1"])),
        "p2": (sum(x["n_ok"] for x in out["p2"]), 4 * len(out["p2"]), sum(x["points"] for x in out["p2"])),
        "p3": (sum(x["ok"] for x in out["p3"]), len(out["p3"]), sum(x["points"] for x in out["p3"])),
    }
    return out


# ----------------------------------------------------------------------------
# Vẽ kết quả
# ----------------------------------------------------------------------------
GREEN, RED, BLUE, ORANGE = (40, 170, 40), (40, 40, 220), (220, 90, 30), (0, 140, 255)


def _ring(img, pt, r, color, t=3):
    cv2.circle(img, (int(round(pt[0])), int(round(pt[1]))), int(r) + 5, color, t, cv2.LINE_AA)


def annotate(res, graded=None):  # noqa: C901
    """Vẽ kết quả lên ảnh. Xanh lá = đáp án đúng, đỏ = ô thí sinh tô sai,
    cam = câu/ý bỏ trống (khoanh ở đáp án đúng), xanh dương = ô tô của câu không chấm."""
    img = res["image"].copy()
    r = res["r"]
    for col in res["id_cols"]:
        for d in col["filled"]:
            _ring(img, col["pts"][d], r, BLUE)

    g1 = {x["q"]: x for x in graded["p1"]} if graded else {}
    for it in res["part1"]:
        g = g1.get(it["q"])
        if g is None:
            for j in it["filled"]:
                _ring(img, it["pts"][j], r, BLUE)
            continue
        _ring(img, it["pts"][LETTERS.index(g["key"])], r, GREEN, 4)
        for j in it["filled"]:
            if LETTERS[j] != g["key"] or it["multi"]:
                _ring(img, it["pts"][j], r, RED, 4)
        if not it["filled"]:
            _ring(img, it["pts"][LETTERS.index(g["key"])], r, ORANGE, 4)

    g2 = {x["q"]: x for x in graded["p2"]} if graded else {}
    for it in res["part2"]:
        g = g2.get(it["q"])
        for si, sub in enumerate(it["items"]):
            if g is None:
                for t in sub["filled"]:
                    _ring(img, sub["pts"][t], r, BLUE)
                continue
            k = TF.index(g["key"][si])
            _ring(img, sub["pts"][k], r, GREEN if sub["filled"] else ORANGE, 4)
            for t in sub["filled"]:
                if t != k or sub["multi"]:
                    _ring(img, sub["pts"][t], r, RED, 4)

    g3 = {x["q"]: x for x in graded["p3"]} if graded else {}
    for it in res["part3"]:
        g = g3.get(it["q"])
        value = g["student"] if g else it["value"]
        color = BLUE if g is None else (GREEN if g["ok"] else (RED if value else ORANGE))
        for i in it["filled"]:                       # các ô tròn thí sinh đã tô
            _ring(img, it["pts"][i], r, color, 4 if g is not None else 3)
        x, y, w, h = it["box"]
        if g is not None:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 4, cv2.LINE_AA)
        # ghi giá trị đọc được (chỉ ký tự ASCII: số, "-", ",") ở góc trên bên phải khung
        label = value if value else "(trong)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        org = (x + w - tw - 8, y + int(0.02 * h) + th + 4)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return img
