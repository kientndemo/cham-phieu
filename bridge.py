"""Cầu nối giữa giao diện điện thoại (JavaScript) và bộ chấm omr.py, chạy trong Pyodide."""
import json
from collections import OrderedDict

import cv2
import numpy as np

import omr

_store = OrderedDict()      # token -> {"res", "vis"}: giữ vài phiếu gần nhất để sửa/chấm lại nhanh
_KEEP = 3


def _jsonable(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (tuple, set)):
        return list(o)
    return str(o)


def _dump(obj):
    return json.dumps(obj, ensure_ascii=False, default=_jsonable)


def _rgba(img):
    """Ảnh BGR -> (rộng, cao, bytes RGBA) để JavaScript vẽ lên canvas."""
    return img.shape[1], img.shape[0], cv2.cvtColor(img, cv2.COLOR_BGR2RGBA).tobytes()


def has(token):
    return token in _store


def process(token, buf, w, h, delta):
    """Đọc phiếu từ ảnh RGBA (w x h). Trả JSON bài làm; ảnh cắt Phần III lấy qua crop()."""
    img = np.frombuffer(buf.to_bytes(), np.uint8).reshape(h, w, 4)[:, :, 2::-1].copy()
    try:
        res = omr.read_sheet(img, delta=float(delta))
    except omr.SheetError as e:
        return _dump({"error": str(e)})
    _store[token] = {"res": res, "vis": None}
    _store.move_to_end(token)
    while len(_store) > _KEEP:
        _store.popitem(last=False)
    return _dump({
        "sbd": res["sbd"], "made": res["made"], "warnings": res["warnings"],
        "part1": [{"q": it["q"], "answer": it["answer"], "multi": it["multi"]} for it in res["part1"]],
        "part2": [{"q": it["q"], "items": [{"answer": s["answer"], "multi": s["multi"]} for s in it["items"]]}
                  for it in res["part2"]],
        "part3": [{"q": it["q"], "value": it["value"], "valid": it["valid"], "note": _p3_note(it)}
                  for it in res["part3"]],
    })


def _p3_note(it):
    if it["multi"]:
        return "tô nhiều ô"
    if it["value"] and not it["valid"]:
        return "không hợp lệ"
    if it["faint"]:
        return "tô mờ"
    return ""


def crop(token, i):
    """Ảnh vùng chữ viết tay của câu Phần III thứ i (để giáo viên đối chiếu)."""
    return _rgba(_store[token]["res"]["part3"][i]["crop"])


def _key(key_json):
    k = json.loads(key_json)
    return {"made": k.get("made") or None,
            "p1": {int(q): a for q, a in (k.get("p1") or {}).items() if a},
            "p2": {int(q): a for q, a in (k.get("p2") or {}).items() if a and len(a) == 4},
            "p3": {int(q): str(a).strip().replace(".", ",") for q, a in (k.get("p3") or {}).items()
                   if str(a).strip()}}


def grade(token, key_json, settings_json, override_json):
    """Chấm phiếu `token`. key_json rỗng -> chỉ vẽ bài làm. Ảnh kết quả lấy qua annotated()."""
    slot = _store[token]
    res = slot["res"]
    st = json.loads(settings_json or "{}")
    override = {int(q): v for q, v in json.loads(override_json or "{}").items()}
    graded = None
    if key_json:
        key = _key(key_json)
        if key["p1"] or key["p2"] or key["p3"]:
            graded = omr.grade(res, key, p1=float(st.get("p1", 0.25)),
                               p2=[float(x) for x in st.get("p2", omr.P2_DEFAULT)],
                               p3=float(st.get("p3", 0.25)), p3_override=override)
    slot["vis"] = omr.annotate(res, graded)
    final3 = omr.p3_answers(res, override)
    return _dump({"graded": graded, "p3final": {str(q): v for q, v in final3.items()}})


def annotated(token, max_h=1800):
    vis = _store[token]["vis"]
    if vis.shape[0] > max_h:
        s = max_h / vis.shape[0]
        vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return _rgba(vis)


def parse_key(text):
    """Đọc đáp án dạng chữ (cùng định dạng với bản máy tính)."""
    try:
        k = omr.parse_key(text)
    except omr.KeyFormatError as e:
        return _dump({"error": str(e)})
    return _dump({"made": k["made"], "p1": k["p1"], "p2": k["p2"], "p3": k["p3"]})
