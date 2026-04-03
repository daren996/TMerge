"""KITTI label parsing helpers for training datasets."""

from __future__ import annotations


def label_data_record(record: str) -> dict[str, float | int | str]:
    record_list = record.split(" ")
    return {
        "frame": int(record_list.pop(0)),
        "tid": int(record_list.pop(0)),
        "type": record_list.pop(0),
        "truncated": float(record_list.pop(0)),
        "occluded": int(record_list.pop(0)),
        "alpha": float(record_list.pop(0)),
        "left": float(record_list.pop(0)),
        "top": float(record_list.pop(0)),
        "right": float(record_list.pop(0)),
        "bottom": float(record_list.pop(0)),
        "height": float(record_list.pop(0)),
        "width": float(record_list.pop(0)),
        "length": float(record_list.pop(0)),
        "rotation_y": float(record_list.pop(0)),
        "x": float(record_list.pop(0)),
        "y": float(record_list.pop(0)),
        "z": float(record_list.pop(0)),
    }


def get_fid2record(
    vid2labelfile: dict[str, str],
    vid: str,
    obj_specify: list[str],
) -> dict[int, list[dict[str, float | int | str]]]:
    fid2record: dict[int, list[dict[str, float | int | str]]] = {}
    with open(vid2labelfile[vid], encoding="utf-8") as in_file:
        for line in in_file:
            single_record = label_data_record(line)
            fid = int(single_record["frame"])
            fid2record.setdefault(fid, [])
            if single_record["type"] in obj_specify:
                fid2record[fid].append(single_record)
    return fid2record

