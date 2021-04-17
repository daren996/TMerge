# common issues

* [RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1] because the unspecified dimension size -1 can be any value and is ambiguous] when using mmtracking related operators.

Solution: make sure the mmdetection version is v2.9.0. 
Reference: https://github.com/open-mmlab/mmtracking/issues/131#issuecomment-821039886