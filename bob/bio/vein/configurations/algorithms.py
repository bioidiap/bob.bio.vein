#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from ..algorithms.MiuraMatch import MiuraMatch

huangwl_tool = MiuraMatch(ch=18, cw=28)
huangwl_gpu_tool = MiuraMatch(ch=18, cw=28, gpu=True)
miuramax_tool = MiuraMatch(ch=80, cw=90)
miuramax_gpu_tool = MiuraMatch(ch=80, cw=90, gpu=True)
miurarlt_tool = MiuraMatch(ch=65, cw=55)
miurarlt_gpu_tool = MiuraMatch(ch=65, cw=55, gpu=True)
