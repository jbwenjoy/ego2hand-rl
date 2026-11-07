此尝试的接触分类与初握通过判定无效，不应解释为物理失败。

原因：当时的helper读取MuJoCo 3.11旧字段contact.geom1/geom2，而Warp get_data_into只填写当前contact.geom[2]，导致漏计物体接触。修正后重新运行的有效结果位于同级 rotating_reference_warp_hold_geomfix/，短hold通过。

保留本目录用于记录发现与修复过程，未修改原始summary/samples。
