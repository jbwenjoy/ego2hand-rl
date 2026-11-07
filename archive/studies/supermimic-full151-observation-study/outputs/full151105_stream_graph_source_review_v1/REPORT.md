# Phase0 stream/graph foreign-source review

Read-only source/saved-JSON review, noframework/native/GPU execution. Exactly25 ExternalStream wrappers,3 Graph and1 Device were tied toactualnative_entry ownerfields, notaccepted bytype alone.

The25 wrappers are TorchArray._torch_stream non-owning handles; sourcecontexts restoreselection andsnapshot/readback synchronize. Allowance is conditional on unchanged live stream/device lifetime andfixed backingpointers. Actual CUDAhandle/currentstream values were notsaved.

Graphs are **mutable**, notreadonly: firstlaunch lazily initializesgraph_exec. Graphcaptures mayalso owntransient GPUworkspaces outsideEnv traversal. Solver/SAP have observed initialization paths, butthis boundedreview is notacomplete used-region write-before-read proof. Wholehiddenstate dependence remains **UNKNOWN**, andINTEGRATION bitexact0 doesnotresolveit. Rootmaydecide aphase0 engineering baseline exploration; no fullhiddenstate/full151/FDqualification followsfromthis review.

TheWarpDevice similarly holdsmutable context/stream/allocator caches; fixed sourcepaths useScopedDevice andnomidbranch stream/device/graph reconfiguration. Itis aconditional resource identity, nota serializedphysicalstate.

Currentexactpaths areA0/0 only. native_entry indiceswilldrift asphase histories/newlazycache objects appear. Afterphase0 qualification, recommend aseparatev2 all8-inventory discovery mode (no restoredbranches) andper-snapshot exactreview/guard, orstrictstructuralpaths withowner/source/arrayrole verification. Neither isimplemented orauthorizedhere. See report.json forexact29 reasons/pins/remainingunknowns. CPUmodel/data/no_grad/spec reviews belongtootherowners.
