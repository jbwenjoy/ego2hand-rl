"""Pure JSON stage/header boundaries, not checkpoint deserialization/health.

Actor-stage tests isolate the orchestration gate from the separate serialized
health validator. Dummy pinned bytes are never interpreted as model weights.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supermimic_research.artifacts import ArtifactRegistry, file_pin
from supermimic_research.learning import online_contract as C


class OnlineContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.path = Path(self.temp.name)
        manifest = self.path / "artifacts.toml"; manifest.write_text('schema=1\n[roots]\nfixture="."\n', encoding="utf8")
        self.registry = ArtifactRegistry(manifest)
        self.root_cp = self.bytes("root_model.bytes", b"not a deserialized checkpoint")
        self.root_health = self.json("root_health.json", {"fixture": "root"})
        self.root_protocol = self.json("root_protocol.json", {"fixture": "root protocol"})
        self.root = dict(checkpoint=self.root_cp, root_health=self.root_health, root_protocol=self.root_protocol, protocol=dict(recipe={"fixed":1}, inputs={"data":"root"}, sources={"trainer":"root"}))
        self.config = dict(training=dict(authorized=False, output="first", resume_artifact="", resume_health="", seed=1734,microbatch=8), online=dict(native_authorized=False, collection_cycles=3))

    def tearDown(self):
        self.temp.cleanup()

    def bytes(self, name, value):
        path = self.path / name; path.write_bytes(value); return file_pin(path)

    def json(self, name, value):
        return self.bytes(name, json.dumps(value).encode())

    def test_cursor_cycle_boundaries_and_types(self):
        for update,cycle,step in [(10,1,10),(100,1,100),(101,2,1),(200,2,100),(201,3,1),(300,3,100)]:
            self.assertEqual(C.cursor(update), dict(cycle=cycle,update_in_cycle=step,global_paired_updates=update,total_lineage_paired_updates=1000+update,shared_WM_lineage_updates=1100+update))
        for value in [0,301,True,10.,"10"]:
            with self.subTest(value=value), self.assertRaises(ValueError):C.cursor(value)

    def test_science_config_only_removes_operational_changes(self):
        changed=copy.deepcopy(self.config)
        changed["training"].update(authorized=True, output="resume", resume_artifact="own", resume_health="health")
        changed["online"]["native_authorized"]=True
        self.assertEqual(C.science_config(self.config), C.science_config(changed))
        changed["training"]["seed"]=1735
        self.assertNotEqual(C.science_config(self.config), C.science_config(changed))

    def test_root_payload_is_exact_final1000_and_not_resume(self):
        ck=dict(schema=C.ROOT_CHECKPOINT, paired_round=1000, shared_WM_updates=1100, protocol=self.root_protocol, **copy.deepcopy(self.root["protocol"]), arms={m:{} for m in C.R.MODES}, WM_loss_scales={"fixed":True})
        C.validate_root_payload(ck,self.root)
        for field,value in [("schema",C.CHECKPOINT),("paired_round",100),("shared_WM_updates",200),("protocol",self.root_health),("sources",{"trainer":"different"})]:
            bad=copy.deepcopy(ck);bad[field]=value
            with self.subTest(field=field), self.assertRaises(ValueError):C.validate_root_payload(bad,self.root)
        del ck["WM_loss_scales"]
        with self.assertRaises(ValueError):C.validate_root_payload(ck,self.root)

    def test_cycle1_actor_must_be_exact_root(self):
        actor=dict(kind="offline1000",stage_id=0,checkpoint=self.root_cp,health=self.root_health,training_protocol=self.root_protocol)
        self.assertIsNone(C.actor_metadata(actor,1,self.config,self.registry,{},self.root))
        for key,value in [("stage_id",False),("kind","own_online"),("checkpoint",self.root_health)]:
            bad=copy.deepcopy(actor);bad[key]=value
            with self.subTest(field=key), self.assertRaises(ValueError):C.actor_metadata(bad,1,self.config,self.registry,{},self.root)

    def test_first10_cannot_collect_cycle2_even_after_positive_header(self):
        cp=self.bytes("online10.bytes",b"dummy10")
        hp=self.json("online10_health.json",dict(update=10,protocol=self.root_protocol))
        actor=dict(kind="own_online",stage_id=1,checkpoint=cp,health=hp,training_protocol=self.root_protocol)
        with patch.object(C,"validate_own_header",return_value=None), self.assertRaisesRegex(ValueError,"checkpoint10"):
            C.actor_metadata(actor,2,self.config,self.registry,{},self.root)

    def test_full100_actor_requires_acyclic_parent_to_root(self):
        cp=self.bytes("online100.bytes",b"dummy100")
        header=dict(update=100,protocol=self.root_protocol,parent_checkpoint=self.root_cp,parent_health=self.root_health)
        hp=self.json("online100_health.json",header)
        actor=dict(kind="own_online",stage_id=1,checkpoint=cp,health=hp,training_protocol=self.root_protocol)
        with patch.object(C,"validate_own_header",return_value=None):
            self.assertEqual(C.actor_metadata(actor,2,self.config,self.registry,{},self.root),header)
        bad=self.json("cyclic_health.json",header|dict(parent_checkpoint=cp,parent_health=hp));actor["health"]=bad
        with patch.object(C,"validate_own_header",return_value=None), self.assertRaisesRegex(ValueError,"cyclic"):
            C.actor_metadata(actor,2,self.config,self.registry,{},self.root)

    def own_header(self):
        cp=self.bytes("own10.bytes",b"dummy own header fixture")
        source=self.bytes("module.py",b"# fixture source, never executed\n")
        sources={"learning/module.py":source}; inputs={"task":self.root_protocol,"root_checkpoint":self.root_cp}
        h=dict(schema=C.HEALTH, checkpoint_schema=C.CHECKPOINT, checks={"saved":True}, checkpoint=cp, update=10, cursor=C.cursor(10), counts=dict(WM_updates=10,paired_updates=10), learning_calls=C.expected_learning_calls(10,8), root_checkpoint=self.root_cp, normalizer_origin_checkpoint=self.root_cp, configuration=copy.deepcopy(self.config), provenance=dict(sources=sources,inputs=inputs), protocol=self.root_protocol, parent_checkpoint=self.root_cp,parent_health=self.root_health,collections=[],pool_identity_map=self.root_protocol)
        return cp,sources,inputs,h

    def test_actual_header_checks_selected_inputs_source_and_typed_cursor(self):
        cp,sources,inputs,h=self.own_header()
        with patch.object(C,"selected_inputs",return_value=inputs):
            C.validate_own_header(h,cp,self.config,sources,self.root,self.registry)
            mutations=[("source",lambda v:v["provenance"]["sources"]["learning/module.py"].update(sha256="0"*64)), ("input",lambda v:v["provenance"]["inputs"]["task"].update(sha256="0"*64)), ("cursor_bool",lambda v:v["cursor"].update(cycle=True)), ("counter_float",lambda v:v["counts"].update(WM_updates=10.)), ("learning_calls",lambda v:v["learning_calls"].update(Actor_FW=41)), ("false_check",lambda v:v["checks"].update(saved=False)), ("wrong_root",lambda v:v.update(normalizer_origin_checkpoint=cp)), ("recipe",lambda v:v["configuration"]["training"].update(seed=1735))]
            for name,mutate in mutations:
                bad=copy.deepcopy(h); mutate(bad)
                with self.subTest(mutation=name), self.assertRaises(ValueError):
                    C.validate_own_header(bad,cp,self.config,sources,self.root,self.registry)

    def test_actor_payload_joins_validated_health_cursor_counts_and_parents(self):
        cp,sources,_,h=self.own_header()
        actor=dict(kind="own_online",stage_id=0,checkpoint=cp,health=self.root_health,training_protocol=self.root_protocol)
        ck=copy.deepcopy(h);ck["schema"]=C.CHECKPOINT
        C.validate_actor_payload(ck,actor,h,self.root,self.config,sources)
        for key,value in [("cursor",C.cursor(100)),("counts",dict(WM_updates=100,paired_updates=100)),("learning_calls",C.expected_learning_calls(100,8)),("parent_checkpoint",cp),("parent_health",self.root_protocol),("pool_identity_map",cp)]:
            bad=copy.deepcopy(ck);bad[key]=value
            with self.subTest(field=key), self.assertRaises(ValueError):C.validate_actor_payload(bad,actor,h,self.root,self.config,sources)

    def collection(self, name):
        source=self.bytes(name+"_worker.py",b"# unexecuted worker fixture\n")
        sources={"learning/collection_worker.py":source}
        actor=dict(kind="offline1000",stage_id=0,checkpoint=self.root_cp,health=self.root_health,training_protocol=self.root_protocol)
        pool=self.bytes(name+"_pool.bytes",b"not an NPZ-open fixture")
        dataset=self.json(name+"_dataset.json",dict(schema="phone_tripod_online_common_replay_dataset:1",data_training_ready=True,collection_cycle=1,common_pool=pool))
        protocol=self.json(name+"_protocol.json",dict(schema=C.COLLECTION,collection_cycle=1,actor_input=actor,sources=sources))
        report=self.json(name+"_report.json",dict(collection_cycle=1,protocol=protocol,dataset_report=dataset,actor_input=actor,episodes=16,rows=1,counts=dict(integration_steps=128,observer_forward=0)))
        terminal=self.json(name+"_terminal.json",dict(returncode=0,protocol=protocol,source=source,sources=sources,command=["python",source["path"],"--protocol",protocol["path"],"--run"]))
        worker=self.json(name+"_worker_receipt.json",dict(protocol=protocol,report=report,own_CPU_checkpoint_reads=1,NN_constructors=4,unused_Critic_constructors=2,WM_FW=0,Critic_FW=0,optimizer_constructors=0,backward=0,learning_updates=0,CUDA_initialized=False))
        return dict(cycle=1,protocol=protocol,terminal=terminal,report=report,dataset_report=dataset,worker_receipt=worker,actor_input=actor)

    def test_collection_protocol_cycle_and_worker_bindings(self):
        receipt=self.collection("good"); C.verify_collections([receipt],self.registry)
        for name,field,value in [("worker","source",self.root_cp),("command","command",["python","--protocol","different.json","--run"])]:
            bad=copy.deepcopy(receipt);terminal=C.read(self.registry.resolve_pin(bad["terminal"]));terminal[field]=value
            bad["terminal"]=self.json(name+"_bad_terminal.json",terminal)
            with self.subTest(field=field),self.assertRaises(ValueError):C.verify_collections([bad],self.registry)
        bad=copy.deepcopy(receipt);report=C.read(self.registry.resolve_pin(bad["report"]));report["collection_cycle"]=2
        bad["report"]=self.json("bad_cycle_report.json",report)
        with self.assertRaisesRegex(ValueError,"cursor join"):C.verify_collections([bad],self.registry)

    def test_worker_receipt_rejects_backend_learning_and_wrong_join(self):
        receipt=self.collection("worker")
        for field,value in [("CUDA_initialized",True),("WM_FW",1),("learning_updates",1),("optimizer_constructors",1),("report",self.root_protocol)]:
            bad=copy.deepcopy(receipt);worker=C.read(self.registry.resolve_pin(receipt["worker_receipt"]));worker[field]=value
            bad["worker_receipt"]=self.json("bad_"+field+".json",worker)
            with self.subTest(field=field),self.assertRaises(ValueError):C.verify_collections([bad],self.registry)


if __name__ == "__main__":
    unittest.main()
