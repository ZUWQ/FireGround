"""Synthetic boundary/clock/coverage regressions, no UE or model calls."""
import json
from pathlib import Path
import tempfile
import unittest

from EmbodiedMAS.evaluation.benchmark_metrics import compute_metrics


class BenchmarkMetricsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def episode(self, name, *, final_value=80, final_health=70, fire_counts=((0, 2), (299, 2), (300.05, 1)),
                initial_value=95, initial_health=90, scene=1, visibility=True):
        root = self.root / name
        evaluation = root / 'evaluation'
        evaluation.mkdir(parents=True)
        (root / 'scene_config.json').write_text(json.dumps({'scene': scene}))
        (evaluation / 'summary.json').write_text(json.dumps({'started_at_monotonic': 1000,
            'deadline_monotonic': 1300, 'initial_object_ids': ['obj-1', 'obj-2'], 'first_zero_fire_elapsed_sec': None}))
        (evaluation / 'metadata.json').write_text(json.dumps({'sample_interval_sec': 1}))
        def sample(phase, elapsed, prop, health, count):
            absolute = 1000 + elapsed
            return {'phase': phase, 'elapsed_sec': elapsed, 'after_deadline_sec': max(0, elapsed-300),
                'queries': {'property': {'valid': True, 'data': prop},
                    'npc_health': {'valid': True, 'data': {'person': health}},
                    'fire': {'valid': True, 'data': {'burning_num': count},
                             'requested_at_monotonic': absolute-.01, 'completed_at_monotonic': absolute}},
                'npcs': [{'id': 'person', 'health': health}],
                'object_catalog': {'valid': True, 'objects': [{'id': 'obj-1'}, {'id': 'obj-2'}]}}
        rows = [sample('pre_ignition', -10, 100, 100, 0), sample('initial', 0.1, initial_value, initial_health, 2)]
        rows += [sample('periodic' if t<=300 else 'final', t, final_value, final_health, n) for t,n in fire_counts]
        (evaluation / 'world_states.jsonl').write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
        if visibility:
            observations = [
                {'type':'visibility_sample','valid':True,'completed_at_monotonic':1001,'visible_object_ids':['obj-1','npc-1']},
                {'type':'visibility_sample','valid':True,'completed_at_monotonic':1002,'visible_object_ids':['obj-1']},
                {'type':'visibility_sample','valid':False,'completed_at_monotonic':1003,'visible_object_ids':['obj-2']},
                {'type':'visibility_sample','valid':True,'completed_at_monotonic':1300.1,'visible_object_ids':['obj-2']}]
            (evaluation / 'observations.jsonl').write_text('\n'.join(json.dumps(row) for row in observations)+'\n')
        return root

    def test_raw_ratios_use_baseline_and_offer_both_initial_phases(self):
        agent=self.episode('agent');baseline=self.episode('base',final_value=50,final_health=40)
        result=compute_metrics(agent,baseline)
        self.assertAlmostEqual(result['svr'],.6)
        self.assertAlmostEqual(result['hpr'],.5)
        self.assertAlmostEqual(result['svr_initial'],30/45)
        self.assertAlmostEqual(result['hpr_initial'],30/50)
        self.assertEqual(result['initial_phase'],'pre_ignition')
        official=compute_metrics(agent,baseline,initial_phase='initial')
        self.assertAlmostEqual(official['svr'],30/45)
        self.assertAlmostEqual(official['hpr'],30/50)

    def test_negative_ratios_are_preserved_and_zero_baseline_loss_is_not_zero_score(self):
        agent=self.episode('a',final_value=40,final_health=20)
        baseline=self.episode('b',final_value=50,final_health=40)
        result=compute_metrics(agent,baseline)
        self.assertAlmostEqual(result['svr'],-.2)
        self.assertAlmostEqual(result['hpr'],-1/3)
        unchanged=self.episode('no-loss',final_value=100,final_health=100)
        result=compute_metrics(agent,unchanged)
        self.assertIsNone(result['svr']);self.assertIsNone(result['hpr'])
        self.assertEqual(result['svr_reason'],'nonpositive_baseline_loss')

    def test_missing_reference_and_mismatched_scene_are_explicit(self):
        agent=self.episode('a')
        result=compute_metrics(agent)
        self.assertIsNone(result['svr']);self.assertEqual(result['hpr_reason'],'baseline_not_supplied')
        result=compute_metrics(agent,self.episode('different',scene=2))
        self.assertIsNone(result['svr']);self.assertEqual(result['svr_reason'],'baseline_scene_configuration_mismatch')

    def test_ct_first_zero_uses_query_clock_and_can_precede_reignition(self):
        agent=self.episode('a',fire_counts=((30,0),(60,2),(300.05,2)))
        result=compute_metrics(agent)
        self.assertEqual(result['ct'],30)
        self.assertEqual(result['ct_status'],'observed_zero')
        self.assertEqual(result['clock'],'monotonic_wall_seconds');self.assertFalse(result['ct_exact'])

    def test_after_deadline_zero_is_not_completed_and_exposes_boundary_uncertainty(self):
        agent=self.episode('a',fire_counts=((299,1),(300.05,0)))
        result=compute_metrics(agent)
        self.assertEqual(result['ct'],400)
        self.assertEqual(result['ct_status'],'boundary_uncertain')
        self.assertIsNone(result['ct_first_zero_sec'])
        self.assertEqual(result['ct_fire_before']['count'],1)
        self.assertEqual(result['ct_fire_after']['count'],0)
        self.assertAlmostEqual(result['ct_fire_after']['gap_to_horizon_sec'],.05)

    def test_terminal_penalty_with_stable_bracketing_counts(self):
        result=compute_metrics(self.episode('a',fire_counts=((299,3),(300.05,3))))
        self.assertEqual(result['ct'],600)
        self.assertEqual(result['ct_status'],'terminal_penalty_estimate')
        self.assertFalse(result['ct_boundary_uncertain'])

    def test_ec_is_deduplicated_budget_limited_and_catalog_scoped(self):
        result=compute_metrics(self.episode('a'))
        self.assertEqual(result['ec'],.5)
        self.assertEqual(result['ec_observed_count'],1)
        self.assertEqual(result['ec_ids_outside_catalog'],1)
        self.assertEqual(result['ec_invalid_samples_in_budget'],1)
        self.assertEqual(result['ec_samples_outside_budget'],1)
        self.assertIsNone(result['pv'])

    def test_missing_visibility_and_evidence_are_not_perfect_or_zero_scores(self):
        result=compute_metrics(self.episode('a',visibility=False))
        self.assertIsNone(result['ec'])
        result=compute_metrics(self.root/'missing')
        for field in ('svr','hpr','ct','ec','pv'):self.assertIsNone(result[field])

    def test_missing_npc_health_is_not_summed_as_zero(self):
        a=self.episode('a');b=self.episode('b',final_value=50,final_health=40)
        p=a/'evaluation/world_states.jsonl';records=[json.loads(l) for l in p.read_text().splitlines()]
        records[-1]['queries']['npc_health']={'valid':False,'data':None}
        p.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
        result=compute_metrics(a,b)
        self.assertIsNone(result['hpr'])
        self.assertEqual(result['hpr_reason'],'missing_or_invalid_boundary_data')

    def test_unfinished_episode_does_not_extrapolate_an_early_fire_count_to_horizon(self):
        result=compute_metrics(self.episode('early',fire_counts=((20,2),)))
        self.assertIsNone(result['ct'])
        self.assertEqual(result['ct_status'],'unavailable')
        self.assertEqual(result['ct_reason'],'no_valid_fire_observation_bracketing_horizon')

    def test_initial_roster_health_missing_is_not_silently_dropped(self):
        a=self.episode('partial');b=self.episode('base',final_value=50,final_health=40)
        p=a/'evaluation/world_states.jsonl';records=[json.loads(l) for l in p.read_text().splitlines()]
        for record in records:
            record['npcs'].append({'id':'missing-person','health_valid':False})
        p.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
        result=compute_metrics(a,b)
        self.assertIsNone(result['hpr'])
        self.assertEqual(result['hpr_reason'],'missing_or_invalid_boundary_data')

    def test_outcome_horizon_mismatch_does_not_compare_different_terminal_times(self):
        a=self.episode('a');b=self.episode('base',final_value=50,final_health=40)
        result=compute_metrics(a,b,horizon_sec=200)
        self.assertIsNone(result['svr']);self.assertIsNone(result['hpr'])
        self.assertEqual(result['svr_reason'],'agent_terminal_horizon_mismatch')
        self.assertEqual(result['recorded_horizon_sec'],300)

    def test_invalid_parameters_are_rejected(self):
        for kwargs in ({'horizon_sec':0},{'horizon_sec':True},{'lambda_ct':-1},{'lambda_ct':float('nan')},{'initial_phase':'unknown'}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):compute_metrics(self.root,**kwargs)


if __name__=='__main__':unittest.main()
