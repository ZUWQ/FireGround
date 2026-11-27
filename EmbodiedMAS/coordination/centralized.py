"""Event-driven CMAS scheduling, isolated from individual agent runtimes."""
from __future__ import annotations
import asyncio
import copy
import json
import math
import time
from pathlib import Path
from typing import Any
from EmbodiedMAS.agent.central_output import validate_payload
from EmbodiedMAS.agent.decision_output import InvalidDecisionOutput
from EmbodiedMAS.agent.perception.visual_distance import planar_distance_m
from EmbodiedMAS.agent.spatial_units import world_xy_m
from EmbodiedMAS.runners.common import validate_max_steps

class CentralizedRuntime:
    def __init__(self, workers, controller, *, time_limit_sec=300, output_dir: Path,
                 run_id: str, idle_poll_sec=2.0):
        if not workers or len({w.robot_id for w in workers}) != len(workers):
            raise ValueError('CMAS requires a nonempty team with unique robot IDs')
        for name, value in (('time_limit_sec', time_limit_sec), ('idle_poll_sec', idle_poll_sec)):
            if value is None and name == 'time_limit_sec':
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        self.workers, self.controller = list(workers), controller
        self.output_dir, self.run_id = Path(output_dir), run_id
        self.time_limit_sec, self.idle_poll_sec = time_limit_sec, idle_poll_sec
        self.end_reason, self.step = None, 0
        self._started = self._deadline = None
        self._events, self._pending = [], []
        self._wake, self._fatal_wake = asyncio.Event(), asyncio.Event()
        self._fatal, self._urgent_revision = None, 0
        self._sos_revisions = {worker.robot_id: 0 for worker in workers}
        self._seen_sos = set()
        self._dispatches = {}
        self._running = self._used = False
        self._run_task = None
        self.monitor = None
        self._vision_pause_controller = getattr(getattr(controller, "context", None), "_emas_vision_pause_controller", None)
        if self._vision_pause_controller is not None and any(getattr(w, "modality", None) != "vision" for w in workers):
            raise ValueError("A paused-vision CMAS cannot mix Oracle workers")
        self._previous_decision_feedback = None
        for worker in self.workers:
            worker.on_event = self.on_event

    @property
    def task_time(self):
        if self._vision_pause_controller is not None:
            state = self._vision_pause_controller.snapshot()
            return {key: state[key] for key in ('elapsed_sec', 'limit_sec', 'remaining_sec')} | {'end_reason': self.end_reason}
        now = time.monotonic()
        return {'elapsed_sec': 0.0 if self._started is None else max(0.0, now - self._started),
                'limit_sec': self.time_limit_sec,
                'remaining_sec': None if self._deadline is None else max(0.0, self._deadline - now),
                'end_reason': self.end_reason}

    async def _wait_vision_running(self):
        if self._vision_pause_controller is not None:
            await self._vision_pause_controller.wait_running()

    def _budget_expired(self):
        remaining = self.task_time['remaining_sec']
        return remaining is not None and remaining <= 0

    def _record(self, event, **fields):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'run_id': self.run_id, 'step': self.step, 'event': event,
                'monotonic_sec': time.monotonic(), **fields}, ensure_ascii=False, allow_nan=False) + '\n')

    def on_event(self, event):
        event = copy.deepcopy(event)
        if event.get('type') == 'sos':
            rid = event.get('robot_id')
            worker = next((w for w in self.workers if w.robot_id == rid), None)
            def semantic(value):
                if isinstance(value, dict):
                    return {k: semantic(v) for k, v in value.items()
                            if k not in {'captured_at', 'received_at'}}
                if isinstance(value, list):
                    return [semantic(v) for v in value]
                return value
            # A real worker's private geometry fingerprint preserves distinct
            # signals without exposing bearing/range to the central model.
            signal = event.pop('_signal_key', None) or json.dumps(
                semantic(event.get('context', {})), sort_keys=True)
            key = (rid, event.get('action_id', getattr(worker, 'action_id', 0)), signal)
            if key in self._seen_sos:
                self._record('sos_coalesced', robot_id=rid, action_id=key[1])
                return
            self._seen_sos.add(key)
            self._sos_revisions[rid] = self._sos_revisions.get(rid, 0) + 1
        event['event_id'] = len(self._events) + 1
        event.setdefault('captured_at', time.monotonic())
        self._events.append(event)
        kind = event.get('type')
        if kind == 'action_result' and isinstance(event.get('decision_step'), int):
            self.controller.remember_action_result(event['decision_step'], event['robot_id'], event['feedback'])
        if kind in {'action_finished', 'sos'}:
            self._pending.append(event)
        if kind == 'sos':
            self._urgent_revision += 1
        if kind == 'worker_error':
            self._fatal = RuntimeError('CMAS robot worker failed: ' + str(event.get('robot_id', 'unknown')))
            self._fatal_wake.set()
        self._record('robot_event', detail=event)
        self._wake.set()

    def _check(self):
        if self._fatal:
            raise self._fatal
        for task in self._dispatches.values():
            if task.done():
                task.result()

    async def _snapshot(self):
        await self._wait_vision_running()
        snapshots = await asyncio.gather(*(w.snapshot() for w in self.workers))
        self._snapshot_image_observations = {worker.robot_id: snapshot.get("image_observation")
                                             for worker, snapshot in zip(self.workers, snapshots)}
        self._check()
        watermark = self._events[-1]['event_id'] if self._events else 0
        events = copy.deepcopy([e for e in self._pending if e['event_id'] <= watermark])
        targets, entities, candidates = {}, [], []
        for index, (worker, snapshot) in enumerate(zip(self.workers, snapshots)):
            coordinates, observed = snapshot.get("target_bindings", (None, None))
            if coordinates is None:
                coordinates, observed = worker.target_snapshot()
            for ref, xyz in coordinates.items():
                targets.setdefault(ref, copy.deepcopy(xyz))
            candidates.extend((float(e.get('captured_at_monotonic', 0)), index, e, coordinates) for e in observed)
        for _, _, entity, coordinates in sorted(candidates, key=lambda item: item[:2]):
            entities.append(copy.deepcopy(entity))
            for key in ('target_ref', 'name'):
                if entity.get(key) in coordinates:
                    targets[entity[key]] = copy.deepcopy(coordinates[entity[key]])
        global_snapshots = [(index, snapshot) for index, snapshot in enumerate(snapshots)
                            if snapshot.get("observation", {}).get("observation_scope") == "global"]
        latest_global_index = None
        if global_snapshots:
            # Each global snapshot covers the entire task world. Using an older
            # worker's bindings could resurrect targets destroyed in a newer one.
            index, latest = max(global_snapshots, key=lambda pair: max(
                (frame.get("captured_at", 0) for frame in pair[1]["observation"].get("frames", [])), default=0))
            latest_global_index = index
            coordinates, observed = latest.get("target_bindings", (None, None))
            if coordinates is None:
                coordinates, observed = self.workers[index].target_snapshot()
            targets, entities = copy.deepcopy(coordinates), copy.deepcopy(list(observed))
        states, observations, images, known = {}, {}, [], {}
        for index, (worker, snapshot) in enumerate(zip(self.workers, snapshots)):
            states[worker.robot_id] = snapshot['state']
            # Global observations and bindings must describe the same full state;
            # older workers' copies must not reintroduce stale or destroyed facts.
            if (snapshot['observation'].get('observation_scope') != 'global'
                    or index == latest_global_index):
                observations[worker.robot_id] = copy.deepcopy(snapshot['observation'])
                images.extend(snapshot.get('image_paths', []))
            known[worker.robot_id] = snapshot.get('known_objects', {})
        # Only SAM+Pixel projections opt in to distance fields. Preserve the
        # source frame's capture-time distance, then estimate each commanded
        # robot's distance using its separately timestamped public state (m).
        # Target coordinates remain private and must belong to this snapshot.
        for observation in observations.values():
            for frame in observation.get("frames", []):
                for entity in frame.get("entities", []):
                    if "distance_xy_m" not in entity:
                        continue
                    target_uu = (targets.get(entity.get("target_ref"))
                                 if entity.get("grounding_available") is True else None)
                    target_m = world_xy_m(target_uu)
                    entity["distances_by_robot_m"] = {
                        robot_id: planar_distance_m(target_m, state.get("location"))
                        for robot_id, state in states.items()
                    }
        memory = copy.deepcopy(self.controller.memory_state)
        if len(global_snapshots) != len(snapshots):
            memory['known_objects'] = known
        else:
            memory.pop('known_objects', None)
        payload = {'step': self.step, 'agent_id': 'central_agent', 'snapshot_time': time.monotonic(),
            'robot_states': states, 'current_observation': {'robots': observations},
            'signal_channel': {'environment_signals': [e for e in events if e.get('type') == 'sos']},
            'completed_events': [e for e in events if e.get('type') == 'action_finished'],
            'memory_state': memory, 'previous_execution_feedback': {
                rid: state.get('last_execution_feedback') for rid, state in states.items()},
            'previous_decision_feedback': copy.deepcopy(self._previous_decision_feedback), 'task_time': self.task_time}
        return copy.deepcopy(payload), images, targets, tuple(entities), watermark

    async def _decide(self, payload, images):
        model = asyncio.create_task(self.controller.decide(payload, image_paths=images), name='cmas-central-decision')
        failed = asyncio.create_task(self._fatal_wake.wait(), name='cmas-worker-failure')
        try:
            done, _ = await asyncio.wait({model, failed}, return_when=asyncio.FIRST_COMPLETED)
            if failed in done:
                self._check()
            return await model
        finally:
            for task in (model, failed):
                if not task.done():
                    task.cancel()
            await asyncio.gather(model, failed, return_exceptions=True)

    async def _replace(self, worker, action, previous, targets, entities, step, image_observation=None,
                       sos_revision=None):
        def sos_changed():
            return sos_revision is not None and self._sos_revisions.get(worker.robot_id, 0) != sos_revision
        def skip():
            self.controller.remember_action_result(step, worker.robot_id,
                {'status': 'interrupted', 'reason': 'sos_before_dispatch',
                 'data': {'execution_started': False}})
            self._record('command_skipped', robot_id=worker.robot_id, reason='sos_before_dispatch', decision_step=step)
        try:
            await self._wait_vision_running()
            if sos_changed():
                skip()
                return
            if previous is not None:
                if not previous.done():
                    previous.cancel()
                await asyncio.gather(previous, return_exceptions=True)
            if sos_changed():
                skip()
                return
            # A new boundary may have replaced refs while the center thought.
            if getattr(worker, 'modality', None) == 'vision':
                current_refs = {ref for source in self.workers for ref in source.target_snapshot()[0]}
                targets = {ref: xyz for ref, xyz in targets.items() if ref in current_refs}
                entities = tuple(e for e in entities if e.get('target_ref') in current_refs)
                latest_image = getattr(worker, '_last_image_observation', None)
                if latest_image is not None:
                    image_observation = latest_image
            image_options = {"image_observation": image_observation} if image_observation is not None else {}
            if getattr(worker, 'modality', None) == 'vision':
                image_options['current_target_refs'] = lambda: {
                    ref for source in self.workers for ref in source.target_snapshot()[0]}
            if getattr(worker, 'supports_dispatch_guard', False):
                image_options['dispatch_allowed'] = lambda: not sos_changed()
            await self._wait_vision_running()
            applied = await worker.start_action(action, decision_step=step, targets=targets, entities=entities, **image_options)
            if applied is False:
                skip()
                return
            self._record('command_applied', robot_id=worker.robot_id, decision_step=step,
                         action=action, action_id=worker.action_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.controller.remember_action_result(step, worker.robot_id,
                {'status': 'failed', 'reason': 'dispatch_failed', 'data': {'execution_started': False}})
            self._fatal = RuntimeError('CMAS command dispatch failed: ' + type(exc).__name__)
            self._fatal_wake.set()
            raise
        finally:
            self._wake.set()

    def _dispatch_done(self, task, robot_id, step):
        # A task cancelled before its coroutine starts never reaches _replace's
        # exception handlers. It still belongs to an accepted decision.
        if task.cancelled():
            self.controller.remember_action_result(step, robot_id,
                {'status': 'interrupted', 'reason': 'dispatch_cancelled', 'data': {'execution_started': False}})

    async def _wait(self):
        # Dispatch/transport notifications alone are not new decision evidence.
        while True:
            self._wake.clear()
            self._check()
            if self._pending:
                return
            if all(w.status == 'idle' for w in self.workers) and all(t.done() for t in self._dispatches.values()):
                try:
                    await asyncio.wait_for(self._wake.wait(), self.idle_poll_sec)
                except TimeoutError:
                    self._record('idle_poll')
                    return
            else:
                await self._wake.wait()

    async def _stop(self):
        for task in self._dispatches.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._dispatches.values(), return_exceptions=True)
        # gather may return immediately for already-done tasks. Let their queued
        # cancellation callbacks publish feedback before shutdown is complete.
        await asyncio.sleep(0)
        stops, tasks = [], []
        for worker in self.workers:
            if callable(getattr(worker, 'stop', None)):
                stops.append(worker.stop())
            elif getattr(worker, 'task', None) is not None:
                worker.task.cancel()
                tasks.append(worker.task)
        outcomes = await asyncio.gather(*stops, *tasks, return_exceptions=True)
        errors = [e for e in outcomes if isinstance(e, Exception)]
        if errors:
            raise ExceptionGroup('CMAS worker shutdown failed', errors)

    async def run(self, *, max_decisions=None):
        validate_max_steps(max_decisions)
        if self._running or self._used:
            raise RuntimeError('Use a fresh CMAS runtime for each episode')
        if self.time_limit_sec is None and max_decisions is None:
            raise ValueError('CMAS requires a time limit or decision cap')
        self._used = self._running = True
        self._run_task = asyncio.current_task()
        if max_decisions == 0:
            if self.monitor is not None:
                await self.monitor.capture_snapshot('initial')
            self.end_reason, self._running = 'decision_limit', False
            self._run_task = None
            return
        self._started = time.monotonic()
        self._deadline = None if self.time_limit_sec is None else self._started + self.time_limit_sec
        pause = self._vision_pause_controller
        if pause is not None:
            pause.start(self._started, self.time_limit_sec)
        deadline = pause.timeout_active(self.time_limit_sec) if pause is not None else asyncio.timeout(self.time_limit_sec)
        try:
            async with deadline as timer:
                if self.monitor is not None:
                    await self.monitor.start(started_at=self._started, deadline=self._deadline)
                initializers = [asyncio.create_task(w.initialize()) for w in self.workers]
                try:
                    await asyncio.gather(*initializers)
                finally:
                    for task in initializers:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*initializers, return_exceptions=True)
                force = True
                while max_decisions is None or self.step < max_decisions:
                    if not force:
                        await self._wait()
                    self._check()
                    force = False
                    self.step += 1
                    payload, images, targets, entities, watermark = await self._snapshot()
                    sos_revisions = dict(self._sos_revisions)
                    self._record('decision_input', runtime_input=payload, image_paths=images)
                    try:
                        output = await self._decide(payload, images)
                    except InvalidDecisionOutput as exc:
                        self._previous_decision_feedback = {'status': 'invalid', 'step': self.step,
                            'errors': copy.deepcopy(exc.errors), 'action_executed': False}
                        self._record('decision_rejected', feedback=self._previous_decision_feedback)
                        force = True
                        continue
                    self._check()
                    await self._wait_vision_running()
                    if self._budget_expired():
                        self.end_reason = 'time_limit'
                        break
                    errors = validate_payload(output, [w.robot_id for w in self.workers],
                        note_max_chars=payload['memory_state']['note_max_chars'])
                    if not errors:
                        for worker in self.workers:
                            action = output['robot_actions'][worker.robot_id]
                            if action['name'] != 'continue':
                                try:
                                    worker.body.validate_action(action)
                                except (ValueError, NotImplementedError):
                                    errors.append({'code': 'action_validation_failed', 'path': f'$.robot_actions.{worker.robot_id}',
                                        'message': "Action does not satisfy this robot's available action contract."})
                    if errors:
                        self._previous_decision_feedback = {'status': 'invalid', 'step': self.step,
                            'errors': errors, 'action_executed': False}
                        self._record('decision_rejected', feedback=self._previous_decision_feedback)
                        force = True
                        continue
                    changed = [w.robot_id for w in self.workers
                               if self._sos_revisions.get(w.robot_id, 0) != sos_revisions.get(w.robot_id, 0)]
                    reconciliation = None
                    if changed:
                        output = copy.deepcopy(output)
                        skipped = {rid: output['robot_actions'][rid] for rid in changed}
                        for rid in changed:
                            output['robot_actions'][rid] = {'name': 'continue', 'parameters': {}}
                        output['memory_update'] = {'note': None}
                        reconciliation = {'status': 'reconciled', 'step': self.step,
                            'reason': 'sos_during_decision', 'skipped_robot_commands': skipped,
                            'message': 'New SOS affected these robots; their old commands were not dispatched. Replan from the pending signals.'}
                        self._record('decision_reconciled', feedback=reconciliation)
                        force = True
                    self.controller.accept_decision(output)
                    self.controller.remember_decision(self.step, output)
                    self._previous_decision_feedback = reconciliation
                    commanded = set()
                    for worker in self.workers:
                        action = output['robot_actions'][worker.robot_id]
                        if action['name'] == 'continue':
                            self._record('continue', robot_id=worker.robot_id, action_id=worker.action_id, execution_status=worker.status)
                        else:
                            commanded.add(worker.robot_id)
                            previous = self._dispatches.get(worker.robot_id)
                            self._dispatches[worker.robot_id] = asyncio.create_task(self._replace(worker,
                                copy.deepcopy(action), previous, copy.deepcopy(targets), copy.deepcopy(entities), self.step,
                                copy.deepcopy(self._snapshot_image_observations.get(worker.robot_id)),
                                sos_revision=sos_revisions.get(worker.robot_id, 0)),
                                name='cmas-dispatch-' + worker.robot_id)
                            self._dispatches[worker.robot_id].add_done_callback(
                                lambda task, rid=worker.robot_id, step=self.step: self._dispatch_done(task, rid, step))
                    self._pending = [e for e in self._pending if e['event_id'] > watermark and not (
                        e.get('type') == 'action_finished' and e.get('robot_id') in commanded)]
                    self._record('decision_accepted', robot_actions=output['robot_actions'])
                    await asyncio.sleep(0)
                if self.end_reason is None:
                    self.end_reason = 'decision_limit'
        except TimeoutError as exc:
            from EmbodiedMAS.agent.runtime.vision_pause import VisionPauseBudgetExhausted
            if not isinstance(exc, VisionPauseBudgetExhausted) and not timer.expired():
                self.end_reason = 'error'
                raise
            self.end_reason = 'time_limit'
        except asyncio.CancelledError:
            self.end_reason = 'cancelled'
            raise
        except Exception:
            self.end_reason = 'error'
            raise
        finally:
            try:
                try:
                    if pause is not None:
                        await pause.close()
                finally:
                    await self._stop()
            finally:
                self._running = False
                self._run_task = None
                self._record('episode_ended', task_time=self.task_time)

    async def close(self):
        from EmbodiedMAS.agent.runtime.vision_protocol import close_and_cleanup
        context = getattr(self.controller, 'context', None)
        await close_and_cleanup(context, self._close_resources, release=False)

    async def _close_resources(self):
        task = self._run_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._stop()
