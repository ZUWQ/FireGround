"""Shared environment startup with strict RPC errors and pre-refresh evidence."""
from __future__ import annotations

import asyncio
import math
from typing import Any, Callable

import tongsim as ts


async def initialize_environment(
    context: Any, *, refresh_actors: bool = True, start_fire: bool = False,
    burn_time: float = 5.0, require_initial_fire: bool = False, record: dict | None = None,
    checkpoint: Callable[[dict], None] | None = None,
) -> dict:
    """Refresh registered actors once at startup, then optionally initialize fire.

    ``clears_destroyed_id_history`` describes the requested refresh's side effect;
    ``actors_refreshed`` confirms that its RPC returned successfully. The callback
    synchronously persists registry evidence before the destructive refresh call.
    Without a callback, that evidence is retained only in the returned record.
    ``require_initial_fire`` additionally verifies a positive burning count after
    the preburn wait; an accepted ignition RPC alone does not establish a fire.
    """
    state = record if record is not None else {}
    state.update(
        actors_refresh_requested=bool(refresh_actors), actors_refreshed=False,
        start_fire_requested=bool(start_fire), fire_initialized=False,
        initial_fire_check_requested=bool(start_fire and require_initial_fire),
        initial_fire_verified=False, initial_fire_burning_num=None,
        clears_destroyed_id_history=bool(refresh_actors),
        destroyed_id_history_note=("Requested refresh clears UE DestroyedIds; "
                                   "actors_refreshed records confirmed RPC completion."),
        destroyed_ids_before_refresh=[], registry_before_count=None,
        registry_after_count=None, new_registered_count=None,
        stage="validate", status="running", error=None,
    )
    state.pop("checkpoint_error", None)

    def save(stage: str) -> None:
        state["stage"] = stage
        if checkpoint is not None:
            checkpoint(state)

    async def strict_rpc(name: str):
        rpc = getattr(ts.UnaryAPI, name)
        # SDK wrappers otherwise turn a failed RPC into [] or None. Refresh
        # legitimately returns None on success, so truthiness cannot validate it.
        return await getattr(rpc, "__wrapped__", rpc)(context.conn)

    def registry(value: Any) -> list[dict]:
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            raise RuntimeError("Actor registry query must return a list of objects")
        return value

    try:
        save("validate")
        try:
            duration = float(burn_time)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("burn_time must be finite and nonnegative") from exc
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("burn_time must be finite and nonnegative")
        state["burn_time_sec"] = duration

        if refresh_actors:
            save("query_registry_before_refresh")
            before = registry(await strict_rpc("query_info"))
            state["registry_before_count"] = len(before)
            state["destroyed_ids_before_refresh"] = list(dict.fromkeys(
                str(row["id"]) for row in before
                if row.get("destroyed") is True and row.get("id")
            ))
            before_ids = {str(row["id"]) for row in before if row.get("id")}
            # This checkpoint must finish before RefreshActorMappings can erase
            # the UE registration subsystem's DestroyedIds history.
            save("before_refresh_backup")
            print("[Environment] Refreshing actor registry; this clears UE DestroyedIds history.", flush=True)
            save("refresh_actors")
            await strict_rpc("refresh_actors_map")
            state["actors_refreshed"] = True
            save("query_registry_after_refresh")
            after = registry(await strict_rpc("query_info"))
            state["registry_after_count"] = len(after)
            if not after:
                raise RuntimeError("Actor registry is empty after refresh")
            after_ids = {str(row["id"]) for row in after if row.get("id")}
            state["new_registered_count"] = len(after_ids - before_ids)
            # Only explicitly selected global episodes own this hook. Reuse the
            # successful registry read before ignition; local/vision do no work.
            prime = getattr(getattr(context, "global_oracle_provider", None), "prime_registry", None)
            if callable(prime):
                prime(after)
            save("registry_refreshed")
            print(f"[Environment] Actor registry: {len(before)} -> {len(after)}; "
                  f"new registered IDs={state['new_registered_count']}", flush=True)

        if start_fire:
            save("start_fire")
            if await strict_rpc("start_to_burn") is not True:
                raise RuntimeError("UE did not confirm fire initialization")
            state["fire_initialized"] = True
            save("wait_after_fire")
            await asyncio.sleep(duration)
            if require_initial_fire:
                save("verify_initial_fire")
                fire = await strict_rpc("get_burned_area")
                count = fire.get("burning_num") if isinstance(fire, dict) else None
                if (isinstance(count, bool) or not isinstance(count, (int, float))
                        or not math.isfinite(count) or count < 0):
                    raise RuntimeError("Initial fire verification requires a finite nonnegative burning_num")
                state["initial_fire_burning_num"] = count
                if count == 0:
                    raise RuntimeError("Initial fire verification failed: UE reports no burning objects after ignition")
                state["initial_fire_verified"] = True
                save("initial_fire_verified")

        state["status"] = "ready"
        save("ready")
        return state
    except (Exception, asyncio.CancelledError) as exc:
        state["status"] = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        if checkpoint is not None:
            try:
                checkpoint(state)
            except Exception as checkpoint_error:
                # Preserve the original startup exception if its error report
                # also cannot be saved; the mutable caller record remains valid.
                state["checkpoint_error"] = f"{type(checkpoint_error).__name__}: {checkpoint_error}"
        raise
