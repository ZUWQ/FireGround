Action records in previous_execution_feedback and memory_state.short_term use the same action object as the response: name, subtype when required, and parameters. The subtype stays outside parameters.
When previous_decision_feedback reports status=invalid and action_executed=false, the previous output was rejected before execution. Its errors identify output-contract violations; its memory update and communication were not committed. Use the current input to choose a new decision that follows the response contract.
Read status together with reason and data. status summarizes either execution or a post-action check: completed, failed, interrupted or unknown. reason identifies the outcome being reported; data contains its evidence.
Key reasons:
- target_reached: the final XY position was within the navigation tolerance; height is not part of arrival checking.
- fire_extinguished: the specified target was not burning in the verification observation.
- target_destroyed: the named target has been destroyed and no longer exists; the action was not executed. Do not retry that target.
- follow_confirmed: new followers were reported compared with the pre-command state.
- following_reported: followers were reported, with no comparable pre-command state.
- following_unchanged: followers remain, with no additions.
- stop_following_succeeded: the stop instruction returned true; cached followers are cleared.
- stop_following_failed: the stop instruction returned false; cached followers are retained.
- wait_elapsed: the fixed waiting interval elapsed.
- perimeter_route_completed: the exploration route was completed and closed without local substitutes.
- adapted_route_completed: the exploration route was completed and closed using in-room projected destinations or local substitutes.
- reachable_route_completed: a partial exploration finished after unreachable waypoints were skipped. Check data.partial_exploration, visited_waypoint_count, skipped_waypoints and skipped_waypoint_reasons. This does not establish a completed perimeter or a fully inspected/clear room.
When present, data.effect_success is true, false or null for that specific check. For follow, true means the current follower list is nonempty; use newly_following_npcs to identify additions. Conclude that a civilian was rescued only from evidence satisfying the task's rescue condition. Route completion establishes traversal, not exhaustive visual inspection.
