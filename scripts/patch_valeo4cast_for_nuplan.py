#!/usr/bin/env python3
"""Apply minimal source fixes to Valeo4Cast's UniTraj for pretrained MTR inference.

Only targets the pinned revision reviewed in October 2026. Stops without modifying
anything if an expected source fragment does not match. --apply creates .bak backups.
"""
from __future__ import annotations
import argparse
import ast
import shutil
from pathlib import Path

# {relative file: [(name, old, new), ...]}
PATCHES = {
    'UniTraj/unitraj/datasets/base_dataset.py': [
        (
            "nuPlan fallback timestamps",
            'timestamps += state["timestamps"].tolist()',
            'timestamps += state.get("timestamps", np.arange(len(state["position"]), dtype=np.int64)).tolist()',
        ),
        (
            "nuPlan unknown object class label",
            "track_infos['object_class_label'].append(v['object_label'])",
            "track_infos['object_class_label'].append(v.get('object_label', 0))",
        ),
        (
            "skip unfinished dynamic-signal parsing for nuPlan only",
            'dynamic_map_infos = self.preprocess_dynamic_map(scenario) #, total_steps)',
            "dynamic_map_infos = ({'lane_id': [], 'state': [], 'stop_point': []} "
            "if scenario['metadata'].get('dataset') == 'nuplan' "
            "else self.preprocess_dynamic_map(scenario))",
        ),
        (
            "register newly created prediction-cache entries",
            '            print(f"scenario processed with {nb_forecasts_to_make} forecasts to make: {file_name}")',
            '            file_list.update(scenario_summary)\n'
            '            print(f"scenario processed with {nb_forecasts_to_make} forecasts to make: {file_name}")',
        ),
        (
            "skip invalid empty scenarios before postprocessing",
            '            output = self.postprocess(output)\n\n            if output is None: continue',
            '            if output is None:\n'
            '                continue\n'
            '            output = self.postprocess(output)',
        ),
        (
            "skip timesteps where no prediction target is observed",
            '            #print("%d objects of interest at timestep %d for scene %s" % (center_objects.shape[0], current_time_index, scene_id))',
            '            if center_objects is None:\n'
            '                # Target not yet / no longer tracked at this frame (common outside the AV2 challenge data)\n'
            '                continue\n'
            '            #print("%d objects of interest at timestep %d for scene %s" % (center_objects.shape[0], current_time_index, scene_id))',
        ),
        (
            "skip timesteps where every target is dropped by the re-indexing filter",
            '            # has at least one valid target?\n'
            '            assert track_index_to_predict.shape[0] > 0',
            '            # has at least one valid target?\n'
            '            if track_index_to_predict.shape[0] == 0:\n'
            '                continue',
        ),
        (
            "return None for scenarios with no valid prediction samples",
            '        #TODO update xyh, record traj type\n'
            '        center_objects_traj_type = np.vstack(center_objects_traj_type)',
            '        if len(center_objects_final) == 0:\n'
            "            print(f'Warning: no valid prediction targets in scene {scene_id}')\n"
            '            return None\n'
            '        #TODO update xyh, record traj type\n'
            '        center_objects_traj_type = np.vstack(center_objects_traj_type)',
        ),
    ],
    'UniTraj/unitraj/datasets/dataset_types.py': [
        (
            "map nuPlan-only polyline types emitted by the ScenarioNet converter",
            "    # for speed bump\n"
            "    MetaDriveType.SPEED_BUMP: 19,\n"
            "\n"
            "}\n",
            "    # for speed bump\n"
            "    MetaDriveType.SPEED_BUMP: 19,\n"
            "\n"
            "}\n"
            "\n"
            "# The ScenarioNet nuPlan converter (metadrive main branch) emits two map types that the\n"
            "# metadrive 0.4.1.2 enum lacks: lanes through intersections (roadblock connectors), which\n"
            "# MTR should see as lanes, and walkway polygons, which have no centreline and are skipped.\n"
            "polyline_type['LANE_SURFACE_UNSTRUCTURE'] = 2\n"
            "polyline_type['ROAD_EDGE_SIDEWALK'] = 0\n",
        ),
    ],
    'UniTraj/unitraj/models/__init__.py': [
        (
            "load non-MTR models lazily (wayformer.py does not parse in this revision)",
            "from unitraj.models.autobot.autobot import AutoBotEgo\n"
            "from unitraj.models.mtr.MTR import MotionTransformer\n"
            "from unitraj.models.wayformer.wayformer import Wayformer\n"
            "\n"
            "__all__ = {\n"
            "    'autobot': AutoBotEgo,\n"
            "    'wayformer': Wayformer,\n"
            "    'MTR': MotionTransformer\n"
            "}\n",
            "from unitraj.models.mtr.MTR import MotionTransformer\n"
            "\n"
            "__all__ = {'MTR': MotionTransformer}\n"
            "\n"
            "# AutoBot and Wayformer are not needed for MTR inference, and wayformer.py in this\n"
            "# revision has a half-edited predict_step that does not parse; import them lazily.\n"
            "for _name, _module, _cls in (('autobot', 'unitraj.models.autobot.autobot', 'AutoBotEgo'),\n"
            "                             ('wayformer', 'unitraj.models.wayformer.wayformer', 'Wayformer')):\n"
            "    try:\n"
            "        __all__[_name] = getattr(__import__(_module, fromlist=[_cls]), _cls)\n"
            "    except (ImportError, SyntaxError) as _e:\n"
            "        print(f'models: {_name} unavailable ({_e.__class__.__name__}); only needed for non-MTR runs')\n",
        ),
    ],
}


def patch_file(target: Path, patches, apply: bool) -> bool:
    before = target.read_text()
    after = before
    changed = []
    for name, old, new in patches:
        if new in after:
            print(f'ALREADY APPLIED: {name}')
        elif after.count(old) == 1:
            after = after.replace(old, new, 1)
            changed.append(name)
            print(f'{"APPLY" if apply else "WILL APPLY"}: {name}')
        else:
            raise RuntimeError(f'Unexpected source revision for {name} in {target}; stopped without changes')
    if not changed:
        return False
    ast.parse(after, filename=str(target))
    if apply:
        backup = target.with_suffix(target.suffix + '.bak')
        if not backup.exists():
            shutil.copy2(target, backup)
        target.write_text(after)
        print(f'Patched {target}; backup at {backup}')
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True, type=Path, help='Root of cloned valeo4cast repository')
    p.add_argument('--apply', action='store_true', help='Actually modify sources; otherwise dry run')
    args = p.parse_args()
    for rel, patches in PATCHES.items():
        target = args.repo / rel
        if not target.is_file():
            p.error(f'Target file not found: {target}')
    any_changed = False
    for rel, patches in PATCHES.items():
        any_changed |= patch_file(args.repo / rel, patches, args.apply)
    if not any_changed:
        print('No changes needed')
    elif not args.apply:
        print('CHECK ONLY. Run again with --apply to write changes.')


if __name__ == '__main__':
    main()
