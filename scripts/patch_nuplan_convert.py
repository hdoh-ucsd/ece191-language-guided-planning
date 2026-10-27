#!/usr/bin/env python3
"""Source patches for the CPU nuPlan -> ScenarioNet conversion image.

1. nuplan-devkit: GPKGMapsDB.__init__ eagerly loads one layer from all four map packages,
   so with NUPLAN_DATA_STORE=local it crashes unless every map.gpkg (about 1.3 GB) is on
   disk. The toy setup ships only the city of the converted log; skip absent packages.

2. ScenarioNet (vendored in valeo4cast): the nuPlan converter hardcodes
   scenario_filter.remove_invalid_goals=true. The converter never reads the mission goal,
   but short mini-split logs have no goal on any scene, so the SQL filter drops every
   frame and the output is empty. Default the flag to false; NUPLAN_REMOVE_INVALID_GOALS=true
   restores upstream behaviour.

3. ScenarioNet: NUPLAN_SCENARIO_TOKENS=<tok1,tok2,...> (lidar_pc tokens, hex) converts exactly one
   scenario per token, anchored at that frame, instead of every tagged frame of the log;
   NUPLAN_TIMESTAMP_THRESHOLD_S overrides the 20 s minimum spacing between scenarios of a type.
   NUPLAN_DEFAULT_WINDOW=true extracts every scenario with the devkit default window (20 s from the
   anchor frame, no offset) instead of the per-type (duration, offset) mapping.

4. nuplan-devkit: the scenario query drops lidar frames of the first two and the last two scenes
   of a log. NUPLAN_INCLUDE_EDGE_SCENES=true keeps them, so a scenario can be anchored near a log's
   start (the caller is then responsible for the history and future it needs).

Each patch targets the pinned revisions and refuses to touch a file whose expected
fragment is not found exactly once. Safe to run repeatedly.
"""
from __future__ import annotations
import argparse
from pathlib import Path

PATCHES = {
    'devkit': [
        (
            'nuplan/database/maps_db/gpkg_mapsdb.py',
            'tolerate a single-city local map download',
            "        for location in MAP_LOCATIONS:\n"
            "            self.load_vector_layer(location, DUMMY_LOAD_LAYER)\n",
            "        for location in MAP_LOCATIONS:\n"
            "            # valeo_mtr_docker patch: tolerate a partial (single-city) local map download.\n"
            "            if isinstance(self._blob_store, LocalStore) and not os.path.exists(\n"
            "                os.path.join(self._map_root, self._get_gpkg_file_path(location))\n"
            "            ):\n"
            "                logger.warning('Map package for %s is not on disk; skipping it.', location)\n"
            "                continue\n"
            "            self.load_vector_layer(location, DUMMY_LOAD_LAYER)\n",
            ('from nuplan.database.common.blob_store.local_store import LocalStore', 'import os',
             'logger = logging.getLogger'),
        ),
        (
            'nuplan/database/nuplan_db/nuplan_scenario_queries.py',
            'import os for the edge-scene switch',
            'import pickle\nimport sqlite3\n',
            'import os\nimport pickle\nimport sqlite3\n',
            (),
        ),
        (
            'nuplan/database/nuplan_db/nuplan_scenario_queries.py',
            'NUPLAN_INCLUDE_EDGE_SCENES=true keeps lidar frames of the first two and last two scenes',
            '        INNER JOIN valid_scenes AS vs\n'
            '            ON lp.scene_token = vs.token\n',
            "        {'' if os.environ.get('NUPLAN_INCLUDE_EDGE_SCENES', '').lower() == 'true' else "
            "'INNER JOIN valid_scenes AS vs ON lp.scene_token = vs.token'}\n",
            ('WITH ordered_scenes AS',),
        ),
    ],
    'scenarionet': [
        (
            'scenarionet/converter/nuplan/utils.py',
            'keep scenarios without a mission goal unless NUPLAN_REMOVE_INVALID_GOALS=true',
            '        "scenario_filter.remove_invalid_goals=true",\n',
            '        # valeo_mtr_docker patch: the converter never uses the goal; short logs have none.\n'
            '        "scenario_filter.remove_invalid_goals={}".format(\n'
            '            os.environ.get("NUPLAN_REMOVE_INVALID_GOALS", "false").lower()),\n',
            ('import os',),
        ),
        (
            'scenarionet/converter/nuplan/utils.py',
            'name every vehicle, pedestrian and cyclist as a forecast target',
            '    result[SD.TRACKS] = extract_traffic(scenario, scenario_center)\n'
            '    result[SD.METADATA][SD.SDC_ID] = EGO\n',
            '    result[SD.TRACKS] = extract_traffic(scenario, scenario_center)\n'
            '    result[SD.METADATA][SD.SDC_ID] = EGO\n'
            '\n'
            '    # valeo_mtr_docker patch: UniTraj predicts only the ego unless the scenario names its\n'
            '    # targets, as the Argoverse 2 and Waymo converters do. Name every vehicle, pedestrian\n'
            '    # and cyclist track (ego included); the reader skips targets that are not observed at\n'
            '    # a given prediction time.\n'
            '    _track_ids = list(result[SD.TRACKS].keys())\n'
            '    result[SD.METADATA]["tracks_to_predict"] = {\n'
            '        _id: {"track_index": _track_ids.index(_id), "track_id": _id, "difficulty": 0,\n'
            '              "object_type": _track[SD.TYPE]}\n'
            '        for _id, _track in result[SD.TRACKS].items()\n'
            '        if _track[SD.TYPE] in (MetaDriveType.VEHICLE, MetaDriveType.PEDESTRIAN, MetaDriveType.CYCLIST)\n'
            '    }\n',
            ('MetaDriveType', 'SD.TYPE'),
        ),
        (
            'scenarionet/converter/nuplan/utils.py',
            'optional scenario_tokens / timestamp_threshold_s overrides from the environment',
            '        "scenario_filter.timestamp_threshold_s=20",  # minial scenario duration (s)\n',
            '        # valeo_mtr_docker patch: NUPLAN_SCENARIO_TOKENS (comma-separated lidar_pc tokens) extracts\n'
            '        # one scenario per token instead of every tagged frame; NUPLAN_TIMESTAMP_THRESHOLD_S is the\n'
            '        # minimum spacing between scenarios of one type (upstream: 20).\n'
            '        "scenario_filter.timestamp_threshold_s={}".format(\n'
            '            os.environ.get("NUPLAN_TIMESTAMP_THRESHOLD_S", "20")),\n'
            '        *(["scenario_filter.scenario_tokens=[{}]".format(os.environ["NUPLAN_SCENARIO_TOKENS"])]\n'
            '          if os.environ.get("NUPLAN_SCENARIO_TOKENS") else []),\n',
            ('import os',),
        ),
        (
            'scenarionet/converter/nuplan/utils.py',
            'NUPLAN_DEFAULT_WINDOW=true gives every scenario the default window (20 s from its anchor frame)',
            '    scenario_builder = build_scenario_builder(cfg=cfg)\n',
            '    scenario_builder = build_scenario_builder(cfg=cfg)\n'
            '    # valeo_mtr_docker patch: with NUPLAN_DEFAULT_WINDOW=true every scenario, whatever its tag, is\n'
            '    # extracted with the devkit defaults (20 s, no offset) from its anchor lidar frame, so that a\n'
            '    # scenario requested through NUPLAN_SCENARIO_TOKENS starts exactly at the given frame.\n'
            '    if os.environ.get("NUPLAN_DEFAULT_WINDOW", "").lower() == "true":\n'
            '        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping\n'
            '        scenario_builder._scenario_mapping = ScenarioMapping(\n'
            '            scenario_map={}, subsample_ratio_override=scenario_builder._scenario_mapping.subsample_ratio_override)\n',
            ('import os', 'build_scenario_filter(cfg.scenario_filter)'),
        ),
    ],
}


def apply(root: Path, patches) -> None:
    for rel, name, old, new, needed in patches:
        path = root / rel
        text = path.read_text()
        if new in text:
            print(f'already patched ({name}): {path}')
            continue
        if text.count(old) != 1:
            raise SystemExit(f'{name}: expected fragment not found exactly once in {path}; refusing to patch')
        for fragment in needed:
            if fragment not in text:
                raise SystemExit(f'{name}: {path} lacks {fragment!r}; patch would not be valid')
        path.write_text(text.replace(old, new))
        print(f'patched ({name}): {path}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--devkit', type=Path, help='nuplan-devkit checkout')
    p.add_argument('--scenarionet', type=Path, help='ScenarioNet checkout (valeo4cast/scenarionet)')
    args = p.parse_args()
    if not (args.devkit or args.scenarionet):
        p.error('give --devkit and/or --scenarionet')
    if args.devkit:
        apply(args.devkit, PATCHES['devkit'])
    if args.scenarionet:
        apply(args.scenarionet, PATCHES['scenarionet'])


if __name__ == '__main__':
    main()
